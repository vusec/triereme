// This file is part of SymCC.
//
// SymCC is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// SymCC is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// SymCC. If not, see <https://www.gnu.org/licenses/>.

mod afl_input;
mod fault_scheduler;
mod symcc;

use afl_input::AflCompatInput;
use anyhow::{Context, Result};
use fault_scheduler::FaultCorpusScheduler;
use flate2::write::GzEncoder;
use flate2::Compression;
use libafl::bolts::current_nanos;
use libafl::bolts::rands::StdRand;
use libafl::bolts::shmem::{ShMem, ShMemProvider, StdShMemProvider};
use libafl::bolts::tuples::tuple_list;
use libafl::corpus::OnDiskCorpus;
use libafl::events::NopEventManager;
use libafl::executors::{ForkserverExecutor, TimeoutForkserverExecutor};
use libafl::feedbacks::{CrashFeedback, MapFeedbackState, MaxMapFeedback};
use libafl::observers::{ConstMapObserver, HitcountsMapObserver};
use libafl::state::StdState;
use libafl::{feedback_and_fast, Evaluator, ExecuteInputResult, StdFuzzer};
use runtime::runtimes::analysis::{self, AnalysisSettings};
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use symcc::{AflConfig, SymCC, TestcaseDir};
use tracing_subscriber::util::SubscriberInitExt;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

const STATS_INTERVAL_SEC: u64 = 60;

// TODO extend timeout when idle? Possibly reprocess previously timed-out
// inputs.

use structopt::clap::arg_enum;
arg_enum! {
    #[derive(Debug)]
    enum PathFilteringMode {
        None,
        Factorizing,
        Concretizing,
        Combined,
    }
}

impl PathFilteringMode {
    fn to_analysis_mode(&self) -> analysis::PathFilteringMode {
        match self {
            PathFilteringMode::None => analysis::PathFilteringMode::None,
            PathFilteringMode::Factorizing => analysis::PathFilteringMode::Factorization,
            PathFilteringMode::Concretizing => analysis::PathFilteringMode::Concretization,
            PathFilteringMode::Combined => analysis::PathFilteringMode::Combined,
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(about = "Make SymCC collaborate with AFL.", no_version)]
struct CLI {
    /// The name of the fuzzer to work with
    #[structopt(short = "a")]
    fuzzer_name: String,

    /// The AFL output directory
    #[structopt(short = "o")]
    output_dir: PathBuf,

    /// Name to use for SymCC
    #[structopt(short = "n")]
    name: String,

    /// Enable verbose logging
    #[structopt(short = "v")]
    verbose: bool,

    /// Whether divs should be concretized or solved
    #[structopt(long, env)]
    concretize_divs: bool,

    /// Path filtering mode
    #[structopt(long, env)]
    path_filtering_mode: PathFilteringMode,

    /// Whether to reset the solver trie after each input
    #[structopt(long, env)]
    no_keep_trie: bool,

    /// Whether to use linear solving instead of the trie
    #[structopt(long, env)]
    linear_solving: bool,

    /// Whether to use the very optimistic mode
    #[structopt(long, env)]
    very_optimistic: bool,

    /// Whether to use 'optimistic mode' as implemented in QSYM
    #[structopt(long, env)]
    optimistic_unsolved: bool,

    /// Whether to use 'optimistic mode', but for pruned instead of unsolved constraints.
    #[structopt(long, env)]
    optimistic_pruned: bool,

    /// Maximum age of nodes in the trie cache
    #[structopt(long, env, default_value = "10")]
    max_node_age: usize,

    /// Disable dynamic timeouts, use QSYM default
    #[structopt(long, env)]
    no_dyn_timeout: bool,

    /// AFL++ fork-mode binary for filtering
    #[structopt(short = "f")]
    aflpp_executable: Option<PathBuf>,

    /// Program under test
    command: Vec<String>,
}

/// Execution statistics.
struct Stats {
    /// Start of the campaign
    campaign_start: Instant,

    /// Time spent waiting for new test cases to process.
    idle_time: Duration,

    /// Time spent in successful executions of SymCC.
    success_time: Duration,

    /// Time spent in the solver as part of successfully running SymCC.
    success_solver_time: Duration,

    /// Time spent in failed SymCC executions.
    failed_time: Duration,

    /// Total number of executions
    total_execs: u64,

    /// Number of successful executions.
    success_execs: u64,

    /// Files that still need to be processed.
    pending_files: u64,
}

#[derive(serde::Serialize)]
struct StatsRecord {
    relative_time_s: u64,
    idle_time_s: u64,
    success_time_s: u64,
    success_solver_time_s: u64,
    success_solver_share: f64,
    failed_time_s: u64,
    total_execs: u64,
    success_execs: u64,
    avg_success_time_ms: Option<u64>,
    avg_success_solver_time_ms: Option<u64>,
    avg_failed_time_ms: Option<u64>,
    pending_files: u64,
}

impl Stats {
    fn new() -> Self {
        Stats {
            campaign_start: Instant::now(),
            idle_time: Duration::ZERO,
            total_execs: 0,
            success_execs: 0,
            success_time: Duration::ZERO,
            success_solver_time: Duration::ZERO,
            failed_time: Duration::ZERO,
            pending_files: 0,
        }
    }

    fn add_execution(&mut self, result: &symcc::SymCCResult) {
        self.total_execs += 1;

        if !result.killed {
            self.success_execs += 1;
            self.success_time += result.time;
            self.success_solver_time += result.solver_time.unwrap_or_default();
        } else {
            self.failed_time += result.time;
        }
    }

    fn log(&self, out: &mut csv::Writer<File>) -> Result<()> {
        let avg_success_time_ms = if self.success_execs > 0 {
            Some((self.success_time / self.success_execs as u32).as_millis() as u64)
        } else {
            None
        };

        let avg_success_solver_time_ms = if self.success_execs > 0 {
            Some((self.success_solver_time / self.success_execs as u32).as_millis() as u64)
        } else {
            None
        };

        let failed_execs = self.total_execs - self.success_execs;
        let avg_failed_time_ms = if failed_execs > 0 {
            Some((self.failed_time / failed_execs as u32).as_millis() as u64)
        } else {
            None
        };

        let success_solver_share =
            self.success_solver_time.as_secs_f64() / self.success_time.as_secs_f64();

        out.serialize(StatsRecord {
            relative_time_s: self.campaign_start.elapsed().as_secs(),
            idle_time_s: self.idle_time.as_secs(),
            success_time_s: self.success_time.as_secs(),
            success_solver_time_s: self.success_solver_time.as_secs(),
            success_solver_share,
            failed_time_s: self.failed_time.as_secs(),
            total_execs: self.total_execs,
            success_execs: self.success_execs,
            avg_success_time_ms,
            avg_success_solver_time_ms,
            avg_failed_time_ms,
            pending_files: self.pending_files,
        })?;

        out.flush()?;

        Ok(())
    }
}

/// Mutable run-time state.
///
/// This is a collection of the state we update during execution.
struct State<S> {
    /// The AFL test cases that have been analyzed so far.
    processed_files: HashSet<PathBuf>,

    /// The place for new test cases that time out.
    hangs: TestcaseDir,

    /// Run-time statistics.
    stats: Stats,

    /// When did we last output the statistics?
    last_stats_output: Instant,

    /// Write statistics to this file.
    stats_file: csv::Writer<File>,

    /// State used by the evaluator
    fuzzer_state: S,
}

impl<S> State<S>
where
    S: libafl::state::State,
{
    /// Initialize the run-time environment in the given output directory.
    ///
    /// This involves creating the output directory and all required
    /// subdirectories.
    fn initialize(output_dir: impl AsRef<Path>, fuzzer_state: S) -> Result<Self> {
        let symcc_dir = output_dir.as_ref();
        let symcc_hangs = TestcaseDir::new(symcc_dir.join("hangs"))?;
        let stats_file = csv::Writer::from_path(symcc_dir.join("stats"))?;

        Ok(State {
            processed_files: HashSet::new(),
            hangs: symcc_hangs,
            stats: Stats::new(),
            last_stats_output: Instant::now(),
            stats_file,
            fuzzer_state,
        })
    }

    fn test_input<EV, E, EM>(
        &mut self,
        parent: impl AsRef<Path>,
        symcc: &mut SymCC,
        evaluator: &mut EV,
        executor: &mut E,
        manager: &mut EM,
    ) -> Result<()>
    where
        EV: Evaluator<E, EM, AflCompatInput, S>,
    {
        log::info!("Running on input {}", parent.as_ref().display());

        // Import parent test case in queue and update coverage map. Run SymCC
        // on it even if not interesting because we are not mutating our own
        // test cases, only those coming from the fuzzer instance.
        let parent_bytes =
            AflCompatInput::new(fs::read(&parent).context("Failed to read parent test case")?);
        evaluator
            .evaluate_input(&mut self.fuzzer_state, executor, manager, parent_bytes)
            .expect("Failed to run parent test case");

        let symcc_result = symcc.run(&parent).context("Failed to run SymCC")?;

        let mut num_interesting = 0u64;
        let mut num_total = 0u64;

        let start = Instant::now();
        for solution in &symcc_result.test_cases {
            let child = AflCompatInput::new(solution.to_vec());

            let result = evaluator.evaluate_input(&mut self.fuzzer_state, executor, manager, child);
            let result = match result {
                Ok((result, _)) => result,
                Err(error) => {
                    log::warn!("unable to process test case: {}", error);
                    continue;
                }
            };

            num_total += 1;
            if result == ExecuteInputResult::Corpus {
                log::debug!("Test case is interesting");
                num_interesting += 1;
            }
        }
        let export_time = start.elapsed();

        tracing::info!(
            export_time = %export_time.as_secs_f64(),
            "export time"
        );

        log::info!(
            "Generated {} test cases ({} new)",
            num_total,
            num_interesting
        );

        if symcc_result.killed {
            log::info!(
                "The target process was killed (probably timeout or out of memory); \
                 archiving to {}",
                self.hangs.path.display()
            );
            symcc::copy_testcase(
                &parent,
                &mut self.hangs,
                &parent,
                self.stats.campaign_start.elapsed().as_secs(),
            )
            .context("Failed to archive the test case")?;
        }

        self.processed_files.insert(parent.as_ref().to_path_buf());
        self.stats.add_execution(&symcc_result);

        Ok(())
    }
}

fn setup_ctrlc_handler() -> Result<Receiver<()>> {
    let (sender, receiver) = channel();
    ctrlc::set_handler(move || {
        eprintln!("Received exit signal.");
        sender.send(()).expect("Could not send exit message");
    })?;

    Ok(receiver)
}

fn main() -> Result<()> {
    std::thread::spawn(|| {
        use jemalloc_ctl::*;
        let e = epoch::mib().unwrap();
        let allocated = stats::allocated::mib().unwrap();
        let resident = stats::resident::mib().unwrap();
        loop {
            // many statistics are cached and only updated when the epoch is advanced.
            e.advance().unwrap();

            let allocated = allocated.read().unwrap();
            let resident = resident.read().unwrap();
            let z3_bytes = unsafe { z3_sys::Z3_get_estimated_alloc_size() };
            println!(
                "{} Mb allocated/{} Mb resident/{} MB Z3",
                allocated / 1024 / 1024,
                resident / 1024 / 1024,
                z3_bytes / 1024 / 1024,
            );
            tracing::info!(rust = allocated, z3 = z3_bytes, "memory usage");
            std::thread::sleep(Duration::from_secs(3));
        }
    });

    let options = CLI::from_args();

    if !options.output_dir.is_dir() {
        log::error!(
            "The directory {} does not exist!",
            options.output_dir.display()
        );
        return Ok(());
    }

    let afl_queue = options.output_dir.join(&options.fuzzer_name).join("queue");
    if !afl_queue.is_dir() {
        log::error!("The AFL queue {} does not exist!", afl_queue.display());
        return Ok(());
    }

    if let Some(aflpp_executable) = options.aflpp_executable.as_ref() {
        if !aflpp_executable.is_file() {
            log::error!(
                "AFL++ fork-mode binary is invalid: {}",
                aflpp_executable.display()
            );
            return Ok(());
        }
    }

    let symcc_dir = options.output_dir.join(&options.name);
    if symcc_dir.is_dir() {
        log::error!(
            "{} already exists; we do not currently support resuming",
            symcc_dir.display()
        );
        return Ok(());
    }

    fs::create_dir_all(&symcc_dir)
        .with_context(|| format!("Failed to create SymCC's directory {}", symcc_dir.display()))?;

    let writer = GzEncoder::new(
        BufWriter::new(fs::File::create(symcc_dir.join("trace.json.gz"))?),
        Compression::best(),
    );
    let (non_blocking, _guard) = tracing_appender::non_blocking(writer);
    tracing_subscriber::fmt()
        .json()
        .with_writer(non_blocking)
        .finish()
        .init();

    let mut symcc = SymCC::new(
        symcc_dir.clone(),
        &options.command,
        AnalysisSettings {
            keep_trie: !options.no_keep_trie,
            solve_division: !options.concretize_divs,
            path_filtering_mode: options.path_filtering_mode.to_analysis_mode(),
            linear_solving: options.linear_solving,
            very_optimistic: options.very_optimistic,
            optimistic_unsolved: options.optimistic_unsolved,
            optimistic_pruned: options.optimistic_pruned,
            max_node_age: options.max_node_age,
            dyn_timeout: !options.no_dyn_timeout,
        },
    );
    let afl_config = AflConfig::load(options.output_dir.join(&options.fuzzer_name))?;
    log::debug!("AFL configuration: {:?}", &afl_config);

    const MAP_SIZE: usize = 65536;
    let mut shmem_provider = StdShMemProvider::new().unwrap();
    let mut shmem = shmem_provider.new_map(MAP_SIZE).unwrap();
    shmem.write_to_env("__AFL_SHM_ID").unwrap();

    let edges_observer = HitcountsMapObserver::new(ConstMapObserver::<_, MAP_SIZE>::new(
        "shared_mem",
        shmem.map_mut(),
    ));

    let feedback_state = MapFeedbackState::with_observer(&edges_observer);
    let feedback = MaxMapFeedback::new_tracking(&feedback_state, &edges_observer, true, false);

    let objective_state = MapFeedbackState::new("crash_edges", MAP_SIZE);
    let objective = feedback_and_fast!(
        CrashFeedback::new(),
        MaxMapFeedback::new(&objective_state, &edges_observer)
    );

    let mut fuzzer = StdFuzzer::new(FaultCorpusScheduler, feedback, objective);

    let fuzzer_state = StdState::new(
        StdRand::with_seed(current_nanos()),
        OnDiskCorpus::new(symcc_dir.join("queue")).unwrap(),
        OnDiskCorpus::new(symcc_dir.join("crashes")).unwrap(),
        tuple_list!(feedback_state, objective_state),
    );
    let mut state = State::initialize(symcc_dir, fuzzer_state)?;

    let mut manager = NopEventManager {};

    let aflpp_executable = options
        .aflpp_executable
        .as_ref()
        .map_or(afl_config.get_target().to_string_lossy().to_string(), |x| {
            x.to_string_lossy().into()
        });
    let arguments = if options.aflpp_executable.is_some() {
        options.command.iter().skip(1).cloned().collect::<Vec<_>>()
    } else {
        afl_config
            .get_args()
            .iter()
            .map(|os_string| os_string.to_string_lossy().to_string())
            .collect::<Vec<_>>()
    };
    let mut executor = TimeoutForkserverExecutor::new(
        ForkserverExecutor::new(
            aflpp_executable,
            &arguments,
            false,
            tuple_list!(edges_observer),
        )
        .expect("Failed to create executor"),
        Duration::from_millis(5000),
    )
    .expect("Failed to create the timeout executor");

    let ctrlc_receiver = setup_ctrlc_handler()?;
    while let Err(TryRecvError::Empty) = ctrlc_receiver.try_recv() {
        let (best_new_testcase, num_candidates) = afl_config
            .best_new_testcase(&state.processed_files)
            .context("Failed to check for new test cases")?;

        state.stats.pending_files = num_candidates;

        if let Some(input) = best_new_testcase {
            let test_input_start = Instant::now();
            state.test_input(&input, &mut symcc, &mut fuzzer, &mut executor, &mut manager)?;
            tracing::info!(
                input = %input.display(),
                test_input_time_us = %test_input_start.elapsed().as_micros(),
                "test input time")
        } else {
            log::debug!("Waiting for new test cases...");
            let wait_time = Duration::from_secs(5);
            state.stats.idle_time += wait_time;
            thread::sleep(wait_time);
        }

        if state.last_stats_output.elapsed().as_secs() > STATS_INTERVAL_SEC {
            if let Err(e) = state.stats.log(&mut state.stats_file) {
                log::error!("Failed to log run-time statistics: {}", e);
            }
            state.last_stats_output = Instant::now();
        }
    }

    eprintln!("Exiting.");

    Ok(())
}
