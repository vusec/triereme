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

use anyhow::{bail, ensure, Context, Result};
use libafl::{
    bolts::shmem::{ShMem, ShMemProvider, StdShMemProvider},
    observers::concolic::serialization_format::{MessageFileReader, DEFAULT_ENV_NAME},
};
use regex::Regex;

use std::{
    cmp,
    collections::HashSet,
    ffi::{OsStr, OsString},
    fs::{self, File},
    io::{self, Read},
    ops::Deref,
    os::unix::process::ExitStatusExt,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str,
    time::{Duration, Instant},
};

use runtime::{
    replay::replay_trace_hl,
    runtimes::analysis::{AnalysisRuntime, AnalysisSettings, SolverSolution},
};

const TIMEOUT: u32 = 90;

/// Replace the first '@@' in the given command line with the input file.
fn insert_input_file<S: AsRef<OsStr>, P: AsRef<Path>>(
    command: &[S],
    input_file: P,
) -> Vec<OsString> {
    let mut fixed_command: Vec<OsString> = command.iter().map(|s| s.into()).collect();
    if let Some(at_signs) = fixed_command.iter_mut().find(|s| *s == "@@") {
        *at_signs = input_file.as_ref().as_os_str().to_os_string();
    }

    fixed_command
}

/// Score of a test case.
///
/// We use the lexical comparison implemented by the derived implementation of
/// Ord in order to compare according to various criteria.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
struct TestcaseScore {
    /// First criterion: new coverage
    new_coverage: bool,

    /// Second criterion: being derived from seed inputs
    derived_from_seed: bool,

    /// Third criterion: size (smaller is better)
    file_size: i128,

    /// Fourth criterion: name (containing the ID)
    base_name: OsString,
}

impl TestcaseScore {
    /// Score a test case.
    ///
    /// If anything goes wrong, return the minimum score.
    fn new(t: impl AsRef<Path>) -> Self {
        let size = match fs::metadata(&t) {
            Err(e) => {
                // Has the file disappeared?
                log::warn!(
                    "Warning: failed to score test case {}: {}",
                    t.as_ref().display(),
                    e
                );

                return TestcaseScore::minimum();
            }
            Ok(meta) => meta.len(),
        };

        let name: OsString = match t.as_ref().file_name() {
            None => return TestcaseScore::minimum(),
            Some(n) => n.to_os_string(),
        };
        let name_string = name.to_string_lossy();

        TestcaseScore {
            new_coverage: name_string.ends_with("+cov"),
            derived_from_seed: name_string.contains("orig:"),
            file_size: -i128::from(size),
            base_name: name,
        }
    }

    /// Return the smallest possible score.
    fn minimum() -> TestcaseScore {
        TestcaseScore {
            new_coverage: false,
            derived_from_seed: false,
            file_size: std::i128::MIN,
            base_name: OsString::from(""),
        }
    }
}

/// A directory that we can write test cases to.
pub struct TestcaseDir {
    /// The path to the (existing) directory.
    pub path: PathBuf,
    /// The next free ID in this directory.
    current_id: u64,
}

impl TestcaseDir {
    /// Create a new test-case directory in the specified location.
    ///
    /// The parent directory must exist.
    pub fn new(path: impl AsRef<Path>) -> Result<TestcaseDir> {
        let dir = TestcaseDir {
            path: path.as_ref().into(),
            current_id: 0,
        };

        fs::create_dir(&dir.path)
            .with_context(|| format!("Failed to create directory {}", dir.path.display()))?;
        Ok(dir)
    }
}

/// Copy a test case to a directory, using the parent test case's name to derive
/// the new name.
pub fn copy_testcase(
    testcase: impl AsRef<Path>,
    target_dir: &mut TestcaseDir,
    parent: impl AsRef<Path>,
    relative_time: u64,
) -> Result<()> {
    let orig_name = parent
        .as_ref()
        .file_name()
        .expect("The input file does not have a name")
        .to_string_lossy();
    ensure!(
        orig_name.starts_with("id:"),
        "The name of test case {} does not start with an ID",
        parent.as_ref().display()
    );

    if let Some(orig_id) = orig_name.get(3..9) {
        let new_name = format!(
            "id:{:06},src:{},time:{}",
            target_dir.current_id, &orig_id, relative_time
        );
        let target = target_dir.path.join(new_name);
        log::debug!("Creating test case {}", target.display());
        fs::copy(testcase.as_ref(), target).with_context(|| {
            format!(
                "Failed to copy the test case {} to {}",
                testcase.as_ref().display(),
                target_dir.path.display()
            )
        })?;

        target_dir.current_id += 1;
    } else {
        bail!(
            "Test case {} does not contain a proper ID",
            parent.as_ref().display()
        );
    }

    Ok(())
}

/// Information on the run-time environment.
///
/// This should not change during execution.
#[derive(Debug)]
pub struct AflConfig {
    /// The command that AFL uses to invoke the target program.
    target: OsString,

    /// The args that AFL uses to invoke the target program.
    target_args: Vec<OsString>,

    /// The fuzzer instance's queue of test cases.
    queue: PathBuf,
}

impl AflConfig {
    /// Read the AFL configuration from a fuzzer instance's output directory.
    pub fn load(fuzzer_output: impl AsRef<Path>) -> Result<Self> {
        let afl_stats_file_path = fuzzer_output.as_ref().join("fuzzer_stats");
        let mut afl_stats_file = File::open(&afl_stats_file_path).with_context(|| {
            format!(
                "Failed to open the fuzzer's stats at {}",
                afl_stats_file_path.display()
            )
        })?;
        let mut afl_stats = String::new();
        afl_stats_file
            .read_to_string(&mut afl_stats)
            .with_context(|| {
                format!(
                    "Failed to read the fuzzer's stats at {}",
                    afl_stats_file_path.display()
                )
            })?;
        let afl_command: Vec<_> = afl_stats
            .lines()
            .find(|&l| l.starts_with("command_line"))
            .expect("The fuzzer stats don't contain the command line")
            .splitn(2, ':')
            .nth(1)
            .expect("The fuzzer stats follow an unknown format")
            .trim()
            .split_whitespace()
            .collect();

        let mut afl_target_command: Vec<_> = afl_command
            .iter()
            .skip_while(|s| **s != "--")
            .skip(1)
            .map(OsString::from)
            .collect();
        let target_args = afl_target_command.split_off(1);

        Ok(AflConfig {
            target: afl_target_command.pop().unwrap(),
            target_args,
            queue: fuzzer_output.as_ref().join("queue"),
        })
    }

    /// Return the most promising unseen test case of this fuzzer.
    pub fn best_new_testcase(&self, seen: &HashSet<PathBuf>) -> Result<(Option<PathBuf>, u64)> {
        let candidates: Vec<_> = fs::read_dir(&self.queue)
            .with_context(|| {
                format!(
                    "Failed to open the fuzzer's queue at {}",
                    self.queue.display()
                )
            })?
            .collect::<io::Result<Vec<_>>>()
            .with_context(|| {
                format!(
                    "Failed to read the fuzzer's queue at {}",
                    self.queue.display()
                )
            })?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && !seen.contains(path))
            .collect();

        let num_candidates = candidates.len() as u64;

        let best = candidates
            .into_iter()
            .max_by_key(|path| TestcaseScore::new(path));

        Ok((best, num_candidates))
    }

    pub fn get_target(&self) -> &OsString {
        &self.target
    }

    pub fn get_args(&self) -> &[OsString] {
        &self.target_args
    }
}

/// The run-time configuration of SymCC.
pub struct SymCC {
    /// Do we pass data to standard input?
    use_standard_input: bool,

    /// The cumulative bitmap for branch pruning.
    bitmap: PathBuf,

    /// The place to store the current input.
    input_file: PathBuf,

    /// The command to run.
    command: Vec<OsString>,

    /// The shared memory mapping used for concolic tracing.
    concolic_shmem: <StdShMemProvider as ShMemProvider>::Mem,

    /// The solving runtime
    solving_runtime: AnalysisRuntime,

    /// How many files we have processed
    files_processed: usize,

    /// In-memory bitmap for branch pruning
    analysis_bitmap: Vec<u8>,

    /// Analysis settings for runtime reconstruction in case it dies
    analysis_setting: AnalysisSettings,
}

/// The result of executing SymCC.
pub struct SymCCResult {
    /// The generated test cases.
    pub test_cases: Vec<SolverSolution>,
    /// Whether the process was killed (e.g., out of memory, timeout).
    pub killed: bool,
    /// The total time taken by the execution.
    pub time: Duration,
    /// The time spent in the solver (Qsym backend only).
    pub solver_time: Option<Duration>,
}

impl SymCC {
    /// Create a new SymCC configuration.
    pub fn new(
        output_dir: PathBuf,
        command: &[String],
        analysis_setting: AnalysisSettings,
    ) -> Self {
        let input_file = output_dir.join(".cur_input");

        let concolic_shmem = setup_shmem();
        let solving_runtime = AnalysisRuntime::new(analysis_setting);
        SymCC {
            use_standard_input: !command.contains(&String::from("@@")),
            bitmap: output_dir.join("bitmap"),
            command: insert_input_file(command, &input_file),
            input_file,
            concolic_shmem,
            solving_runtime,
            files_processed: 0,
            analysis_bitmap: Vec::new(),
            analysis_setting,
        }
    }

    /// Try to extract the solver time from the logs produced by the Qsym
    /// backend.
    fn parse_solver_time(output: Vec<u8>) -> Option<Duration> {
        let re = Regex::new(r#""solving_time": (\d+)"#).unwrap();
        output
            // split into lines
            .rsplit(|n| *n == b'\n')
            // convert to string
            .filter_map(|s| str::from_utf8(s).ok())
            // check that it's an SMT log line
            .filter(|s| s.trim_start().starts_with("[STAT] SMT:"))
            // find the solving_time element
            .filter_map(|s| re.captures(s))
            // convert the time to an integer
            .filter_map(|c| c[1].parse().ok())
            // associate the integer with a unit
            .map(Duration::from_micros)
            // get the first one
            .next()
    }

    /// Run SymCC on the given input, writing results to the provided temporary
    /// directory.
    ///
    /// If SymCC is run with the Qsym backend, this function attempts to
    /// determine the time spent in the SMT solver and report it as part of the
    /// result. However, the mechanism that the backend uses to report solver
    /// time is somewhat brittle.
    pub fn run(&mut self, input: impl AsRef<Path>) -> Result<SymCCResult> {
        fs::copy(&input, &self.input_file).with_context(|| {
            format!(
                "Failed to copy the test case {} to our workbench at {}",
                input.as_ref().display(),
                self.input_file.display()
            )
        })?;

        let mut analysis_command = Command::new("timeout");
        analysis_command
            .args(&["-k", "5", &TIMEOUT.to_string()])
            .args(&self.command)
            .env("SYMCC_ENABLE_LINEARIZATION", "1")
            .env("SYMCC_AFL_COVERAGE_MAP", &self.bitmap)
            .stdout(Stdio::null())
            .stderr(Stdio::piped()); // capture SMT logs

        if self.use_standard_input {
            analysis_command.stdin(Stdio::piped());
        } else {
            analysis_command.stdin(Stdio::null());
            analysis_command.env("SYMCC_INPUT_FILE", &self.input_file);
        }

        log::debug!("Running SymCC as follows: {:?}", &analysis_command);
        let start = Instant::now();
        let mut child = analysis_command.spawn().context("Failed to run SymCC")?;

        if self.use_standard_input {
            io::copy(
                &mut File::open(&self.input_file).with_context(|| {
                    format!(
                        "Failed to read the test input at {}",
                        self.input_file.display()
                    )
                })?,
                child
                    .stdin
                    .as_mut()
                    .expect("Failed to pipe to the child's standard input"),
            )
            .context("Failed to pipe the test input to SymCC")?;
        }

        let result = child
            .wait_with_output()
            .context("Failed to wait for SymCC")?;
        let total_time = start.elapsed();
        tracing::info!(
            tracing_time = %total_time.as_secs_f64(),
            "trace timing"
        );
        let killed = match result.status.code() {
            Some(code) => {
                log::debug!("SymCC returned code {}", code);
                (code == 124) || (code == -9) // as per the man-page of timeout
            }
            None => {
                let maybe_sig = result.status.signal();
                if let Some(signal) = maybe_sig {
                    log::warn!("SymCC received signal {}", signal);
                }
                maybe_sig.is_some()
            }
        };

        let new_tests = match std::panic::catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<_> {
            self.solving_runtime
                .receive_input(std::fs::read(&self.input_file)?);

            let start = Instant::now();
            {
                let mut reader =
                    MessageFileReader::from_length_prefixed_buffer(self.concolic_shmem.map())
                        .context("unable to create trace reader")?;
                replay_trace_hl(&mut self.solving_runtime, &mut reader);
            }
            let replay_time = start.elapsed();

            tracing::info!(
                replay_time = %replay_time.as_secs_f64(),
                "replay time"
            );

            let start = Instant::now();
            self.solving_runtime
                .solve(Duration::from_secs(TIMEOUT.into()));
            let solve_time = start.elapsed();

            tracing::info!(
                solve_time = %solve_time.as_secs_f64(),
                "total solve time"
            );

            Ok(self.solving_runtime.finish())
        })) {
            Ok(res) => {
                self.analysis_bitmap.clear();
                self.solving_runtime
                    .export_state(&mut self.analysis_bitmap)
                    .context("failed to export runtime state")?;
                res?
            }
            Err(panic) => {
                let cause = panic
                    .downcast_ref::<String>()
                    .map(String::deref)
                    .unwrap_or("<unknown cause>");
                log::warn!("Backend panic'd! {:?}", cause);
                self.solving_runtime = AnalysisRuntime::new(self.analysis_setting);
                if !self.analysis_bitmap.is_empty() {
                    self.solving_runtime
                        .import_state(self.analysis_bitmap.as_slice())
                        .context("failed to re-import runtime state after panic")?;
                } else {
                    log::warn!("Runtime panic'd and we don't have a bitmap to recover from");
                }
                vec![]
            }
        };

        let solver_time = SymCC::parse_solver_time(result.stderr);
        if solver_time.is_some() && solver_time.unwrap() > total_time {
            log::warn!("Backend reported inaccurate solver time!");
        }

        self.files_processed += 1;

        Ok(SymCCResult {
            test_cases: new_tests,
            killed,
            time: total_time,
            solver_time: solver_time.map(|t| cmp::min(t, total_time)),
        })
    }
}

fn setup_shmem() -> <StdShMemProvider as ShMemProvider>::Mem {
    let mut shmemprovider = StdShMemProvider::default();
    let concolic_shmem = shmemprovider
        .new_map(1024 * 1024 * 100) // typical traces are < 10 MB
        .expect("unable to create shared mapping");
    concolic_shmem
        .write_to_env(DEFAULT_ENV_NAME)
        .expect("unable to write shared mapping info to environment");
    concolic_shmem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_ordering() {
        let min_score = TestcaseScore::minimum();
        assert!(
            TestcaseScore {
                new_coverage: true,
                ..TestcaseScore::minimum()
            } > min_score
        );
        assert!(
            TestcaseScore {
                derived_from_seed: true,
                ..TestcaseScore::minimum()
            } > min_score
        );
        assert!(
            TestcaseScore {
                file_size: -4,
                ..TestcaseScore::minimum()
            } > min_score
        );
        assert!(
            TestcaseScore {
                base_name: OsString::from("foo"),
                ..TestcaseScore::minimum()
            } > min_score
        );
    }

    #[test]
    fn test_solver_time_parsing() {
        let output = r#"[INFO] New testcase: /tmp/output/000005
[STAT] SMT: { "solving_time": 14539, "total_time": 185091 }
[STAT] SMT: { "solving_time": 14869 }
[STAT] SMT: { "solving_time": 14869, "total_time": 185742 }
[STAT] SMT: { "solving_time": 15106 }"#;

        assert_eq!(
            SymCC::parse_solver_time(output.as_bytes().to_vec()),
            Some(Duration::from_micros(15106))
        );
        assert_eq!(
            SymCC::parse_solver_time("whatever".as_bytes().to_vec()),
            None
        );
    }
}
