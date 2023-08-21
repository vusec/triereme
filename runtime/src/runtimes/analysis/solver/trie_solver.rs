use crate::runtimes::analysis::{
    constraint_language::ExprWrapper, expression_pool::ExprRef, GLOBAL_SOLVER_TIMEOUT,
};

use self::trie::{Trie, TrieVisitorStep};

use super::{get_model_assignment, optimistic_solver::OptimisticSolver, Solver, SolverSolution};

mod trie;

use std::{
    cell::{Cell, RefCell},
    convert::TryInto,
    hash::Hash,
    rc::Rc,
    time::{Duration, Instant},
};

use hdrhistogram::Histogram;

#[derive(Debug, Clone)]
enum SatResult {
    Unsat,
    Sat(Rc<Vec<(usize, u8)>>),
    Other,
}

#[derive(Debug, Clone)]
enum SolvedState {
    Unsolved,
    Solved(SatResult),
}

impl Default for SolvedState {
    fn default() -> Self {
        Self::Unsolved
    }
}

#[derive(Debug, Clone)]
struct KeyElement {
    expr: ExprRef<ExprWrapper>,
    solved_state: RefCell<SolvedState>,
    simplification_result: Cell<Option<Option<bool>>>,
    last_used: Cell<usize>,
}

impl KeyElement {
    pub fn new(expr: ExprRef<ExprWrapper>, current_iteration: usize) -> Self {
        Self {
            expr,
            solved_state: RefCell::default(),
            simplification_result: Cell::default(),
            last_used: Cell::new(current_iteration),
        }
    }

    pub fn simplification_result(&self, ctx: &z3::Context) -> Option<bool> {
        if let Some(res) = self.simplification_result.get() {
            res
        } else {
            let res = self
                .expr
                .to_z3_simplified(ctx, true)
                .as_bool()
                .unwrap()
                .as_bool();
            self.simplification_result.set(Some(res));
            res
        }
    }

    pub fn precomputed_simplification_result(&self) -> Option<Option<bool>> {
        self.simplification_result.get()
    }

    pub fn mark_known_unsat(&self) {
        *self.solved_state.borrow_mut() = SolvedState::Solved(SatResult::Unsat);
    }
}

impl PartialEq for KeyElement {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

impl Eq for KeyElement {}

impl Hash for KeyElement {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.expr.hash(state);
    }
}

impl PartialOrd for KeyElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.expr.partial_cmp(&other.expr)
    }
}

impl Ord for KeyElement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.expr.cmp(&other.expr)
    }
}

#[derive(Debug, Clone)]
struct Value {
    input: Rc<Vec<u8>>,
    extracted: Cell<bool>,
}

impl Value {
    pub fn new(input: Rc<Vec<u8>>) -> Self {
        Self {
            input,
            extracted: Cell::new(false),
        }
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.input.hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
    }
}

impl Eq for Value {}

const TIMEOUT_MIN_SAMPLES: u64 = 1_000;
const TIMEOUT_PERCENTILE: f64 = 0.95;
const TIMEOUT_COEFF: f64 = 2.0;

pub struct TrieSolver {
    trie: Trie<KeyElement, Value>,
    solve_calls: usize,

    max_node_age: usize,
    dyn_timeout: bool,

    check_time_us_hist: Histogram<u64>,
    push_time_us_hist: Histogram<u64>,
}

impl Default for TrieSolver {
    fn default() -> Self {
        Self::new(10, true)
    }
}

impl TrieSolver {
    pub fn new(max_node_age: usize, dyn_timeout: bool) -> Self {
        tracing::info!("max_node_age: {max_node_age}, dyn_timeout: {dyn_timeout}");

        let check_time_us_hist =
            Histogram::new_with_max(GLOBAL_SOLVER_TIMEOUT.as_micros().try_into().unwrap(), 2)
                .expect("check_time_hist creation failed");
        let push_time_us_hist =
            Histogram::new_with_max(GLOBAL_SOLVER_TIMEOUT.as_micros().try_into().unwrap(), 2)
                .expect("check_time_hist creation failed");

        Self {
            trie: Trie::new(),
            solve_calls: 0,

            max_node_age,
            dyn_timeout,

            check_time_us_hist,
            push_time_us_hist,
        }
    }
}

impl Solver for TrieSolver {
    fn add_request(
        &mut self,
        constraints: Vec<ExprRef<ExprWrapper>>,
        input: Rc<Vec<u8>>,
    ) -> Option<Rc<Vec<u8>>> {
        let current_iteration = self.solve_calls;
        self.trie
            .insert(
                constraints
                    .into_iter()
                    .map(|ke| KeyElement::new(ke, current_iteration)),
                Value::new(input),
            )
            .map(|v| v.input)
    }

    fn extract_solutions(&mut self) -> Vec<SolverSolution> {
        let mut current_constraint = None;
        let mut result = Vec::new();
        for step in self.trie.visitor() {
            match step {
                TrieVisitorStep::PushPathComponentElements(ke) => current_constraint = ke.last(),
                TrieVisitorStep::PopPath(_) => {
                    current_constraint = None;
                }
                TrieVisitorStep::Value(values) => {
                    if let Some(KeyElement { solved_state, .. }) = current_constraint {
                        if let SolvedState::Solved(SatResult::Sat(replacements)) =
                            &*solved_state.borrow()
                        {
                            for v in values {
                                if !v.extracted.get() {
                                    v.extracted.set(true);

                                    let solution = SolverSolution {
                                        input: v.input.clone(),
                                        replacements: replacements.clone(),
                                    };
                                    if !solution.is_valid_replacement() {
                                        tracing::warn!(
                                            replacements = ?solution.replacements.as_slice(),
                                            input_length = %solution.input.len(),
                                            "invalid trie assignment",
                                        );
                                    } else {
                                        result.push(solution);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result
    }

    fn solve(&mut self, ctx: &z3::Context, timeout: Duration, optimistic: bool) {
        let start_solve = Instant::now();
        let span = tracing::info_span!("solve");
        let _guard = span.enter();
        let solve_trie = {
            let span = tracing::info_span!("solve trie construction");
            let _guard = span.enter();
            let mut solve_trie = Trie::new();

            let mut current_path = Vec::new();
            let mut filtered_constraints_path = Vec::new();
            let mut total_filtered_constraints = 0;
            for step in self.trie.visitor() {
                match step {
                    TrieVisitorStep::PushPathComponentElements(ke) => {
                        let mut filtered_constraints = 0;
                        current_path.extend(ke.filter(|ke| {
                            if ke.simplification_result(ctx).map_or(true, |b| !b) {
                                true
                            } else {
                                filtered_constraints += 1;
                                false
                            }
                        }));
                        filtered_constraints_path.push(filtered_constraints);
                    }
                    TrieVisitorStep::PopPath(n) => {
                        current_path.truncate(
                            current_path.len() - (n - filtered_constraints_path.pop().unwrap()),
                        );
                    }
                    TrieVisitorStep::Value(values) => {
                        if let Some(constraint) = current_path.last() {
                            if let SolvedState::Unsolved = &*constraint.solved_state.borrow() {
                                let mut added_any = false;
                                for v in values.iter() {
                                    if !v.extracted.get() {
                                        solve_trie.insert(current_path.iter().copied(), v);
                                        added_any = true;
                                    }
                                }
                                if added_any {
                                    total_filtered_constraints +=
                                        filtered_constraints_path.last().unwrap();
                                }
                            }
                        }
                    }
                }
            }

            tracing::info!(
                orig_trie_len = self.trie.len(),
                orig_trie_weight = self.trie.weight(),
                solve_trie_len = solve_trie.len(),
                solve_trie_weight = solve_trie.weight(),
                "solve trie reduction",
            );

            tracing::info!(
                constraints_filtered = total_filtered_constraints,
                total_constraints = solve_trie.len(),
                "always sat constraints filtered",
            );

            solve_trie
        };

        let solver = z3::Solver::new(ctx);
        let mut optimistic_solver = if optimistic {
            Some(OptimisticSolver::new(ctx))
        } else {
            None
        };

        // Wait for the histogram to have enough samples before setting a lower timeout.
        if self.dyn_timeout && self.check_time_us_hist.len() > TIMEOUT_MIN_SAMPLES {
            let perc_thres_ms = self
                .check_time_us_hist
                .value_at_quantile(TIMEOUT_PERCENTILE) as f64
                / 1_000.0;
            let perc_timeout_ms = (perc_thres_ms * TIMEOUT_COEFF).ceil() as u64;
            let timeout_ms =
                perc_timeout_ms.min(GLOBAL_SOLVER_TIMEOUT.as_millis().try_into().unwrap());

            tracing::info!("Setting timeout at {timeout_ms} ms");

            let mut solver_params = z3::Params::new(ctx);
            solver_params.set_u32("timeout", timeout_ms.try_into().unwrap());
            solver.set_params(&solver_params);
        }

        let mut current_constraint = None;
        let mut known_unsat = None;

        let mut solver_checks = 0;
        let mut solver_pushes = 0;
        let mut solver_pops = 0;
        let mut solver_resets = 0;
        let mut avoided_solver_push_through_derived_unsat = 0;
        let mut solver_check_sat = 0;
        let mut solver_check_unsat = 0;
        let mut solver_check_other = 0;
        let mut const_unsat_from_solver = 0;
        let mut const_unsat_derived = 0;
        let mut absolute_path_length = 0;
        let mut relative_path_length = 0;
        let mut total_push_time = Duration::ZERO;
        let mut total_assert_time = Duration::ZERO;
        let mut total_check_time = Duration::ZERO;
        let mut total_pop_time = Duration::ZERO;
        let mut total_reset_time = Duration::ZERO;
        let start = Instant::now();
        for step in solve_trie.visitor() {
            if start_solve.elapsed() > timeout {
                break;
            }

            match step {
                TrieVisitorStep::PushPathComponentElements(ke) => {
                    relative_path_length = 0;
                    let ke = ke.map(|k| {
                        absolute_path_length += 1;
                        relative_path_length += 1;
                        k.last_used.set(self.solve_calls);
                        k
                    });
                    if let Some(ref mut known_unsat) = known_unsat {
                        avoided_solver_push_through_derived_unsat += 1;
                        *known_unsat += 1;
                        for k in ke {
                            k.mark_known_unsat();
                            const_unsat_derived += 1;
                        }
                    } else {
                        let push_start = Instant::now();
                        solver.push();
                        let push_time = push_start.elapsed();
                        total_push_time += push_time;
                        solver_pushes += 1;

                        let mut assert_time = Duration::ZERO;
                        for k in ke {
                            if let Some(false) = k.simplification_result(ctx) {
                                known_unsat = Some(0);
                                k.mark_known_unsat();
                                const_unsat_from_solver += 1;
                            } else if known_unsat.is_some() {
                                k.mark_known_unsat();
                                const_unsat_derived += 1;
                            } else {
                                let constraint =
                                    k.expr.to_z3_simplified(ctx, true).as_bool().unwrap();

                                let assert_start = Instant::now();
                                solver.assert(&constraint);
                                assert_time += assert_start.elapsed();

                                current_constraint = Some(*k);
                            }
                        }
                        total_assert_time += assert_time;

                        self.push_time_us_hist
                            .saturating_record(push_time.as_micros().try_into().unwrap());
                        tracing::info!(
                            push_time = %push_time.as_secs_f64(),
                            absolute_path_length,
                            relative_path_length,
                            "solver push call",
                        );
                    }
                }
                TrieVisitorStep::PopPath(n) => {
                    known_unsat = match known_unsat {
                        Some(0) | None => None,
                        Some(n) => Some(n - 1),
                    };
                    absolute_path_length -= n;
                    if known_unsat.is_none() {
                        current_constraint = None;

                        let pop_start = Instant::now();
                        solver.pop(1);
                        let pop_time = pop_start.elapsed();
                        total_pop_time += pop_time;
                        solver_pops += 1;
                    }
                }
                TrieVisitorStep::Value(_values) => {
                    if known_unsat.is_none() {
                        if let Some(current_constraint) = current_constraint {
                            let solved_state = &mut *current_constraint.solved_state.borrow_mut();
                            if matches!(solved_state, SolvedState::Unsolved) {
                                let check_start = Instant::now();
                                let z3_result = solver.check();
                                let check_time = check_start.elapsed();
                                total_check_time += check_time;
                                solver_checks += 1;
                                self.check_time_us_hist
                                    .saturating_record(check_time.as_micros().try_into().unwrap());
                                tracing::info!(
                                    check_time = %check_time.as_secs_f64(),
                                    absolute_path_length,
                                    relative_path_length,
                                    sat_result = ?z3_result,
                                    "solver check call",
                                );

                                let mut sat_result = match z3_result {
                                    z3::SatResult::Unsat => {
                                        solver_check_unsat += 1;

                                        SatResult::Unsat
                                    }
                                    z3::SatResult::Unknown => {
                                        solver_check_other += 1;

                                        SatResult::Other
                                    }
                                    z3::SatResult::Sat => {
                                        solver_check_sat += 1;

                                        SatResult::Sat(Rc::new(get_model_assignment(&solver)))
                                    }
                                };

                                if z3_result != z3::SatResult::Sat && optimistic {
                                    let optimistic_solver = optimistic_solver.as_mut().unwrap();
                                    if let Some(optimistic_assignment) =
                                        optimistic_solver.solve(current_constraint.expr.clone())
                                    {
                                        sat_result = SatResult::Sat(Rc::new(optimistic_assignment))
                                    }
                                }

                                *solved_state = SolvedState::Solved(sat_result);
                            }
                        }
                    }
                }
            }
        }
        let total_time = start.elapsed();

        if optimistic {
            let optimistic_solver = optimistic_solver.as_ref().unwrap();
            let opt_stats = optimistic_solver.stats();
            tracing::info!(
                solver_check_unsat = opt_stats.solver_check_unsat,
                solver_check_unknown = opt_stats.solver_check_unknown,
                solver_check_sat = opt_stats.solver_check_sat,

                total_reset_time_us = %opt_stats.total_reset_time.as_micros(),
                total_assert_time_us = %opt_stats.total_assert_time.as_micros(),
                total_check_time_us = %opt_stats.total_check_time.as_micros(),
                "solve call (optimistic)",
            );

            let opt_query_count = opt_stats.solver_check_unsat
                + opt_stats.solver_check_unknown
                + opt_stats.solver_check_sat;
            solver_resets += opt_query_count;
            solver_checks += opt_query_count;
            solver_check_unsat += opt_stats.solver_check_unsat;
            solver_check_other += opt_stats.solver_check_unknown;
            solver_check_sat += opt_stats.solver_check_sat;
            total_reset_time += opt_stats.total_reset_time;
            total_assert_time += opt_stats.total_assert_time;
            total_check_time += opt_stats.total_check_time;
        }

        let overhead_time = total_time
            - total_push_time
            - total_assert_time
            - total_check_time
            - total_pop_time
            - total_reset_time;
        tracing::info!(
            solver_checks,
            solver_pushes,
            solver_pops,
            solver_resets,
            avoided_solver_push_through_derived_unsat,
            solver_check_sat,
            solver_check_unsat,
            solver_check_other,
            const_unsat_from_solver,
            const_unsat_derived,
            total_push_time = %total_push_time.as_secs_f64(),
            total_assert_time = %total_assert_time.as_secs_f64(),
            total_check_time = %total_check_time.as_secs_f64(),
            total_pop_time = %total_pop_time.as_secs_f64(),
            total_reset_time = %total_reset_time.as_secs_f64(),
            total_time = %total_time.as_secs_f64(),
            overhead_time = %overhead_time.as_secs_f64(),
            "solve call",
        );
        self.solve_calls += 1;
    }

    fn prune(&mut self, ctx: &z3::Context, optimistic: bool) -> Vec<SolverSolution> {
        let mut optimistic_solver = if optimistic {
            Some(OptimisticSolver::new(ctx))
        } else {
            None
        };

        let mut pruned_solutions = Vec::new();

        let mut pruned_trie = Trie::new();
        let mut current_path = Vec::new();
        let mut filtered_constraints_path = Vec::new();
        for step in self.trie.visitor() {
            match step {
                TrieVisitorStep::PushPathComponentElements(ke) => {
                    let mut filtered_constraints = 0;
                    current_path.extend(ke.filter(|ke| {
                        if ke
                            .precomputed_simplification_result()
                            .flatten()
                            .map_or(true, |b| !b)
                        {
                            true
                        } else {
                            filtered_constraints += 1;
                            false
                        }
                    }));
                    filtered_constraints_path.push(filtered_constraints);
                }
                TrieVisitorStep::PopPath(n) => {
                    current_path.truncate(
                        current_path.len() - (n - filtered_constraints_path.pop().unwrap()),
                    );
                }
                TrieVisitorStep::Value(values) => {
                    if let Some(constraint) = current_path.last() {
                        let current_node_age = self.solve_calls - constraint.last_used.get();
                        if current_node_age < self.max_node_age {
                            for v in values.iter() {
                                if !v.extracted.get() {
                                    pruned_trie
                                        .insert(current_path.iter().copied().cloned(), v.clone());
                                }
                            }
                        } else {
                            // this node is not inserted into the pruned tree, so it is dropped
                            if optimistic
                                && matches!(
                                    *constraint.solved_state.borrow(),
                                    SolvedState::Unsolved
                                )
                            {
                                let optimistic_solver = optimistic_solver.as_mut().unwrap();
                                if let Some(optimistic_assignment) =
                                    optimistic_solver.solve(constraint.expr.clone())
                                {
                                    let replacements = Rc::new(optimistic_assignment);
                                    for value in values {
                                        let solution = SolverSolution {
                                            input: value.input.clone(),
                                            replacements: Rc::clone(&replacements),
                                        };
                                        if !solution.is_valid_replacement() {
                                            tracing::warn!(
                                                replacements = ?solution.replacements.as_slice(),
                                                input_length = %solution.input.len(),
                                                model = %optimistic_solver.get_model().unwrap(),
                                                "invalid pruned assignment",
                                            );
                                        } else {
                                            pruned_solutions.push(solution);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if optimistic {
            let optimistic_solver = optimistic_solver.as_ref().unwrap();
            let opt_stats = optimistic_solver.stats();
            tracing::info!(
                solver_check_unsat = opt_stats.solver_check_unsat,
                solver_check_unknown = opt_stats.solver_check_unknown,
                solver_check_sat = opt_stats.solver_check_sat,

                total_reset_time_us = %opt_stats.total_reset_time.as_micros(),
                total_assert_time_us = %opt_stats.total_assert_time.as_micros(),
                total_check_time_us = %opt_stats.total_check_time.as_micros(),
                "prune call (optimistic)",
            );
        }

        tracing::info!(
            pre_prune_len = self.trie.len(),
            post_prune_len = pruned_trie.len(),
            pre_prune_weight = self.trie.weight(),
            post_prune_weight = pruned_trie.weight(),
            "solver trie prune"
        );

        self.trie = pruned_trie;

        pruned_solutions
    }

    fn un_z3(&mut self) {
        for step in self.trie.visitor() {
            if let TrieVisitorStep::PushPathComponentElements(ke) = step {
                for k in ke {
                    k.expr.un_z3()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        TrieSolver::default();
    }
}
