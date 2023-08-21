use std::{
    rc::Rc,
    time::{Duration, Instant},
};

use crate::runtimes::analysis::{constraint_language::ExprWrapper, expression_pool::ExprRef};

use super::{get_model_assignment, optimistic_solver::OptimisticSolver, Solver, SolverSolution};

#[derive(Default)]
pub struct LinearSolver {
    requests: Vec<(Vec<ExprRef<ExprWrapper>>, Rc<Vec<u8>>)>,
    solutions: Vec<SolverSolution>,
}

impl Solver for LinearSolver {
    fn add_request(
        &mut self,
        constraints: Vec<ExprRef<ExprWrapper>>,
        input: Rc<Vec<u8>>,
    ) -> Option<Rc<Vec<u8>>> {
        self.requests.push((constraints, input));
        None
    }

    fn solve(&mut self, ctx: &z3::Context, timeout: Duration, optimistic: bool) {
        let start_solve = Instant::now();

        let solver = z3::Solver::new_for_logic(ctx, "QF_BV").expect("Could not create solver");
        let mut optimistic_solver = if optimistic {
            Some(OptimisticSolver::new(ctx))
        } else {
            None
        };

        let mut solver_checks = 0;
        let mut solver_resets = 0;
        let mut solver_check_sat = 0;
        let mut solver_check_unsat = 0;
        let mut solver_check_other = 0;
        let mut total_reset_time = Duration::ZERO;
        let mut total_assert_time = Duration::ZERO;
        let mut total_check_time = Duration::ZERO;

        for (constraints, input) in std::mem::take(&mut self.requests) {
            let reset_start = Instant::now();
            solver.reset();
            total_reset_time += reset_start.elapsed();
            solver_resets += 1;

            let mut assert_time = Duration::ZERO;
            for constraint in &constraints {
                let constraint = constraint.to_z3_simplified(ctx, true).as_bool().unwrap();

                let assert_start = Instant::now();
                solver.assert(&constraint);
                assert_time += assert_start.elapsed()
            }
            total_assert_time += assert_time;

            let check_start = Instant::now();
            let check_result = solver.check();
            let check_time = check_start.elapsed();
            total_check_time += check_time;
            solver_checks += 1;

            let solve_path_time = assert_time + check_time;
            let path_len = constraints.len();
            tracing::info!(
                solve_path_time = %solve_path_time.as_secs_f64(),
                assert_time = %assert_time.as_secs_f64(),
                check_time = %check_time.as_secs_f64(),
                absolute_path_length = path_len,
                relative_path_length = path_len,
                sat_result = ?check_result,
                "solver check call",
            );

            match check_result {
                z3::SatResult::Unsat => solver_check_unsat += 1,
                z3::SatResult::Unknown => solver_check_other += 1,
                z3::SatResult::Sat => {
                    solver_check_sat += 1;

                    let solution = SolverSolution {
                        replacements: Rc::new(get_model_assignment(&solver)),
                        input: Rc::clone(&input),
                    };
                    if !solution.is_valid_replacement() {
                        tracing::warn!(
                            replacements = ?solution.replacements.as_slice(),
                            input_length = %solution.input.len(),
                            model = %solver.get_model().unwrap(),
                            "invalid normal assignment",
                        );
                    } else {
                        self.solutions.push(solution);
                    }
                }
            }

            if check_result != z3::SatResult::Sat && optimistic {
                if let Some(last_constraint) = constraints.last() {
                    let optimistic_solver = optimistic_solver.as_mut().unwrap();
                    if let Some(optimistic_assignment) =
                        optimistic_solver.solve(last_constraint.clone())
                    {
                        let solution = SolverSolution {
                            replacements: Rc::new(optimistic_assignment),
                            input: Rc::clone(&input),
                        };
                        if !solution.is_valid_replacement() {
                            tracing::warn!(
                                replacements = ?solution.replacements.as_slice(),
                                input_length = %solution.input.len(),
                                model = %optimistic_solver.get_model().unwrap(),
                                "invalid optimistic assignment",
                            );
                        } else {
                            self.solutions.push(solution);
                        }
                    }
                }
            }

            if start_solve.elapsed() > timeout {
                break;
            }
        }
        let total_time = start_solve.elapsed();

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

        let overhead_time = total_time - total_reset_time - total_assert_time - total_check_time;
        tracing::info!(
            solver_checks,
            solver_resets,
            solver_check_sat,
            solver_check_unsat,
            solver_check_other,
            total_reset_time = %total_reset_time.as_secs_f64(),
            total_assert_time = %total_assert_time.as_secs_f64(),
            total_check_time = %total_check_time.as_secs_f64(),
            total_time = %total_time.as_secs_f64(),
            overhead_time = %overhead_time.as_secs_f64(),
            "solve call",
        );
    }

    fn extract_solutions(&mut self) -> Vec<SolverSolution> {
        std::mem::take(&mut self.solutions)
    }

    fn prune(&mut self, _ctx: &z3::Context, _optimistic: bool) -> Vec<SolverSolution> {
        self.requests.clear();
        // we should never have any requests here anyways, because the requests
        // are cleared completely with every solver call, so just return an
        // empty vector.
        Vec::new()
    }

    fn un_z3(&mut self) {
        for (path, _) in &self.requests {
            for e in path {
                e.un_z3();
            }
        }
    }
}
