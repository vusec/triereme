use std::time::{Duration, Instant};

use crate::runtimes::analysis::{
    constraint_language::ExprWrapper, expression_pool::ExprRef, solver::get_model_assignment,
};

#[derive(Default)]
pub(crate) struct OptimisticStats {
    pub(crate) total_reset_time: Duration,
    pub(crate) total_assert_time: Duration,
    pub(crate) total_check_time: Duration,
    pub(crate) solver_check_unsat: u64,
    pub(crate) solver_check_unknown: u64,
    pub(crate) solver_check_sat: u64,
}

pub(crate) struct OptimisticSolver<'ctx> {
    ctx: &'ctx z3::Context,
    solver: z3::Solver<'ctx>,

    stats: OptimisticStats,
}

impl<'ctx> OptimisticSolver<'ctx> {
    pub(crate) fn new(ctx: &'ctx z3::Context) -> Self {
        let solver = z3::Solver::new_for_logic(ctx, "QF_BV").expect("Could not create solver");
        Self {
            ctx,
            solver,
            stats: Default::default(),
        }
    }

    #[must_use]
    pub(crate) fn solve(&mut self, constraint: ExprRef<ExprWrapper>) -> Option<Vec<(usize, u8)>> {
        let reset_start = Instant::now();
        self.solver.reset();
        let reset_time = reset_start.elapsed();
        self.stats.total_reset_time += reset_time;

        let constraint = constraint
            .to_z3_simplified(self.ctx, true)
            .as_bool()
            .unwrap();
        let assert_start = Instant::now();
        self.solver.assert(&constraint);
        let assert_time = assert_start.elapsed();
        self.stats.total_assert_time += assert_time;

        let check_start = Instant::now();
        let check_result = self.solver.check();
        let check_time = check_start.elapsed();
        self.stats.total_check_time += check_time;

        tracing::info!(
            reset_time_us = %reset_time.as_micros(),
            assert_time_us = %assert_time.as_micros(),
            check_time_us = %check_time.as_micros(),
            sat_result = ?check_result,
            "optimistic solver query",
        );

        match check_result {
            z3::SatResult::Unsat => {
                self.stats.solver_check_unsat += 1;
                None
            }
            z3::SatResult::Unknown => {
                self.stats.solver_check_unknown += 1;
                None
            }
            z3::SatResult::Sat => {
                self.stats.solver_check_sat += 1;
                Some(get_model_assignment(&self.solver))
            }
        }
    }

    pub(crate) fn stats(&self) -> &OptimisticStats {
        &self.stats
    }

    pub(crate) fn get_model(&self) -> Option<z3::Model> {
        self.solver.get_model()
    }
}
