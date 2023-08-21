use std::{
    convert::TryInto,
    io::{Read, Write},
    os::raw::c_int,
    rc::Rc,
    time::Duration,
};

use apint::ApInt;
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use z3::{Config, Context};

use self::{
    constraint_info::ConstraintInfo,
    constraint_language::{Expr, ExprWrapper},
    expression_pool::{ExprPool, ExprRef},
    path_constraints_filter::{PathConstraintsFilter, PathFilter},
    solver::{LinearSolver, Solver, TrieSolver},
};

pub use self::{path_constraints_filter::PathFilteringMode, solver::SolverSolution};

use super::high_level::HighLevelRuntime;
use crate::runtimes::analysis::coverage::BranchCoverageFilter;

mod const_eval;
mod constraint_info;
mod constraint_language;
mod coverage;
mod expression_pool;
mod path_constraints_filter;
mod solver;

const GLOBAL_SOLVER_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy)]
pub struct AnalysisSettings {
    pub keep_trie: bool,
    pub solve_division: bool,
    pub path_filtering_mode: PathFilteringMode,
    pub linear_solving: bool,
    pub max_node_age: usize,
    /// inspired by QSYM optimistic solving, but even more optimistic. solves all constraints with and without path constraints
    pub very_optimistic: bool,
    /// inspired by QSYM optimistic solving, optimistically try to solve the last path constraint of unsolved paths.
    pub optimistic_unsolved: bool,
    /// inspired by QSYM optimistic solving, optimistically try to solve the last path constraint of pruned paths.
    pub optimistic_pruned: bool,
    /// Set timeout on single conditions dynamically, relying on statistics for the current benchmark.
    pub dyn_timeout: bool,
}

#[derive(Default)]
struct FileStats {
    expressions_processed: usize,
    expressions_filtered: usize,
    branches_processed: usize,
    branches_filtered: usize,
    const_true_constraints: usize,
    const_false_constraints: usize,
    variable_constraints: usize,
}

pub struct AnalysisRuntime {
    path_constraints: Vec<ConstraintInfo>,
    expr_pool: ExprPool<ExprWrapper>,
    input: Option<Rc<Vec<u8>>>,
    solver: Box<dyn Solver>,
    settings: AnalysisSettings,
    solutions: Vec<SolverSolution>,
    branch_filter: BranchCoverageFilter,
    path_filter: PathFilter,
    z3: Context,

    file_stats: FileStats,

    files_processed: usize,
    problems_generated: usize,
    pending_problems: usize,
    constraints_generated: usize,
    constraints_deduped: usize,
    problems_deduped: usize,

    branch_variables: HashMap<usize, HashSet<usize>>,
}

impl AnalysisRuntime {
    fn make_solver(settings: AnalysisSettings) -> Box<dyn Solver> {
        if settings.linear_solving {
            Box::new(LinearSolver::default()) as Box<dyn Solver>
        } else {
            Box::new(TrieSolver::new(settings.max_node_age, settings.dyn_timeout)) as Box<dyn Solver>
        }
    }

    #[must_use]
    pub fn new(settings: AnalysisSettings) -> Self {
        tracing::info!("runtime constructed");
        let z3 = setup_z3();
        let path_filter = settings.path_filtering_mode.create_filter();
        let solver = Self::make_solver(settings);
        Self {
            path_constraints: Vec::new(),
            expr_pool: ExprPool::default(),
            input: None,
            solver,
            solutions: Vec::new(),
            settings,
            branch_filter: BranchCoverageFilter::default(),
            path_filter,
            z3,
            file_stats: FileStats::default(),
            files_processed: 0,
            problems_generated: 0,
            pending_problems: 0,
            constraints_generated: 0,
            constraints_deduped: 0,
            problems_deduped: 0,
            branch_variables: HashMap::new(),
        }
    }

    pub fn export_state(&self, write: impl Write) -> std::io::Result<()> {
        self.branch_filter.export(write)
    }

    pub fn import_state(&mut self, read: impl Read) -> std::io::Result<()> {
        self.branch_filter.import(read)
    }

    pub fn receive_input(&mut self, input: Vec<u8>) {
        tracing::info!(
            expressions_processed = self.file_stats.expressions_processed,
            expressions_filtered = self.file_stats.expressions_filtered,
            branches_processed = self.file_stats.branches_processed,
            branches_filtered = self.file_stats.branches_filtered,
            const_true_constraints = self.file_stats.const_true_constraints,
            const_false_constraints = self.file_stats.const_false_constraints,
            variable_constraints = self.file_stats.variable_constraints,
            "file stats",
        );
        self.branch_variables.clear();
        self.file_stats = FileStats::default();
        tracing::info!(input_len = input.len(), "received input");
        self.files_processed += 1;
        // new execution, clear path constraints!
        self.path_constraints.clear();
        self.path_filter = self.settings.path_filtering_mode.create_filter();
        self.pending_problems = 0;
        self.branch_filter.reset();
        self.input = Some(Rc::new(input));
    }

    pub fn solve(&mut self, timeout: Duration) {
        self.solver
            .solve(&self.z3, timeout, self.settings.optimistic_unsolved);
        self.solutions.append(&mut self.solver.extract_solutions());

        if self.settings.keep_trie {
            let mut pruned_solutions = self.solver.prune(&self.z3, self.settings.optimistic_pruned);
            self.solutions.append(&mut pruned_solutions);
        } else {
            self.solver = Self::make_solver(self.settings);
        };

        for _ in 0..10 {
            if self.expr_pool.gc().count() == 0 {
                break;
            }
        }

        self.solver.un_z3();
        self.expr_pool.iter().for_each(|e| e.un_z3());
        if self.files_processed % 10 == 0 {
            // reset the complete z3 context now and then to avoid leaky z3
            self.expr_pool.iter().for_each(|e| e.uncache());
            self.solver.un_z3();
            self.expr_pool.iter().for_each(|e| e.un_z3());
            self.z3 = setup_z3();
        }
    }

    pub fn finish(&mut self) -> Vec<SolverSolution> {
        std::mem::take(&mut self.solutions)
    }

    fn build_expr(&mut self, expr: Expr) -> Option<ExprRef<ExprWrapper>> {
        self.file_stats.expressions_processed += 1;
        let is_div = matches!(
            &expr,
            Expr::UDiv { .. } | Expr::SDiv { .. } | Expr::URem { .. } | Expr::SRem { .. }
        );
        let ignore_div = is_div && !self.settings.solve_division;
        if !ignore_div {
            let const_eval = expr.const_eval();
            let allocated = self.expr_pool.alloc(const_eval.into());
            Some(allocated)
        } else {
            self.file_stats.expressions_filtered += 1;
            let concretized = if expr.is_concrete() {
                let v = expr.const_eval();
                self.expr_pool.alloc(v.into())
            } else {
                let wrapped: ExprWrapper = expr.into();
                let r = self.expr_pool.alloc(wrapped);
                let v = r
                    .inner
                    .complete_concretize(self.input.as_ref().unwrap().as_ref(), &mut self.expr_pool)
                    .unwrap();
                self.expr_pool.alloc(v.into())
            };
            Some(concretized)
        }
    }

    fn path_constraint(&mut self, info: ConstraintInfo) {
        self.pending_problems += 1;

        self.branch_variables
            .entry(info.location_id())
            .or_default()
            .extend(info.constraint_ref().variables().iter());

        if !self
            .branch_filter
            .is_interesting_branch(info.location_id(), info.taken())
        {
            self.file_stats.branches_filtered += 1;
            self.path_constraints.push(info);
            return;
        }

        let pool = &mut self.expr_pool;
        let mut path_constraints = self.path_filter.compute_path_constraints(
            self.path_constraints.as_slice(),
            &info,
            pool,
            &self.branch_variables[&info.location_id()],
        );

        {
            for path_constraints in &path_constraints {
                let full_path_len = self.path_constraints.len();
                let reduced_path_len = path_constraints.len();
                let full_path_variables = self
                    .path_constraints
                    .iter()
                    .flat_map(|e| e.variables().iter())
                    .unique()
                    .count();
                let reduced_path_variables = path_constraints
                    .iter()
                    .flat_map(|e| e.variables().iter())
                    .unique()
                    .count();
                tracing::info!(
                    full_path_len,
                    reduced_path_len,
                    full_path_variables,
                    reduced_path_variables,
                    "path complexity reduction"
                );
            }
        }

        let negated_constraint =
            pool.alloc(Expr::Not(info.constraint_ref().clone()).const_eval().into());
        for path_constraints in &mut path_constraints {
            path_constraints.push(negated_constraint.clone());
        }

        // in very optimistic mode, we always try to additionally solve the last path constraint on its own
        if self.settings.very_optimistic {
            path_constraints.push(vec![negated_constraint])
        }

        self.pending_problems = 0;
        for path_constraints in path_constraints {
            self.problems_generated += 1 + self.pending_problems;
            self.constraints_generated += path_constraints.len();
            let constraints_len = path_constraints.len();
            if let Some(_previous_constraint) = self
                .solver
                .add_request(path_constraints, info.input().clone())
            {
                self.problems_deduped += 1;
                self.constraints_deduped += constraints_len;
                println!(
                    "{}/{} ({:.2}) {}",
                    self.problems_deduped,
                    self.problems_generated,
                    (self.problems_deduped as f64 / self.problems_generated as f64) * 100.0,
                    self.constraints_deduped
                );
            }
        }

        self.path_constraints.push(info);
    }
}

fn setup_z3() -> Context {
    let mut cfg = Config::new();
    cfg.set_timeout_msec(GLOBAL_SOLVER_TIMEOUT.as_millis().try_into().unwrap());
    Context::new(&cfg)
}

impl HighLevelRuntime for AnalysisRuntime {
    type Expr = ExprRef<ExprWrapper>;

    fn build_integer(&mut self, value: u64, bits: u8) -> Option<Self::Expr> {
        self.build_expr(Expr::Integer(
            ApInt::from_u64(value)
                .into_zero_resize(bits as usize)
                .into(),
        ))
    }

    #[allow(unused)]
    fn build_integer128(&mut self, high: u64, low: u64) -> Option<Self::Expr> {
        self.build_expr(Expr::Integer(
            ApInt::from_u128((high as u128) << 64 | low as u128).into(),
        ))
    }

    #[allow(unused)]
    fn build_float(&mut self, value: f64, is_double: c_int) -> Option<Self::Expr> {
        None
    }

    fn build_null_pointer(&mut self) -> Option<Self::Expr> {
        self.build_expr(Expr::Integer(
            ApInt::zero((usize::BITS as usize).into()).into(),
        ))
    }

    fn build_true(&mut self) -> Option<Self::Expr> {
        self.build_expr(Expr::True)
    }

    fn build_false(&mut self) -> Option<Self::Expr> {
        self.build_expr(Expr::False)
    }

    fn build_bool(&mut self, value: bool) -> Option<Self::Expr> {
        if value {
            self.build_true()
        } else {
            self.build_false()
        }
    }

    fn build_neg(&mut self, expr: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let Some(expr) = expr {
            self.build_expr(Expr::Neg(expr.clone()))
        } else {
            None
        }
    }

    fn build_add(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Add(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_sub(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Sub(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_mul(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Mul(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_unsigned_div(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::UDiv(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_signed_div(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::SDiv(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_unsigned_rem(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::URem(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_signed_rem(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::SRem(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_shift_left(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Shl(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_logical_shift_right(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Lshr(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_arithmetic_shift_right(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Ashr(a.clone(), b.clone()))
        } else {
            None
        }
    }

    #[allow(unused)]
    fn build_fp_add(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_fp_sub(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_fp_mul(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_fp_div(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_fp_rem(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_fp_abs(&mut self, a: Option<&Self::Expr>) -> Option<Self::Expr> {
        None
    }

    fn build_not(&mut self, expr: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let Some(expr) = expr {
            self.build_expr(Expr::Not(expr.clone()))
        } else {
            None
        }
    }

    fn build_signed_less_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Slt(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_signed_less_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Sle(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_signed_greater_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Sgt(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_signed_greater_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Sge(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_unsigned_less_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Ult(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_unsigned_less_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Ule(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_unsigned_greater_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Ugt(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_unsigned_greater_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Uge(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Equal(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_not_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::NotEqual(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_bool_and(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::BoolAnd(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_and(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::And(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_bool_or(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::BoolOr(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_or(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Or(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_bool_xor(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::BoolXor(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn build_xor(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Xor(a.clone(), b.clone()))
        } else {
            None
        }
    }

    #[allow(unused)]
    fn build_float_ordered_greater_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_ordered_greater_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_ordered_less_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_ordered_less_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_ordered_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_ordered_not_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_ordered(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_unordered(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_unordered_greater_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_unordered_greater_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_unordered_less_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_unordered_less_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_unordered_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_unordered_not_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        None
    }

    fn build_sext(&mut self, expr: Option<&Self::Expr>, bits: u8) -> Option<Self::Expr> {
        if let Some(expr) = expr {
            self.build_expr(Expr::Sext {
                op: expr.clone(),
                bits,
            })
        } else {
            None
        }
    }

    fn build_zext(&mut self, expr: Option<&Self::Expr>, bits: u8) -> Option<Self::Expr> {
        if let Some(expr) = expr {
            self.build_expr(Expr::Zext {
                op: expr.clone(),
                bits,
            })
        } else {
            None
        }
    }

    fn build_trunc(&mut self, expr: Option<&Self::Expr>, bits: u8) -> Option<Self::Expr> {
        if let Some(expr) = expr {
            self.build_expr(Expr::Trunc {
                op: expr.clone(),
                bits,
            })
        } else {
            None
        }
    }

    fn build_bswap(&mut self, expr: Option<&Self::Expr>) -> Option<Self::Expr> {
        if let Some(expr) = expr {
            self.build_expr(Expr::Bswap(expr.clone()))
        } else {
            None
        }
    }

    #[allow(unused)]
    fn build_int_to_float(
        &mut self,
        value: Option<&Self::Expr>,
        is_double: c_int,
        is_signed: c_int,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_to_float(
        &mut self,
        expr: Option<&Self::Expr>,
        to_double: c_int,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_bits_to_float(
        &mut self,
        expr: Option<&Self::Expr>,
        to_double: c_int,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_to_bits(&mut self, expr: Option<&Self::Expr>) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_to_signed_integer(
        &mut self,
        expr: Option<&Self::Expr>,
        bits: u8,
    ) -> Option<Self::Expr> {
        None
    }

    #[allow(unused)]
    fn build_float_to_unsigned_integer(
        &mut self,
        expr: Option<&Self::Expr>,
        bits: u8,
    ) -> Option<Self::Expr> {
        None
    }

    fn build_bool_to_bits(&mut self, expr: Option<&Self::Expr>, bits: u8) -> Option<Self::Expr> {
        if let Some(expr) = expr {
            self.build_expr(Expr::BoolToBits {
                op: expr.clone(),
                bits,
            })
        } else {
            None
        }
    }

    fn concat_helper(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr> {
        if let (Some(a), Some(b)) = (a, b) {
            self.build_expr(Expr::Concat(a.clone(), b.clone()))
        } else {
            None
        }
    }

    fn extract_helper(
        &mut self,
        expr: Option<&Self::Expr>,
        first_bit: usize,
        last_bit: usize,
    ) -> Option<Self::Expr> {
        if let Some(expr) = expr {
            self.build_expr(Expr::Extract {
                op: expr.clone(),
                first_bit,
                last_bit,
            })
        } else {
            None
        }
    }

    fn push_path_constraint(
        &mut self,
        constraint: Option<&Self::Expr>,
        taken: c_int,
        site_id: usize,
    ) {
        if let Some(constraint) = constraint {
            let constraint = if taken == 0 {
                self.build_not(Some(constraint)).unwrap()
            } else {
                constraint.clone()
            };
            self.file_stats.branches_processed += 1;
            if let Expr::True = **constraint {
                self.file_stats.const_true_constraints += 1;
            } else if let Expr::False = **constraint {
                self.file_stats.const_false_constraints += 1;
            } else {
                self.file_stats.variable_constraints += 1;
                let constraint_info = ConstraintInfo::new(
                    constraint,
                    self.input.as_ref().unwrap().clone(),
                    site_id,
                    taken != 0,
                );
                self.path_constraint(constraint_info);
            }
        }
        {
            let expr_pool = &mut self.expr_pool;
            expr_pool.gc_point().for_each(|_| {});
        }
        //println!("{:?}", self.expr_pool.stats())
    }

    fn get_input_byte(&mut self, offset: usize) -> Option<Self::Expr> {
        self.build_expr(Expr::Variable { offset })
    }

    fn expression_unreachable(&mut self, expr: Option<Self::Expr>) {
        if let Some(expr) = expr {
            std::mem::drop(expr);
        }
    }
}
