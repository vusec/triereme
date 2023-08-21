use std::{
    io::{self, Write},
    rc::Rc,
    time::Duration,
};

use super::{constraint_language::ExprWrapper, expression_pool::ExprRef};

mod linear_solver;
mod optimistic_solver;
mod trie_solver;

pub use linear_solver::LinearSolver;
pub use trie_solver::TrieSolver;

#[derive(Clone)]
pub struct SolverSolution {
    pub(crate) input: Rc<Vec<u8>>,
    pub(crate) replacements: Rc<Vec<(usize, u8)>>,
}

impl SolverSolution {
    pub fn to_vec(&self) -> Vec<u8> {
        let mut res = (*self.input).clone();
        for (index, value) in &*self.replacements {
            res[*index] = *value;
        }
        res
    }

    pub fn write_mutated_solution(&self, mut writer: impl Write) -> io::Result<()> {
        let res = self.to_vec();
        writer.write_all(&res)
    }

    pub fn is_valid_replacement(&self) -> bool {
        let input_len = self.input.len();
        for (idx, _) in self.replacements.as_ref() {
            if *idx >= input_len {
                return false;
            }
        }
        true
    }
}

pub fn get_model_assignment(solver: &z3::Solver) -> Vec<(usize, u8)> {
    let model = solver.get_model().unwrap();
    let model_string = model.to_string();
    let mut replacements = Vec::new();
    for l in model_string.lines() {
        if let [offset_str, value_str] = l.split(" -> ").collect::<Vec<_>>().as_slice() {
            let offset = offset_str
                .trim_start_matches("k!")
                .parse::<usize>()
                .unwrap();
            let value = u8::from_str_radix(value_str.trim_start_matches("#x"), 16).unwrap();
            replacements.push((offset, value));
        } else {
            panic!();
        }
    }
    replacements
}

pub trait Solver {
    /// Tries to add the request. If the request existed already, it is returned again and the original request stays in its place.
    fn add_request(
        &mut self,
        constraints: Vec<ExprRef<ExprWrapper>>,
        input: Rc<Vec<u8>>,
    ) -> Option<Rc<Vec<u8>>>;

    fn extract_solutions(&mut self) -> Vec<SolverSolution>;

    /// Solves the currently inserted queries, given a timeout.
    fn solve(&mut self, ctx: &z3::Context, timeout: Duration, optimistic: bool);

    /// Prunes the solver state to reduce memory pressure, removing old state that we think is no longer relevant.
    /// Returns optimistic solutions for pruned paths.
    fn prune(&mut self, ctx: &z3::Context, optimistic: bool) -> Vec<SolverSolution>;

    fn un_z3(&mut self);
}
