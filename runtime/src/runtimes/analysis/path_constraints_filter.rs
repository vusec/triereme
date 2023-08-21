use std::ops::Deref;

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;

use crate::runtimes::analysis::constraint_language::Expr;

use super::{
    constraint_info::ConstraintInfo,
    constraint_language::ExprWrapper,
    expression_pool::{ExprPool, ExprRef},
};

pub trait PathConstraintsFilter {
    fn compute_path_constraints(
        &mut self,
        path_constraints: &[ConstraintInfo],
        next_constraint: &ConstraintInfo,
        pool: &mut ExprPool<ExprWrapper>,
        variables: &HashSet<usize>,
    ) -> Vec<Vec<ExprRef<ExprWrapper>>>;
}

#[derive(Default)]
pub struct NoConstraintsFilter;

impl PathConstraintsFilter for NoConstraintsFilter {
    fn compute_path_constraints(
        &mut self,
        path_constraints: &[ConstraintInfo],
        _next_constraint: &ConstraintInfo,
        _pool: &mut ExprPool<ExprWrapper>,
        _variables: &HashSet<usize>,
    ) -> Vec<Vec<ExprRef<ExprWrapper>>> {
        vec![path_constraints
            .iter()
            .map(ConstraintInfo::constraint_ref)
            .cloned()
            .collect::<Vec<_>>()]
    }
}

pub struct FactorizingConstraintsFilter {
    dependent_variables_cache: HashMap<ExprRef<ExprWrapper>, HashSet<ExprRef<ExprWrapper>>>,
}

impl Default for FactorizingConstraintsFilter {
    fn default() -> Self {
        Self {
            dependent_variables_cache: HashMap::new(),
        }
    }
}

impl PathConstraintsFilter for FactorizingConstraintsFilter {
    fn compute_path_constraints(
        &mut self,
        path_constraints: &[ConstraintInfo],
        next_constraint: &ConstraintInfo,
        _pool: &mut ExprPool<ExprWrapper>,
        _variables: &HashSet<usize>,
    ) -> Vec<Vec<ExprRef<ExprWrapper>>> {
        let (dependent_constraints, _n_dependend_variables) = {
            let mut dependent_constraints = HashSet::new();
            let mut dependent_constraints_to_check = path_constraints
                .iter()
                .map(ConstraintInfo::constraint_ref)
                .collect::<HashSet<_>>();
            let mut dependent_variables = HashSet::new();
            let mut dependent_variables_to_check = next_constraint
                .variables()
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            while let Some(var) = dependent_variables_to_check
                .iter()
                .next()
                .copied()
                .map(|v| dependent_variables_to_check.take(&v).unwrap())
            {
                dependent_variables.insert(var);
                for constraint in
                    dependent_constraints_to_check.drain_filter(|c| c.variables().contains(&var))
                {
                    // we are dependent on this constraint
                    if !dependent_constraints.insert(constraint.clone()) {
                        // we have already processed this constraint
                        continue;
                    }
                    // we are also dependent on all of its variables
                    for v in constraint.variables().iter().copied() {
                        if dependent_variables.insert(v) {
                            // only need to check those variables, which we have not already checked
                            dependent_variables_to_check.insert(v);
                        }
                    }
                    if let Some(precomputed_dependent_constraints) =
                        self.dependent_variables_cache.get(constraint)
                    {
                        // we already know we will be dependent on all of these constraints as well
                        dependent_constraints
                            .extend(precomputed_dependent_constraints.iter().cloned());
                        // and are therefore also dependent on all of its variables
                        for v in precomputed_dependent_constraints
                            .iter()
                            .flat_map(|c| c.variables().iter())
                            .copied()
                        {
                            if dependent_variables.insert(v) {
                                dependent_variables_to_check.insert(v);
                            }
                        }
                    }
                }
            }
            assert!(
                dependent_variables_to_check.is_empty()
                    || dependent_constraints_to_check.is_empty()
            );
            self.dependent_variables_cache.insert(
                next_constraint.constraint_ref().clone(),
                dependent_constraints.clone(),
            );
            (dependent_constraints, dependent_variables.len())
        };
        vec![path_constraints
            .iter()
            .filter(|i| dependent_constraints.contains(i.constraint_ref()))
            .map(ConstraintInfo::constraint_ref)
            .unique()
            .cloned()
            .collect::<Vec<_>>()]
    }
}

#[derive(Default)]
pub struct ConcretizingConstraintsFilter;

impl PathConstraintsFilter for ConcretizingConstraintsFilter {
    fn compute_path_constraints(
        &mut self,
        path_constraints: &[ConstraintInfo],
        _next_constraint: &ConstraintInfo,
        pool: &mut ExprPool<ExprWrapper>,
        variables: &HashSet<usize>,
    ) -> Vec<Vec<ExprRef<ExprWrapper>>> {
        // due to concretization, only the variables of the last constraint are dependent
        vec![path_constraints
            .iter()
            .map(|other_constraint| {
                if let Some(concretized) =
                    other_constraint.get_concretized_with_mapping(variables, pool)
                {
                    pool.alloc(concretized.into())
                } else {
                    other_constraint.constraint_ref().clone()
                }
            })
            // filter out constraints that are trivially satisfied
            .filter(|e| !matches!(e.deref().deref(), Expr::True))
            .unique()
            .collect::<Vec<_>>()]
    }
}

pub enum PathFilter {
    None(NoConstraintsFilter),
    Factorizing(FactorizingConstraintsFilter),
    Concretizing(ConcretizingConstraintsFilter),
    Combined(FactorizingConstraintsFilter, ConcretizingConstraintsFilter),
}

impl PathConstraintsFilter for PathFilter {
    fn compute_path_constraints(
        &mut self,
        path_constraints: &[ConstraintInfo],
        next_constraint: &ConstraintInfo,
        pool: &mut ExprPool<ExprWrapper>,
        variables: &HashSet<usize>,
    ) -> Vec<Vec<ExprRef<ExprWrapper>>> {
        match self {
            PathFilter::None(f) => {
                f.compute_path_constraints(path_constraints, next_constraint, pool, variables)
            }
            PathFilter::Factorizing(f) => {
                f.compute_path_constraints(path_constraints, next_constraint, pool, variables)
            }
            PathFilter::Concretizing(f) => {
                f.compute_path_constraints(path_constraints, next_constraint, pool, variables)
            }
            PathFilter::Combined(f1, f2) => {
                let mut res = Vec::new();
                res.append(&mut f1.compute_path_constraints(
                    path_constraints,
                    next_constraint,
                    pool,
                    variables,
                ));
                res.append(&mut f2.compute_path_constraints(
                    path_constraints,
                    next_constraint,
                    pool,
                    variables,
                ));
                res
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum PathFilteringMode {
    None,
    Factorization,
    Concretization,
    Combined,
}

impl PathFilteringMode {
    #[must_use]
    pub fn create_filter(&self) -> PathFilter {
        match self {
            PathFilteringMode::None => PathFilter::None(NoConstraintsFilter::default()),
            PathFilteringMode::Factorization => {
                PathFilter::Factorizing(FactorizingConstraintsFilter::default())
            }
            PathFilteringMode::Concretization => {
                PathFilter::Concretizing(ConcretizingConstraintsFilter::default())
            }
            PathFilteringMode::Combined => PathFilter::Combined(
                FactorizingConstraintsFilter::default(),
                ConcretizingConstraintsFilter::default(),
            ),
        }
    }
}
