use core::hash::Hash;
use std::{cmp::Ordering, rc::Rc};

use hashbrown::HashSet;
use itertools::Itertools;

use super::{
    constraint_language::{Expr, ExprWrapper},
    expression_pool::{ExprPool, ExprRef},
};

#[allow(unused)]
fn sorted_deduped_vec_difference(
    mut a: &[usize],
    mut b: &[usize],
    associated_values: &[u8],
) -> Vec<(usize, u8)> {
    if a.is_empty() {
        return Vec::new();
    }
    if b.is_empty() ||
    // SAFETY a and b are non-empty as per the last two checks
    unsafe {
        a.get_unchecked(0)  > b.get_unchecked(b.len() - 1) ||
        b.get_unchecked(0) > a.get_unchecked(a.len() - 1)
    } {
        return a
            .iter()
            .map(|&o| (o, associated_values[o]))
            .collect::<Vec<_>>();
    }
    let mut res = Vec::with_capacity(a.len().saturating_sub(b.len()));
    loop {
        // SAFETY a and b are non-empty for each iteration of the loop. checked at the beginning and after each case in the loop.
        let first_a = unsafe { *a.get_unchecked(0) };
        let first_b = unsafe { *b.get_unchecked(0) };
        match first_a.cmp(&first_b) {
            Ordering::Less => {
                if let Ok(new_slice_start) = a.binary_search(&first_b) {
                    let (include, rest) = a.split_at(new_slice_start);
                    res.extend(include.iter().map(|&o| (o, associated_values[o])));
                    a = &rest[1..];
                    if a.is_empty() {
                        return res;
                    }
                    // SAFETY a is non-empty
                    b = &b[1..];
                    if b.is_empty() {
                        res.extend(a.iter().map(|&o| (o, associated_values[o])));
                        return res;
                    }
                    // SAFETY b is non-empty
                } else {
                    res.extend(a.iter().map(|&o| (o, associated_values[o])));
                    return res;
                }
            }
            Ordering::Equal => {
                a = &a[1..];
                if a.is_empty() {
                    return res;
                }
                // SAFETY a is non-empty
                b = &b[1..];
                if b.is_empty() {
                    res.extend(a.iter().map(|&o| (o, associated_values[o])));
                    return res;
                }
                // SAFETY b is non-empty
            }
            Ordering::Greater => {
                let last_b = unsafe { *b.get_unchecked(b.len() - 1) };
                if last_b < first_a {
                    // b contains no element which is >= a[0]
                    res.extend(a.iter().map(|&o| (o, associated_values[o])));
                    return res;
                }
                b = &b[b.partition_point(|&x| x < a[0])..];
                // SAFETY a is non-empty because it was not modified
                // SAFETY b is non-empty because partition point will have returned a valid index
                // and therefore the slicing operation will leave at least one element in the resulting slice.
            }
        }
    }
}

#[derive(Clone, Debug, Eq)]
pub struct ConstraintInfo {
    expression: ExprRef<ExprWrapper>,
    input: Rc<Vec<u8>>,
    loc: usize,
    taken: bool,
}

impl PartialEq for ConstraintInfo {
    fn eq(&self, other: &Self) -> bool {
        self.expression == other.expression && self.input == other.input
    }
}

impl Hash for ConstraintInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.expression.hash(state);
        self.input.hash(state);
    }
}

impl ConstraintInfo {
    pub fn new(
        constraint: ExprRef<ExprWrapper>,
        input: Rc<Vec<u8>>,
        loc: usize,
        taken: bool,
    ) -> Self {
        Self {
            expression: constraint,
            input,
            loc,
            taken,
        }
    }

    pub fn constraint_ref(&self) -> &ExprRef<ExprWrapper> {
        &self.expression
    }

    pub fn variables(&self) -> &Rc<Vec<usize>> {
        self.expression.variables()
    }

    pub fn input(&self) -> &Rc<Vec<u8>> {
        &self.input
    }

    pub fn location_id(&self) -> usize {
        self.loc
    }

    pub fn taken(&self) -> bool {
        self.taken
    }

    pub fn get_concretized_with_mapping(
        &self,
        keep: &HashSet<usize>,
        pool: &mut ExprPool<ExprWrapper>,
    ) -> Option<Expr> {
        if let Some(max) = self.expression.variables().last() {
            if *max >= self.input.len() {
                panic!(
                    "constraint references input out of bounds {:#?} {:#?}",
                    self.constraint_ref(),
                    self.input
                );
            }
        }

        /*let replacements = Rc::new(sorted_deduped_vec_difference(
            self.expression.variables(),
            keep.iter().copied().collect_vec().as_slice(),
            self.input.as_slice(),
        )); */
        let replacements = Rc::new(
            self.expression
                .variables()
                .iter()
                .copied()
                .filter(|offset| !keep.contains(offset))
                .map(|offset| (offset, self.input[offset]))
                .collect_vec(),
        );
        let res =
            self.expression
                .inner
                .concretize(&replacements, self.expression.variables(), pool);
        if replacements.len() > 0 {
            assert!(res.is_some());
        }
        if replacements.len() != 0 && self.expression.variables().len() - replacements.len() == 0 {
            match &res {
                Some(Expr::True) => {}
                Some(Expr::False) => {
                    //println!("{:#?}", replacements);
                    //println!("{:#?}", self.constraint());
                    //println!();
                    //println!("Constraint was concretized to false, making for an impossible path constraint. Ignoring constraint...");
                    return Some(Expr::True);
                }
                _ => panic!("unexpected expr as result of replacement: {:#?}, replacements: {:#?}, original: {:#?}", res, replacements, self.expression),
            };
        }
        res
    }
}
