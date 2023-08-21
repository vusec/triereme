use std::{
    cell::{Ref, RefCell},
    convert::TryFrom,
    fmt::Debug,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    rc::Rc,
};

use apint::{ApInt, Width};
use hashbrown::{HashMap, HashSet};
use z3::{
    ast::{Ast, Bool, Dynamic, BV},
    Context, Symbol,
};

use super::expression_pool::{ExprPool, ExprRef};

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Hash)]
pub enum Expr {
    Variable {
        offset: usize,
    },
    Integer(ApIntOrd),
    True,
    False,
    Neg(ExprRef<ExprWrapper>),
    Add(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Sub(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Mul(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    UDiv(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    SDiv(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    URem(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    SRem(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Shl(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Lshr(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Ashr(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Not(ExprRef<ExprWrapper>),
    Slt(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Sle(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Sgt(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Sge(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Ult(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Ule(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Ugt(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Uge(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Equal(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    NotEqual(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    BoolOr(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    BoolAnd(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    BoolXor(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    And(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Or(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Xor(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Sext {
        op: ExprRef<ExprWrapper>,
        bits: u8,
    },
    Zext {
        op: ExprRef<ExprWrapper>,
        bits: u8,
    },
    Trunc {
        op: ExprRef<ExprWrapper>,
        bits: u8,
    },
    Bswap(ExprRef<ExprWrapper>),
    BoolToBits {
        op: ExprRef<ExprWrapper>,
        bits: u8,
    },
    Concat(ExprRef<ExprWrapper>, ExprRef<ExprWrapper>),
    Extract {
        op: ExprRef<ExprWrapper>,
        first_bit: usize,
        last_bit: usize,
    },
}

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct ApIntOrd(ApInt);

impl Ord for ApIntOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.width().cmp(&other.0.width()) {
            std::cmp::Ordering::Equal => {
                if self.0.checked_ult(&other.0).unwrap() {
                    std::cmp::Ordering::Less
                } else if self.0 == other.0 {
                    std::cmp::Ordering::Equal
                } else {
                    std::cmp::Ordering::Greater
                }
            }
            unequal => unequal,
        }
    }
}

impl PartialOrd for ApIntOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Deref for ApIntOrd {
    type Target = ApInt;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ApIntOrd {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ApInt> for ApIntOrd {
    fn from(v: ApInt) -> Self {
        Self(v)
    }
}

impl From<bool> for Expr {
    fn from(b: bool) -> Self {
        if b {
            Self::True
        } else {
            Self::False
        }
    }
}

type InputReplacements = Vec<(usize, u8)>;

type ReplacementsCache = HashMap<Rc<InputReplacements>, Option<Expr>>;

#[derive(Clone)]
pub struct ExprWrapper {
    expr: Expr,
    variables: Rc<Vec<usize>>,
    replacements: RefCell<ReplacementsCache>,
    z3_expr: RefCell<Option<Dynamic<'static>>>,
    z3_expr_simplified: RefCell<Option<Dynamic<'static>>>,
}

impl Hash for ExprWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.hash(state);
    }
}

impl PartialOrd for ExprWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.expr.partial_cmp(&other.expr)
    }
}

impl Ord for ExprWrapper {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.expr.cmp(&other.expr)
    }
}

impl Debug for ExprWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.expr.fmt(f)
    }
}

impl PartialEq for ExprWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.expr.eq(&other.expr)
    }
}

impl Eq for ExprWrapper {}

impl Deref for ExprWrapper {
    type Target = Expr;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

fn sorted_deduped_vec_intersection(a: &[usize], b: &[(usize, u8)]) -> Vec<(usize, u8)> {
    let mut res = Vec::new();
    if a.len() < b.len() {
        for a_val in a {
            if let Ok(idx) = b.binary_search_by_key(a_val, |&(v, _)| v) {
                res.push(unsafe { *b.get_unchecked(idx) });
            }
        }
    } else {
        for &(b_val, associated_value) in b {
            if a.binary_search(&b_val).is_ok() {
                res.push((b_val, associated_value));
            }
        }
    }
    res
}

impl ExprWrapper {
    fn new(expr: Expr) -> Self {
        let variables = expr.collect_variables();
        Self {
            expr,
            variables,
            replacements: RefCell::default(),
            z3_expr: RefCell::new(None),
            z3_expr_simplified: RefCell::new(None),
        }
    }

    pub fn variables(&self) -> &Rc<Vec<usize>> {
        &self.variables
    }

    pub fn into_inner(self) -> Expr {
        self.expr
    }

    pub fn uncache(&self) {
        self.replacements.borrow_mut().clear();
        self.z3_expr.replace(None);
    }

    pub fn un_z3(&self) {
        self.z3_expr.replace(None);
        self.z3_expr_simplified.replace(None);
    }

    pub fn complete_concretize(
        self: &Rc<Self>,
        input: &[u8],
        pool: &mut ExprPool<ExprWrapper>,
    ) -> Option<Expr> {
        let replacements = Rc::new(
            self.variables()
                .iter()
                .map(|&v| (v, input[v]))
                .collect::<Vec<_>>(),
        );
        self.concretize(&replacements, self.variables().as_slice(), pool)
    }

    pub fn concretize(
        self: &Rc<Self>,
        parent_replacements: &Rc<InputReplacements>,
        parent_variables: &[usize],
        pool: &mut ExprPool<ExprWrapper>,
    ) -> Option<Expr> {
        let replacements = if parent_variables.len() == self.variables().len() {
            parent_replacements.clone()
        } else {
            let res = sorted_deduped_vec_intersection(self.variables(), parent_replacements);
            if res.is_empty() {
                return None;
            }
            Rc::new(res)
        };
        if replacements.len() == 0 {
            return None;
        }

        let res = if Rc::strong_count(self) <= 2 {
            self.do_concretize(&replacements, pool)
        } else {
            //println!("{:#?}", self.replacements.borrow());
            self.replacements
                .borrow_mut()
                .entry(replacements.clone())
                .or_insert_with(|| self.do_concretize(&replacements, pool))
                .clone()
        };

        if replacements.len() > 0 {
            assert!(res.is_some());
        }
        if self.variables.len() - replacements.len() == 0 && self.variables.len() > 0 {
            match &res {
                Some(Expr::True | Expr::False | Expr::Integer { .. }) => {}
                _ => panic!("unexpected expr as result of replacement: {:?}", res),
            }
        }
        res
    }

    fn do_concretize(
        self: &Rc<Self>,
        replacements: &Rc<InputReplacements>,
        pool: &mut ExprPool<ExprWrapper>,
    ) -> Option<Expr> {
        macro_rules! rewrite {
                ($op:ident => $constructor:path, $parent_replacements:ident, $parent_variables:ident, $pool:ident) => {
        $op.inner.concretize($parent_replacements, $parent_variables, $pool).map(|rewritten_child|
            $constructor(pool.alloc(rewritten_child.into())).const_eval()
        )
                };
                ($op:ident => $constructor:path {$($extra_var_name:ident),*}, $parent_replacements:ident, $parent_variables:ident, $pool:ident) => {
        $op.inner.concretize($parent_replacements, $parent_variables, $pool).map(|rewritten_child| {
                $constructor {
                    op: pool.alloc(rewritten_child.into()),
                    $( $extra_var_name: *$extra_var_name , )*
                }.const_eval()
            }
        )
                };
                ($a:ident, $b:ident => $constructor:path, $parent_replacements:ident, $parent_variables:ident, $pool:ident) => {
        match (
            $a.inner.concretize($parent_replacements, $parent_variables, $pool),
            $b.inner.concretize($parent_replacements, $parent_variables, $pool),
        ) {
            (Some(a_rewritten), Some(b_rewritten)) => Some(
                $constructor(
                    pool.alloc(a_rewritten.into()),
                    pool.alloc(b_rewritten.into()),
                )
                .const_eval(),
            ),
            (Some(a_rewritten), None) => {
                Some($constructor(pool.alloc(a_rewritten.into()), $b.clone()).const_eval())
            }
            (None, Some(b_rewritten)) => {
                Some($constructor($a.clone(), pool.alloc(b_rewritten.into())).const_eval())
            }
            (None, None) => None,
        }
                };
            }
        let parent_replacements = replacements;
        let parent_variables: &[usize] = &self.variables;
        match &self.expr {
            Expr::Variable { offset } => replacements
                .binary_search_by_key(offset, |&(offset, _)| offset)
                .map(|idx| Expr::Integer(ApInt::from_u8(replacements[idx].1).into()))
                .ok(),
            Expr::Integer { .. } | Expr::True | Expr::False => None,
            Expr::Neg(op) => rewrite!(op => Expr::Neg, parent_replacements, parent_variables, pool),
            Expr::Not(op) => rewrite!(op => Expr::Not, parent_replacements, parent_variables, pool),
            Expr::Sext { op, bits } => {
                rewrite!(op => Expr::Sext { bits }, parent_replacements, parent_variables, pool)
            }
            Expr::Zext { op, bits } => {
                rewrite!(op => Expr::Zext { bits }, parent_replacements, parent_variables, pool)
            }
            Expr::Trunc { op, bits } => {
                rewrite!(op => Expr::Trunc { bits }, parent_replacements, parent_variables, pool)
            }
            Expr::Bswap(op) => {
                rewrite!(op => Expr::Bswap, parent_replacements, parent_variables, pool)
            }
            Expr::BoolToBits { op, bits } => {
                rewrite!(op => Expr::BoolToBits { bits }, parent_replacements, parent_variables, pool)
            }
            Expr::Extract {
                op,
                first_bit,
                last_bit,
            } => {
                rewrite!(op => Expr::Extract { first_bit, last_bit }, parent_replacements, parent_variables, pool)
            }
            Expr::Add(a, b) => {
                rewrite!(a,b => Expr::Add, parent_replacements, parent_variables, pool)
            }
            Expr::Sub(a, b) => {
                rewrite!(a,b => Expr::Sub, parent_replacements, parent_variables, pool)
            }
            Expr::Mul(a, b) => {
                rewrite!(a,b => Expr::Mul, parent_replacements, parent_variables, pool)
            }
            Expr::UDiv(a, b) => {
                rewrite!(a,b => Expr::UDiv, parent_replacements, parent_variables, pool)
            }
            Expr::SDiv(a, b) => {
                rewrite!(a,b => Expr::SDiv, parent_replacements, parent_variables, pool)
            }
            Expr::URem(a, b) => {
                rewrite!(a,b => Expr::URem, parent_replacements, parent_variables, pool)
            }
            Expr::SRem(a, b) => {
                rewrite!(a,b => Expr::SRem, parent_replacements, parent_variables, pool)
            }
            Expr::Shl(a, b) => {
                rewrite!(a,b => Expr::Shl, parent_replacements, parent_variables, pool)
            }
            Expr::Lshr(a, b) => {
                rewrite!(a,b => Expr::Lshr, parent_replacements, parent_variables, pool)
            }
            Expr::Ashr(a, b) => {
                rewrite!(a,b => Expr::Ashr, parent_replacements, parent_variables, pool)
            }
            Expr::Slt(a, b) => {
                rewrite!(a,b => Expr::Slt, parent_replacements, parent_variables, pool)
            }
            Expr::Sle(a, b) => {
                rewrite!(a,b => Expr::Sle, parent_replacements, parent_variables, pool)
            }
            Expr::Sgt(a, b) => {
                rewrite!(a,b => Expr::Sgt, parent_replacements, parent_variables, pool)
            }
            Expr::Sge(a, b) => {
                rewrite!(a,b => Expr::Sge, parent_replacements, parent_variables, pool)
            }
            Expr::Ult(a, b) => {
                rewrite!(a,b => Expr::Ult, parent_replacements, parent_variables, pool)
            }
            Expr::Ule(a, b) => {
                rewrite!(a,b => Expr::Ule, parent_replacements, parent_variables, pool)
            }
            Expr::Ugt(a, b) => {
                rewrite!(a,b => Expr::Ugt, parent_replacements, parent_variables, pool)
            }
            Expr::Uge(a, b) => {
                rewrite!(a,b => Expr::Uge, parent_replacements, parent_variables, pool)
            }
            Expr::Equal(a, b) => {
                rewrite!(a,b => Expr::Equal, parent_replacements, parent_variables, pool)
            }
            Expr::NotEqual(a, b) => {
                rewrite!(a,b => Expr::NotEqual, parent_replacements, parent_variables, pool)
            }
            Expr::BoolOr(a, b) => {
                rewrite!(a,b => Expr::BoolOr, parent_replacements, parent_variables, pool)
            }
            Expr::BoolAnd(a, b) => {
                rewrite!(a,b => Expr::BoolAnd, parent_replacements, parent_variables, pool)
            }
            Expr::BoolXor(a, b) => {
                rewrite!(a,b => Expr::BoolXor, parent_replacements, parent_variables, pool)
            }
            Expr::And(a, b) => {
                rewrite!(a,b => Expr::And, parent_replacements, parent_variables, pool)
            }
            Expr::Or(a, b) => {
                rewrite!(a,b => Expr::Or, parent_replacements, parent_variables, pool)
            }
            Expr::Xor(a, b) => {
                rewrite!(a,b => Expr::Xor, parent_replacements, parent_variables, pool)
            }
            Expr::Concat(a, b) => {
                rewrite!(a,b => Expr::Concat, parent_replacements, parent_variables, pool)
            }
        }
    }

    pub fn collect_nodes<'l>(&'l self, set: &mut HashSet<&'l ExprWrapper>) {
        if set.insert(self) {
            match &**self {
                Expr::Variable { .. } | Expr::Integer { .. } | Expr::True | Expr::False => (),
                Expr::Neg(op)
                | Expr::Not(op)
                | Expr::Sext { op, .. }
                | Expr::Zext { op, .. }
                | Expr::Trunc { op, .. }
                | Expr::Bswap(op)
                | Expr::BoolToBits { op, .. }
                | Expr::Extract { op, .. } => op.collect_nodes(set),
                Expr::Add(a, b)
                | Expr::Sub(a, b)
                | Expr::Mul(a, b)
                | Expr::UDiv(a, b)
                | Expr::SDiv(a, b)
                | Expr::URem(a, b)
                | Expr::SRem(a, b)
                | Expr::Shl(a, b)
                | Expr::Lshr(a, b)
                | Expr::Ashr(a, b)
                | Expr::Slt(a, b)
                | Expr::Sle(a, b)
                | Expr::Sgt(a, b)
                | Expr::Sge(a, b)
                | Expr::Ult(a, b)
                | Expr::Ule(a, b)
                | Expr::Ugt(a, b)
                | Expr::Uge(a, b)
                | Expr::Equal(a, b)
                | Expr::NotEqual(a, b)
                | Expr::BoolOr(a, b)
                | Expr::BoolAnd(a, b)
                | Expr::BoolXor(a, b)
                | Expr::And(a, b)
                | Expr::Or(a, b)
                | Expr::Xor(a, b)
                | Expr::Concat(a, b) => {
                    a.collect_nodes(set);
                    b.collect_nodes(set);
                }
            }
        }
    }
}

impl From<Expr> for ExprWrapper {
    fn from(expr: Expr) -> Self {
        Self::new(expr)
    }
}

impl Expr {
    fn collect_variables(&self) -> Rc<Vec<usize>> {
        match self {
            Expr::Variable { offset } => Rc::new(vec![*offset]),
            Expr::Integer { .. } | Expr::True | Expr::False => Rc::new(Vec::new()),
            Expr::Neg(op)
            | Expr::Not(op)
            | Expr::Sext { op, .. }
            | Expr::Zext { op, .. }
            | Expr::Trunc { op, .. }
            | Expr::Bswap(op)
            | Expr::BoolToBits { op, .. }
            | Expr::Extract { op, .. } => op.variables.clone(),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::UDiv(a, b)
            | Expr::SDiv(a, b)
            | Expr::URem(a, b)
            | Expr::SRem(a, b)
            | Expr::Shl(a, b)
            | Expr::Lshr(a, b)
            | Expr::Ashr(a, b)
            | Expr::Slt(a, b)
            | Expr::Sle(a, b)
            | Expr::Sgt(a, b)
            | Expr::Sge(a, b)
            | Expr::Ult(a, b)
            | Expr::Ule(a, b)
            | Expr::Ugt(a, b)
            | Expr::Uge(a, b)
            | Expr::Equal(a, b)
            | Expr::NotEqual(a, b)
            | Expr::BoolOr(a, b)
            | Expr::BoolAnd(a, b)
            | Expr::BoolXor(a, b)
            | Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Xor(a, b)
            | Expr::Concat(a, b) => {
                let vars_a = &a.variables;
                let vars_b = &b.variables;
                match (vars_a.len(), vars_b.len()) {
                    (0, _) => vars_b.clone(),
                    (_, 0) => vars_a.clone(),
                    _ => {
                        let mut res = Vec::new();
                        res.extend(vars_a.iter().copied());
                        res.extend(vars_b.iter().copied());
                        res.sort_unstable();
                        res.dedup();
                        Rc::new(res)
                    }
                }
            }
        }
    }

    pub fn is_concrete(&self) -> bool {
        match self {
            Expr::Variable { .. } => false,
            Expr::Integer { .. } | Expr::True | Expr::False => true,
            Expr::Neg(op)
            | Expr::Not(op)
            | Expr::Sext { op, .. }
            | Expr::Zext { op, .. }
            | Expr::Trunc { op, .. }
            | Expr::Bswap(op)
            | Expr::BoolToBits { op, .. }
            | Expr::Extract { op, .. } => op.variables.len() == 0,
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::UDiv(a, b)
            | Expr::SDiv(a, b)
            | Expr::URem(a, b)
            | Expr::SRem(a, b)
            | Expr::Shl(a, b)
            | Expr::Lshr(a, b)
            | Expr::Ashr(a, b)
            | Expr::Slt(a, b)
            | Expr::Sle(a, b)
            | Expr::Sgt(a, b)
            | Expr::Sge(a, b)
            | Expr::Ult(a, b)
            | Expr::Ule(a, b)
            | Expr::Ugt(a, b)
            | Expr::Uge(a, b)
            | Expr::Equal(a, b)
            | Expr::NotEqual(a, b)
            | Expr::BoolOr(a, b)
            | Expr::BoolAnd(a, b)
            | Expr::BoolXor(a, b)
            | Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Xor(a, b)
            | Expr::Concat(a, b) => a.variables().len() == 0 && b.variables().len() == 0,
        }
    }
}

impl ExprWrapper {
    pub fn to_z3<'ctx>(self: &Rc<Self>, ctx: &'ctx Context) -> Dynamic<'ctx> {
        self.to_z3_internal(ctx, Rc::strong_count(self) > 2)
    }

    fn to_z3_internal<'ctx>(&self, ctx: &'ctx Context, cache_result: bool) -> Dynamic<'ctx> {
        let mut cache = self.z3_expr.borrow_mut();
        if let Some(cached) = &*cache {
            cached.clone()
        } else {
            macro_rules! construct_bv {
                ($a:ident . $method:tt ($b:ident)) => {
                    $a.inner
                        .to_z3(ctx)
                        .as_bv()
                        .unwrap()
                        .$method(&$b.inner.to_z3(ctx).as_bv().unwrap())
                        .into()
                };
                ($a:ident . $method:tt ()) => {
                    $a.inner.to_z3(ctx).as_bv().unwrap().$method().into()
                };
            }

            let ret: Dynamic<'ctx> = match self.deref().deref() {
                Expr::Variable { offset } => {
                    BV::new_const(ctx, Symbol::Int(u32::try_from(*offset).unwrap()), 8).into()
                }
                Expr::Integer(i) => BV::from_u64(
                    ctx,
                    i.try_to_u64().unwrap(),
                    u32::try_from(i.width().to_usize()).unwrap(),
                )
                .into(),
                Expr::True => Bool::from_bool(ctx, true).into(),
                Expr::False => Bool::from_bool(ctx, false).into(),
                Expr::Neg(op) => op.inner.to_z3(ctx).as_bv().unwrap().bvneg().into(),
                Expr::Add(a, b) => construct_bv!(a.bvadd(b)),
                Expr::Sub(a, b) => construct_bv!(a.bvsub(b)),
                Expr::Mul(a, b) => construct_bv!(a.bvmul(b)),
                Expr::UDiv(a, b) => construct_bv!(a.bvudiv(b)),
                Expr::SDiv(a, b) => construct_bv!(a.bvsdiv(b)),
                Expr::URem(a, b) => construct_bv!(a.bvurem(b)),
                Expr::SRem(a, b) => construct_bv!(a.bvsrem(b)),
                Expr::Shl(a, b) => construct_bv!(a.bvshl(b)),
                Expr::Lshr(a, b) => construct_bv!(a.bvlshr(b)),
                Expr::Ashr(a, b) => construct_bv!(a.bvashr(b)),
                Expr::Not(op) => {
                    let z3 = op.inner.to_z3(ctx);
                    if let Some(bv) = z3.as_bv() {
                        bv.bvnot().into()
                    } else if let Some(bool) = z3.as_bool() {
                        bool.not().into()
                    } else {
                        panic!(
                            "unexpected z3 expr of type {:?} when applying not operation",
                            z3.kind()
                        )
                    }
                }
                Expr::Slt(a, b) => construct_bv!(a.bvslt(b)),
                Expr::Sle(a, b) => construct_bv!(a.bvsle(b)),
                Expr::Sgt(a, b) => construct_bv!(a.bvsgt(b)),
                Expr::Sge(a, b) => construct_bv!(a.bvsge(b)),
                Expr::Ult(a, b) => construct_bv!(a.bvult(b)),
                Expr::Ule(a, b) => construct_bv!(a.bvule(b)),
                Expr::Ugt(a, b) => construct_bv!(a.bvugt(b)),
                Expr::Uge(a, b) => construct_bv!(a.bvuge(b)),
                Expr::Equal(a, b) => a.inner.to_z3(ctx)._eq(&b.inner.to_z3(ctx)).into(),
                Expr::NotEqual(a, b) => a.inner.to_z3(ctx)._eq(&b.inner.to_z3(ctx)).not().into(),
                Expr::BoolOr(a, b) => Bool::or(
                    ctx,
                    &[
                        &a.inner.to_z3(ctx).as_bool().unwrap(),
                        &b.inner.to_z3(ctx).as_bool().unwrap(),
                    ],
                )
                .into(),
                Expr::BoolAnd(a, b) => Bool::and(
                    ctx,
                    &[
                        &a.inner.to_z3(ctx).as_bool().unwrap(),
                        &b.inner.to_z3(ctx).as_bool().unwrap(),
                    ],
                )
                .into(),
                Expr::BoolXor(a, b) => a
                    .inner
                    .to_z3(ctx)
                    .as_bool()
                    .unwrap()
                    .xor(&b.inner.to_z3(ctx).as_bool().unwrap())
                    .into(),
                Expr::And(a, b) => construct_bv!(a.bvand(b)),
                Expr::Or(a, b) => construct_bv!(a.bvor(b)),
                Expr::Xor(a, b) => construct_bv!(a.bvxor(b)),
                Expr::Sext { op, bits } => op
                    .inner
                    .to_z3(ctx)
                    .as_bv()
                    .unwrap()
                    .sign_ext(u32::from(*bits))
                    .into(),
                Expr::Zext { op, bits } => op
                    .inner
                    .to_z3(ctx)
                    .as_bv()
                    .unwrap()
                    .zero_ext(u32::from(*bits))
                    .into(),
                Expr::Trunc { op, bits } => op
                    .inner
                    .to_z3(ctx)
                    .as_bv()
                    .unwrap()
                    .extract(u32::from(*bits - 1), 0)
                    .into(),
                Expr::Bswap(op) => {
                    let bv = op.inner.to_z3(ctx).as_bv().unwrap();
                    let total_bits = bv.get_size();
                    let mut result = bv.extract(total_bits - 1, total_bits - 8);
                    for i in (8..(total_bits)).step_by(8) {
                        result = bv
                            .extract(total_bits - i - 1, total_bits - (i + 8))
                            .concat(&result);
                    }
                    result.into()
                }
                Expr::BoolToBits { op, bits } => op
                    .inner
                    .to_z3(ctx)
                    .as_bool()
                    .unwrap()
                    .ite(
                        &BV::from_u64(ctx, 1, u32::from(*bits)),
                        &BV::from_u64(ctx, 0, u32::from(*bits)),
                    )
                    .into(),
                Expr::Concat(a, b) => construct_bv!(a.concat(b)),
                Expr::Extract {
                    op,
                    first_bit,
                    last_bit,
                } => op
                    .inner
                    .to_z3(ctx)
                    .as_bv()
                    .unwrap()
                    .extract(
                        u32::try_from(*first_bit).unwrap(),
                        u32::try_from(*last_bit).unwrap(),
                    )
                    .into(),
            };
            if cache_result {
                // increase ref counter of the ast node
                let clone = unsafe {
                    // we are extending the lifetime of the ast node here.
                    // this is, of course, really unsafe.
                    // the argument for safety is that the only way to get to this Dynamic with extendend lifetime
                    // is via the private member of this struct, which nobody has access to.
                    // The only other use is in dealloc_z3, where we need to pass in a Context as witness.
                    // this mustn't necessarily be the context that was used to construct the AST node,
                    // but this is not even guaranteed with the rust z3 API. therefore, this isn't more unsafe than it
                    // was before.
                    std::mem::transmute::<&Dynamic<'ctx>, &Dynamic<'static>>(&ret).clone()
                };
                *cache = Some(clone);
            }
            ret
        }
    }

    pub fn to_z3_simplified<'ctx>(
        &self,
        ctx: &'ctx Context,
        cache: bool,
    ) -> Ref<'_, Dynamic<'ctx>> {
        {
            // check cached value
            let r = self.z3_expr_simplified.borrow();
            if r.is_some() {
                return Ref::map(r, |o| o.as_ref().unwrap());
            }
        }
        let res = self.to_z3_internal(ctx, false).simplify();
        if cache {
            let mut cache = self.z3_expr_simplified.borrow_mut();
            cache.replace(unsafe { std::mem::transmute::<Dynamic<'ctx>, Dynamic<'static>>(res) });
        }
        let r = self.z3_expr_simplified.borrow();

        fn add_ctx_lt<'b, 'ctx>(d: &'b Dynamic<'static>) -> &'b Dynamic<'ctx> {
            unsafe { std::mem::transmute(d) }
        }

        Ref::map(r, |o| o.as_ref().map(add_ctx_lt).unwrap())
    }
}
