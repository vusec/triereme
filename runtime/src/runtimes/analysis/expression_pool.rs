use std::{
    fmt::Debug,
    hash::{BuildHasherDefault, Hash, Hasher},
    rc::Rc,
};

use rustc_hash::FxHasher;

pub trait Expr: Hash + PartialEq + Eq {}

impl<T: Hash + PartialEq + Eq> Expr for T {}

#[derive(Eq, Clone)]
pub struct ExprRef<T: Expr> {
    pub(crate) inner: Rc<T>,
}

impl<T: Expr> ExprRef<T> {
    fn new(inner: T) -> Self {
        Self {
            inner: Rc::new(inner),
        }
    }

    pub fn strong_count(this: &Self) -> usize {
        Rc::strong_count(&this.inner)
    }
}

impl<T: Expr> Hash for ExprRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.inner) as usize).hash(state);
    }
}

impl<T: Expr> PartialEq for ExprRef<T> {
    fn eq(&self, other: &Self) -> bool {
        (Rc::as_ptr(&self.inner) as usize).eq(&(Rc::as_ptr(&other.inner) as usize))
    }
}

impl<T: Expr + PartialOrd> PartialOrd for ExprRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (Rc::as_ptr(&self.inner) as usize).partial_cmp(&(Rc::as_ptr(&other.inner) as usize))
    }
}

impl<T: Expr + Ord> Ord for ExprRef<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (Rc::as_ptr(&self.inner) as usize).cmp(&(Rc::as_ptr(&other.inner) as usize))
    }
}

impl<T: Expr + Debug> Debug for ExprRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (&**self).fmt(f)
    }
}

impl<T: Expr> std::ops::Deref for ExprRef<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

#[derive(Eq, Clone)]
struct ValueEquality<T: Expr> {
    expr: ExprRef<T>,
}

impl<T: Expr> ValueEquality<T> {
    fn new(v: ExprRef<T>) -> Self {
        Self { expr: v }
    }
}

impl<T: Expr> Hash for ValueEquality<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.inner.as_ref().hash(state);
    }
}

impl<T: Expr> PartialEq for ValueEquality<T> {
    fn eq(&self, other: &Self) -> bool {
        self.expr.inner.as_ref().eq(other.expr.inner.as_ref())
    }
}

pub struct ExprPool<T: Expr> {
    values: hashbrown::HashSet<ValueEquality<T>, BuildHasherDefault<FxHasher>>,
    deduped_exprs: usize,
    garbage_collected: usize,
    last_gc: usize,
}

#[derive(Default, Debug)]
pub struct ExprPoolStats {
    deduped_exprs: usize,
    garbage_collected: usize,
    stored_exprs: usize,
}

impl<T: Expr + Clone> Default for ExprPool<T> {
    fn default() -> Self {
        Self {
            values: hashbrown::HashSet::default(),
            deduped_exprs: 0,
            garbage_collected: 0,
            last_gc: 0,
        }
    }
}

impl<T: Expr + Clone> ExprPool<T> {
    pub fn alloc(&mut self, v: T) -> ExprRef<T> {
        let res = {
            let exprref = ExprRef::new(v);
            self.values
                .get_or_insert_owned(&ValueEquality::new(exprref))
                .expr
                .clone()
        };
        if ExprRef::strong_count(&res) > 2 {
            self.deduped_exprs += 1;
        }
        res
    }

    pub fn iter(&self) -> impl Iterator<Item = &ExprRef<T>> {
        self.values.iter().map(|e| &e.expr)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn alive_len(&self) -> usize {
        self.values
            .iter()
            .map(|e| ExprRef::strong_count(&e.expr) - 1)
            .sum()
    }

    pub fn gc(&mut self) -> impl Iterator<Item = ExprRef<T>> + '_ {
        self.last_gc = self.values.len();
        let values = &mut self.values;
        let garbage_collected = &mut self.garbage_collected;
        let last_gc = &mut self.last_gc;
        values
            .drain_filter(|e| ExprRef::strong_count(&e.expr) == 1)
            .map(move |e| {
                *last_gc -= 1;
                *garbage_collected += 1;
                e.expr
            })
    }

    pub fn gc_point(&mut self) -> impl Iterator<Item = ExprRef<T>> + '_ {
        ((self.values.len() - self.last_gc) > 16384)
            .then(move || self.gc())
            .into_iter()
            .flatten()
    }

    pub fn stats(&self) -> ExprPoolStats {
        ExprPoolStats {
            garbage_collected: self.garbage_collected,
            deduped_exprs: self.deduped_exprs,
            stored_exprs: self.values.len(),
        }
    }
}
