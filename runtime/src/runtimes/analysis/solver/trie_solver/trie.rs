use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Peekable;

use hashbrown::{hash_set, HashMap, HashSet};
use itertools::Itertools;

#[derive(Debug)]
enum TrieChildren<KE, V: Eq + Hash> {
    None,
    Single {
        child_key: Vec<KE>,
        child: Box<Trie<KE, V>>,
    },
    Multiple {
        children: HashMap<KE, Trie<KE, V>>,
    },
}

#[derive(Debug)]
pub struct Trie<KE, V: Eq + Hash> {
    value: HashSet<V>,
    children: TrieChildren<KE, V>,
}

struct TrieValueIterator<'l, KE, V: Eq + Hash> {
    root_values: hash_set::Iter<'l, V>,
    current_path: Vec<TrieChildrenIterator<'l, KE, V>>,
}

impl<'l, KE, V: Eq + Hash> TrieValueIterator<'l, KE, V> {
    fn new(trie: &'l Trie<KE, V>) -> Self {
        Self {
            root_values: trie.value.iter(),
            current_path: vec![trie.children.iter()],
        }
    }
}

impl<'l, KE, V: Eq + Hash> Iterator for TrieValueIterator<'l, KE, V> {
    type Item = &'l V;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(root_value) = self.root_values.next() {
            Some(root_value)
        } else {
            while let Some(mut last_element) = self.current_path.pop() {
                if let Some(next_child) = last_element.next() {
                    self.current_path.push(last_element);
                    let mut iter = next_child.child.value.iter();
                    self.current_path.push(next_child.child.children.iter());
                    if let Some(value) = iter.next() {
                        self.root_values = iter;
                        return Some(value);
                    }
                }
            }
            None
        }
    }
}

impl<KE, V: Eq + Hash> TrieChildren<KE, V> {
    fn iter(&'_ self) -> TrieChildrenIterator<'_, KE, V> {
        match self {
            TrieChildren::None => TrieChildrenIterator::None,
            TrieChildren::Single { child_key, child } => TrieChildrenIterator::Single {
                key: child_key,
                child,
                is_finished: false,
            },
            TrieChildren::Multiple { children } => TrieChildrenIterator::Multiple {
                iter: {
                    let mut vec = children.iter().collect_vec();
                    vec.sort_by_cached_key(|(_, t)| t.weight());
                    vec.into_iter()
                },
            },
        }
    }
}

enum TrieChildrenIterator<'l, KE, V: Eq + Hash> {
    None,
    Single {
        key: &'l Vec<KE>,
        child: &'l Trie<KE, V>,
        is_finished: bool,
    },
    Multiple {
        iter: std::vec::IntoIter<(&'l KE, &'l Trie<KE, V>)>,
    },
}

impl<'l, KE, V: Eq + Hash> Iterator for TrieChildrenIterator<'l, KE, V> {
    type Item = TriePathComponent<'l, KE, V>;

    fn next(&mut self) -> Option<Self::Item> {
        match *self {
            TrieChildrenIterator::None => None,
            TrieChildrenIterator::Single {
                key,
                child,
                ref mut is_finished,
            } => {
                if *is_finished {
                    None
                } else {
                    *is_finished = true;
                    Some(TriePathComponent {
                        key_elements: TriePathComponentElementIter::Single { iter: key.iter() },
                        child,
                    })
                }
            }
            TrieChildrenIterator::Multiple { ref mut iter } => {
                if let Some((key, child)) = iter.next() {
                    Some(TriePathComponent {
                        key_elements: TriePathComponentElementIter::Multiple {
                            iter: std::iter::once(key),
                        },
                        child,
                    })
                } else {
                    None
                }
            }
        }
    }
}

pub(crate) enum TrieVisitorStep<'l, KE, V: Eq + Hash> {
    PushPathComponentElements(TriePathComponentElementIter<'l, KE>),
    PopPath(usize),
    Value(&'l HashSet<V>),
}

pub(crate) struct TrieVisitor<'l, KE, V: Eq + Hash> {
    next_value: Option<&'l HashSet<V>>,
    current_path: Vec<TrieChildrenIterator<'l, KE, V>>,
    path_component_sizes: Vec<usize>,
}

impl<'l, KE, V: Eq + Hash> TrieVisitor<'l, KE, V> {
    fn new(trie: &'l Trie<KE, V>) -> Self {
        Self {
            next_value: if trie.value.is_empty() {
                None
            } else {
                Some(&trie.value)
            },
            current_path: vec![trie.children.iter()],
            path_component_sizes: vec![],
        }
    }
}

impl<'l, KE, V: Eq + Hash> Iterator for TrieVisitor<'l, KE, V> {
    type Item = TrieVisitorStep<'l, KE, V>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next_value) = self.next_value.take() {
            Some(TrieVisitorStep::Value(next_value))
        } else {
            while let Some(mut last_element) = self.current_path.pop() {
                if let Some(next_child) = last_element.next() {
                    self.current_path.push(last_element);
                    if !next_child.child.value.is_empty() {
                        self.next_value = Some(&next_child.child.value);
                    }
                    self.current_path.push(next_child.child.children.iter());
                    self.path_component_sizes
                        .push(next_child.key_elements.size_hint().1.unwrap());
                    return Some(TrieVisitorStep::PushPathComponentElements(
                        next_child.key_elements,
                    ));
                } else if let Some(n) = self.path_component_sizes.pop() {
                    return Some(TrieVisitorStep::PopPath(n));
                }
            }
            None
        }
    }
}

pub(crate) enum TriePathComponentElementIter<'l, KE> {
    Single { iter: std::slice::Iter<'l, KE> },
    Multiple { iter: std::iter::Once<&'l KE> },
}

impl<'l, KE> Iterator for TriePathComponentElementIter<'l, KE> {
    type Item = &'l KE;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TriePathComponentElementIter::Single { iter } => iter.next(),
            TriePathComponentElementIter::Multiple { iter } => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            TriePathComponentElementIter::Single { iter } => iter.size_hint(),
            TriePathComponentElementIter::Multiple { iter } => iter.size_hint(),
        }
    }
}

struct TriePathComponent<'l, KE, V: Eq + Hash> {
    key_elements: TriePathComponentElementIter<'l, KE>,
    child: &'l Trie<KE, V>,
}

pub trait TrieDotFormatter<KE, V> {
    fn attributes(
        &mut self,
        ke: &KE,
        id: usize,
        label: usize,
        values: &HashSet<V>,
    ) -> HashMap<String, String>;
}

impl<KE, V: Eq + Hash> Trie<KE, V> {
    /// Returns the amount of Key Elements stored inside the tree. This number should be less than or equal to the number
    /// of inserted key elements.
    pub fn weight(&self) -> usize {
        let children_count = match &self.children {
            TrieChildren::None => 0,
            TrieChildren::Single { child_key, child } => child_key.len() + child.weight(),
            TrieChildren::Multiple { children } => {
                children.values().map(Trie::weight).sum::<usize>() + children.len()
            }
        };
        children_count
    }

    /// Returns the number of values stored inside the tree.
    pub fn len(&self) -> usize {
        let self_count = self.value.len();
        let children_count = match &self.children {
            TrieChildren::None => 0,
            TrieChildren::Single { child, .. } => child.len(),
            TrieChildren::Multiple { children } => children.values().map(Trie::len).sum::<usize>(),
        };
        self_count + children_count
    }
}

impl<KE: Hash + PartialEq + Eq + Debug + Ord, V: Debug + Eq + Hash> Trie<KE, V> {
    pub fn new() -> Self {
        Self {
            value: HashSet::new(),
            children: TrieChildren::None,
        }
    }

    fn split_into_common_prefix<Iter: Iterator<Item = KE>>(
        k1: &mut Vec<KE>,
        k2: &mut Peekable<Iter>,
    ) -> Vec<KE> {
        let mut split_point = 0;
        while let Some(peeked) = k2.peek() {
            if split_point < k1.len() && &k1[split_point] == peeked {
                std::mem::drop(k2.next());
                split_point += 1;
            } else {
                break;
            }
        }

        if split_point == 0 {
            // no equal elements
            Vec::new()
        } else {
            match (split_point == k1.len(), k2.peek().is_none()) {
                (false, _) => {
                    let new_k1 = k1.split_off(split_point);
                    std::mem::replace(k1, new_k1)
                }
                (true, _) => std::mem::take(k1),
            }
        }
    }

    pub fn insert<KeyIter: Iterator<Item = KE>>(&mut self, k: KeyIter, v: V) -> Option<V> {
        self.insert_internal2(k.peekable(), Some(v))
    }

    fn insert_internal2<KeyIter: Iterator<Item = KE>>(
        mut self: &mut Self,
        mut k: Peekable<KeyIter>,
        v: Option<V>,
    ) -> Option<V> {
        loop {
            // Within this loop, `self` is used to identify the current node in
            // the traversal of the trie.

            // The key sequence is finished, so the current node is supposed to
            // hold the inserted value.
            if k.peek().is_none() {
                return v.and_then(|v| self.value.replace(v));
            }

            match self.children {
                TrieChildren::None => {
                    // The current node does not have children, add a new one
                    // which represents the remaining key sequence and the
                    // inserted value.

                    let child = Box::new(Self {
                        value: v.into_iter().collect(),
                        children: TrieChildren::None,
                    });

                    self.children = TrieChildren::Single {
                        child_key: k.collect(),
                        child,
                    };

                    return None;
                }
                TrieChildren::Single {
                    ref mut child_key, ..
                } => {
                    let common_prefix = Self::split_into_common_prefix(child_key, &mut k);
                    //dbg!(&common_prefix, &child_key, &k);

                    match (
                        child_key.is_empty(),
                        k.peek().is_none(),
                        common_prefix.is_empty(),
                    ) {
                        (false, _, false) => {
                            // split current node into two
                            // self will become the common prefix and we create a new node that points to our original child and value
                            let new_child_key = std::mem::replace(child_key, common_prefix);
                            // replace our child with an empty dummy node
                            let new_child = Box::new(Trie {
                                value: HashSet::new(),
                                children: TrieChildren::None,
                            });

                            let child =
                                if let TrieChildren::Single { ref mut child, .. } = self.children {
                                    child
                                } else {
                                    unreachable!();
                                };
                            let old_child = std::mem::replace(child, new_child);
                            // append the old child as a child of the new child
                            // note, that child is now new_child
                            child.children = TrieChildren::Single {
                                child_key: new_child_key,
                                child: old_child,
                            };
                            // finally, we insert the new key, which should now have an empty common prefix with our new child
                            self = child;
                        }
                        (false, false, true) => {
                            // we don't have a common prefix and both the current prefix and the key are non-empty
                            // we need to split using multiple children here
                            // self will become a multiple children trie node with two children
                            // first, we will turn ourself into a dummy multiple children node with empty children
                            let mut old_children = std::mem::replace(
                                &mut self.children,
                                TrieChildren::Multiple {
                                    children: HashMap::new(),
                                },
                            );
                            // now we insert our original self as one of the children of the new self
                            // note, that the old_children is definitely a TrieChildren::Single
                            let (new_child_key_first, new_child_key_rest_len) =
                                if let TrieChildren::Single {
                                    ref mut child_key, ..
                                } = old_children
                                {
                                    (child_key.remove(0), child_key.len())
                                } else {
                                    unreachable!();
                                };
                            let new_child = if new_child_key_rest_len == 0 {
                                if let TrieChildren::Single { child, .. } = old_children {
                                    *child
                                } else {
                                    unreachable!();
                                }
                            } else {
                                Trie {
                                    value: HashSet::new(),
                                    children: old_children,
                                }
                            };
                            if let TrieChildren::Multiple { ref mut children } = self.children {
                                assert!(children.insert(new_child_key_first, new_child).is_none());
                                // finally, we create the new subtree for the key to be inserted
                                let key = k.next().unwrap();
                                let mut subtree = Trie::new();
                                assert!(subtree.insert_internal2(k, v).is_none());
                                assert!(children.insert(key, subtree).is_none());
                                return None;
                            }
                            unreachable!();
                        }
                        (true, _, false) => {
                            // we have a common prefix that is exactly our own key.
                            // therefore the new key belongs into our subtree
                            // we restore the common prefix as our own key here
                            *child_key = common_prefix;
                            let child =
                                if let TrieChildren::Single { ref mut child, .. } = self.children {
                                    child
                                } else {
                                    unreachable!();
                                };
                            self = child;
                        }
                        (false, true, true) => {
                            // we don't have a common prefix and k is also empty
                            // this is impossible, because then the if statement at the beginning of the function
                            // would have already been triggered
                            unreachable!();
                        }
                        (true, _, true) => {
                            // we don't have a common prefix and also our own key is empty. this is a bug.
                            unreachable!();
                        }
                    }
                }
                TrieChildren::Multiple { ref mut children } => {
                    // Edges exiting from a node with multiple children
                    // represent only a single element of the key sequence.

                    let next_key_elem = k.next().unwrap();
                    let next_node = children.entry(next_key_elem).or_insert_with(Trie::new);

                    self = next_node;
                }
            }
        }
    }

    #[allow(unused)]
    pub fn print_structure<W: std::io::Write, F: TrieDotFormatter<KE, V>>(
        &self,
        w: &mut W,
        mut formatter: F,
    ) -> std::io::Result<()> {
        let mut name_counter = 1_usize;
        let mut label_counter = 1_usize;
        let mut names = HashMap::new();
        writeln!(w, "digraph {{")?;
        self.print_structure_rec(
            w,
            0,
            &mut names,
            &mut name_counter,
            &mut label_counter,
            &mut formatter,
        )?;
        writeln!(w, "}}")?;
        Ok(())
    }

    #[allow(unstable_name_collisions)]
    fn print_structure_rec<'s, W: std::io::Write, F: TrieDotFormatter<KE, V>>(
        &'s self,
        w: &mut W,
        parent: usize,
        names: &mut HashMap<&'s KE, usize>,
        name_counter: &mut usize,
        label_counter: &mut usize,
        formatter: &mut F,
    ) -> std::io::Result<()> {
        let mut parents = Vec::new();
        let empty_hashset = HashSet::new();
        for c in self.children.iter() {
            let mut p = parent;
            let mut iter = c.key_elements.peekable();
            while let Some(ke) = iter.next() {
                *name_counter += 1;
                let ke_name = *name_counter;
                let label = names.entry(ke).or_insert_with(|| {
                    *label_counter += 1;
                    *label_counter
                });
                let attrs = Itertools::intersperse(
                    formatter
                        .attributes(
                            ke,
                            ke_name,
                            *label,
                            if iter.peek().is_some() {
                                &empty_hashset
                            } else {
                                &c.child.value
                            },
                        )
                        .into_iter()
                        .map(|(k, v)| format!("{}={}", k, v)),
                    ", ".to_string(),
                )
                .collect::<String>();
                writeln!(w, "{} [{}];", ke_name, attrs)?;
                writeln!(w, "{}->{};", p, ke_name)?;
                p = ke_name;
            }
            parents.push(p);
        }
        for (c, p) in self.children.iter().zip(parents) {
            c.child
                .print_structure_rec(w, p, names, name_counter, label_counter, formatter)?;
        }
        Ok(())
    }

    #[allow(unused)]
    pub fn keys(&self) -> Vec<Vec<&KE>> {
        self.keys_rec(&[])
    }

    fn keys_rec<'l>(&'l self, prefix: &[&'l KE]) -> Vec<Vec<&'l KE>> {
        let mut res = Vec::new();
        if !self.value.is_empty() {
            res.push(prefix.to_owned());
        }
        match &self.children {
            TrieChildren::None => {}
            TrieChildren::Single { child_key, child } => {
                let mut new_prefix = prefix.to_owned();
                new_prefix.extend(child_key.iter());
                res.extend(child.keys_rec(&new_prefix));
            }
            TrieChildren::Multiple { children } => {
                for (key, child) in children {
                    let mut new_prefix = prefix.to_owned();
                    new_prefix.push(key);
                    let child_res = child.keys_rec(&new_prefix);
                    res.extend(child_res);
                }
            }
        };
        res
    }

    #[allow(unused)]
    pub fn values(&self) -> impl Iterator<Item = &V> {
        TrieValueIterator::new(self)
    }

    pub(crate) fn visitor(&self) -> impl Iterator<Item = TrieVisitorStep<'_, KE, V>> {
        TrieVisitor::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::quickcheck;
    use std::collections::HashSet;

    #[test]
    fn test_name() {
        let mut trie = Trie::new();
        let v = vec![vec![0, 1], vec![0]];
        for v in v {
            trie.insert(v.into_iter(), ());
        }
    }

    fn generate_kvps(common_prefixes: Vec<Vec<u8>>, suffixes: Vec<Vec<u8>>) -> Vec<(Vec<u8>, u8)> {
        common_prefixes
            .iter()
            .flat_map(|prefix| {
                suffixes
                    .iter()
                    .map(|suffix| {
                        prefix
                            .iter()
                            .chain(suffix.iter())
                            .copied()
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(index, key)| (key, index as u8))
            .collect::<Vec<_>>()
    }

    #[quickcheck]
    fn check_len(common_prefixes: Vec<Vec<u8>>, suffixes: Vec<Vec<u8>>) -> bool {
        let mut trie = Trie::new();
        let kvps = generate_kvps(common_prefixes, suffixes);
        let kvps_len = kvps.len();
        for (key, value) in kvps {
            trie.insert(key.into_iter(), value);
        }
        kvps_len == trie.len()
    }

    #[quickcheck]
    fn check_keys(common_prefixes: Vec<Vec<u8>>, suffixes: Vec<Vec<u8>>) -> bool {
        let mut trie = Trie::new();
        let kvps = generate_kvps(common_prefixes, suffixes);
        for (key, value) in &kvps {
            trie.insert(key.clone().into_iter(), *value);
        }
        let mut stored_keys = trie
            .keys()
            .iter()
            .map(|k| k.iter().map(|v| **v).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        stored_keys.sort();
        let mut inserted_keys = kvps.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>();
        inserted_keys.sort();
        stored_keys == inserted_keys
    }

    #[quickcheck]
    fn check_values(common_prefixes: Vec<Vec<u8>>, suffixes: Vec<Vec<u8>>) -> bool {
        let mut trie = Trie::new();
        let kvps = generate_kvps(common_prefixes, suffixes);
        for (key, value) in &kvps {
            trie.insert(key.clone().into_iter(), *value);
        }
        let mut stored_values = trie.values().copied().collect::<Vec<_>>();
        stored_values.sort_unstable();
        let mut inserted_values = kvps.iter().map(|(_, v)| v).copied().collect::<Vec<_>>();
        inserted_values.sort_unstable();
        stored_values == inserted_values
    }

    #[test]
    fn keys_manual_test() {
        let test_cases = vec![vec![vec![], vec![0u8]]];
        for mut test_case in test_cases {
            let mut trie = Trie::new();
            for key in &test_case {
                trie.insert(key.clone().into_iter(), ());
            }
            let mut keys_inserted = trie
                .keys()
                .iter()
                .map(|k| k.iter().map(|v| **v).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            test_case.sort();
            keys_inserted.sort();
            assert_eq!(test_case, keys_inserted);
        }
    }

    #[quickcheck]
    fn check_no_return_from_insert(common_prefixes: Vec<Vec<u8>>, suffixes: Vec<Vec<u8>>) -> bool {
        let mut trie = Trie::new();
        for (key, value) in generate_kvps(common_prefixes, suffixes) {
            if trie.insert(key.into_iter(), value).is_some() {
                return false;
            }
        }
        true
    }

    #[quickcheck]
    fn check_return_from_insert(common_prefixes: Vec<Vec<u8>>, suffixes: Vec<Vec<u8>>) -> bool {
        let mut trie = Trie::new();
        for (key, value) in generate_kvps(common_prefixes, suffixes) {
            let value = value as usize;
            if trie.insert(key.clone().into_iter(), value).is_some() {
                return false;
            }
            if let Some(previous) = trie.insert(key.clone().into_iter(), value + 1) {
                if previous != value {
                    return false;
                }
            }
            if let Some(previous) = trie.insert(key.clone().into_iter(), value + 1) {
                if previous != value + 1 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    #[test]
    fn check_return_from_insert_manual_cases() {
        let cases = vec![vec![vec![228, 0], vec![0], vec![228]]];
        for case in cases {
            let mut trie = Trie::new();
            for key in case {
                dbg!(&key, &trie);
                assert!(trie.insert(key.into_iter(), ()).is_none());
                dbg!(&trie);
            }
        }
    }

    #[quickcheck]
    fn check_split(common_prefix: Vec<u8>, mut suffix1: Vec<u8>, mut suffix2: Vec<u8>) -> bool {
        // remove any common prefix inside the two suffixes
        loop {
            match (suffix1.first(), suffix2.first()) {
                (Some(a), Some(b)) if a == b => {
                    suffix1.remove(0);
                    suffix2.remove(0)
                }
                _ => {
                    break;
                }
            };
        }
        let mut test_a = common_prefix.clone();
        test_a.extend(&suffix1);
        let mut test_b = common_prefix.clone();
        test_b.extend(&suffix2);
        let mut test_b = test_b.into_iter().peekable();
        let common_prefix_computed =
            Trie::<_, ()>::split_into_common_prefix(&mut test_a, &mut test_b);
        let test_b = test_b.collect::<Vec<_>>();
        dbg!(
            &common_prefix_computed,
            &common_prefix,
            &test_a,
            &suffix1,
            &test_b,
            &suffix2,
        );
        common_prefix_computed == common_prefix && test_a == suffix1 && test_b == suffix2
    }

    fn insert_str<'a>(trie: &mut Trie<char, &'a str>, s: &'a str) {
        trie.insert(s.chars(), s);
    }

    #[test]
    fn print_multi_split() {
        let mut trie = Trie::new();

        dbg!(&trie);
        insert_str(&mut trie, "AB");
        dbg!(&trie);
        insert_str(&mut trie, "CD");
        dbg!(&trie);

        // The resulting `Trie` will contain 4 nodes since nodes with
        // `TrieChildren::Multiple` use only one element of the sequence as key.
    }
}
