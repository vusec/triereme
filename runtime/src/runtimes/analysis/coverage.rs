use std::{
    convert::TryInto,
    default::Default,
    hash::{BuildHasher, BuildHasherDefault, Hash, Hasher},
    io::{self, Read, Write},
    marker::PhantomData,
};

use ahash::AHasher;
use hashbrown::HashSet;

const MAP_SIZE: usize = 65536;

const CHAR_BIT: usize = u8::BITS as usize;

pub struct BranchCoverageFilter<
    THasher: Hasher = AHasher,
    THashBuilder: BuildHasher = BuildHasherDefault<THasher>,
> {
    trace_map: Vec<u8>,
    virgin_map: Vec<u8>,
    context_map: Vec<u8>,
    visited: HashSet<usize>,
    prev_loc: usize,
    hasher_builder: THashBuilder,
    hasher_phantom: PhantomData<THasher>,
}

impl Default for BranchCoverageFilter<AHasher, BuildHasherDefault<AHasher>> {
    fn default() -> Self {
        Self {
            trace_map: vec![0; MAP_SIZE],
            virgin_map: vec![0; MAP_SIZE],
            context_map: vec![0; MAP_SIZE],
            visited: HashSet::new(),
            prev_loc: 0,
            hasher_builder: BuildHasherDefault::default(),
            hasher_phantom: PhantomData,
        }
    }
}

impl<THasher: Hasher, THashBuilder: BuildHasher> BranchCoverageFilter<THasher, THashBuilder> {
    fn hash_pc(&self, location: usize, taken: bool) -> u64 {
        let mut hasher = self.hasher_builder.build_hasher();
        location.hash(&mut hasher);
        taken.hash(&mut hasher);
        hasher.finish()
    }

    pub fn import<R: Read>(&mut self, mut reader: R) -> io::Result<()> {
        reader.read_exact(&mut self.trace_map)?;
        reader.read_exact(&mut self.context_map)?;
        Ok(())
    }

    pub fn export<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(&self.trace_map)?;
        writer.write_all(&self.context_map)?;
        Ok(())
    }

    fn get_index(&self, h: usize) -> usize {
        ((self.prev_loc >> 1) ^ h) % MAP_SIZE
    }

    fn is_interesting_context(&mut self, h: usize, bits: u8) -> bool {
        if !(bits == 0 || bits.is_power_of_two()) {
            return false;
        }

        let mut is_interesting = false;
        for prev_h in &self.visited {
            let mut hasher = self.hasher_builder.build_hasher();
            prev_h.hash(&mut hasher);
            h.hash(&mut hasher);

            let hash = hasher.finish() as usize % (MAP_SIZE * CHAR_BIT);
            let idx = hash / CHAR_BIT;
            let mask = 1 << (hash % CHAR_BIT);

            if (self.context_map[idx] & mask) == 0 {
                self.context_map[idx] |= mask;
                is_interesting = true;
            }
        }

        if bits == 0 {
            self.visited.insert(h);
        }

        is_interesting
    }

    pub fn is_interesting_branch(&mut self, location: usize, taken: bool) -> bool {
        let h = self.hash_pc(location, taken).try_into().unwrap();
        let idx = self.get_index(h);
        let is_new_context = self.is_interesting_context(h, self.virgin_map[idx]);

        let v = self.virgin_map.get_mut(idx).unwrap();
        *v = v.saturating_add(1);

        let res = if (self.virgin_map[idx] | self.trace_map[idx]) == self.trace_map[idx] {
            is_new_context
        } else {
            let inv_h = self.hash_pc(location, !taken).try_into().unwrap();
            let inv_idx = self.get_index(inv_h);

            *self.trace_map.get_mut(idx).unwrap() |= self.virgin_map[idx];

            *self.trace_map.get_mut(inv_idx).unwrap() |= self.virgin_map[inv_idx].saturating_add(1);

            true
        };

        self.prev_loc = h;

        res
    }

    pub fn reset(&mut self) {
        self.virgin_map.iter_mut().for_each(|b| *b = 0);
        self.visited.clear();
        self.prev_loc = 0;
    }
}
