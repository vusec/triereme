use std::{path::Path, sync::Mutex, time::Instant};

use libafl::{
    bolts::ownedref::OwnedSlice,
    inputs::{BytesInput, HasTargetBytes, Input},
};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

static START_TIME: Lazy<Mutex<Instant>> = Lazy::new(|| Mutex::new(Instant::now()));

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AflCompatInput(BytesInput);

impl Input for AflCompatInput {
    fn generate_name(&self, idx: usize) -> String {
        let seconds = START_TIME.lock().unwrap().elapsed().as_secs();
        format!("id:{idx:06},time:{seconds}")
    }

    fn from_file<P>(path: P) -> Result<Self, libafl::Error>
    where
        P: AsRef<Path>,
    {
        let bytes_input = BytesInput::from_file(path)?;
        Ok(Self(bytes_input))
    }

    fn to_file<P>(&self, path: P) -> Result<(), libafl::Error>
    where
        P: AsRef<Path>,
    {
        self.0.to_file(path)
    }
}

impl AflCompatInput {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(BytesInput::new(bytes))
    }
}

impl HasTargetBytes for AflCompatInput {
    fn target_bytes(&self) -> OwnedSlice<u8> {
        self.0.target_bytes()
    }
}
