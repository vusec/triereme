#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]

pub type SymExprRef = usize;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
