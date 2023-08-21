#![allow(non_snake_case)]

use libafl::{
    bolts::shmem::{ShMemProvider, StdShMemProvider},
    observers::concolic::serialization_format::DEFAULT_ENV_NAME,
};
use symcc_runtime::{
    export_runtime,
    filter::{CallStackCoverage, NoFloat},
    invoke_macro_with_rust_runtime_exports,
    tracing::{self, StdShMemMessageFileWriter, TracingRuntime},
    Runtime,
};

struct OptionalRuntime<RT> {
    inner: Option<RT>,
}

impl<RT> OptionalRuntime<RT>
where
    RT: Runtime,
{
    fn new(rt: Option<RT>) -> Self {
        Self { inner: rt }
    }
}

macro_rules! rust_runtime_function_declaration {
    (pub fn expression_unreachable(expressions: *mut RSymExpr, num_elements: usize), $c_name:ident;) => {
        #[allow(clippy::default_trait_access)]
        fn expression_unreachable(&mut self, exprs: &[RSymExpr]) {
            if let Some(inner) = &mut self.inner {
                inner.expression_unreachable(exprs);
            }
        }
    };

    (pub fn $name:ident($( $arg:ident : $type:ty ),*$(,)?) -> $ret:ty,  $c_name:ident;) => {
        fn $name(&mut self, $( $arg : $type),*) -> Option<$ret> {
            if let Some(inner) = &mut self.inner {
                inner.$name($($arg,)*)
            } else {
                None
            }
        }
    };

    (pub fn $name:ident($( $arg:ident : $type:ty ),*$(,)?), $c_name:ident;) => {
        fn $name(&mut self, $( $arg : $type),*) {
            if let Some(inner) = &mut self.inner {
                inner.$name($($arg,)*);
            }
        }
    };
}

impl<RT> Runtime for OptionalRuntime<RT>
where
    RT: Runtime,
{
    invoke_macro_with_rust_runtime_exports!(rust_runtime_function_declaration;);
}

fn create_tracer() -> Option<tracing::TracingRuntime> {
    let mut provider = StdShMemProvider::new().expect("unable to initialize StdShMemProvider");
    let shmem = provider.existing_from_env(DEFAULT_ENV_NAME.as_ref()).ok()?;
    let writer = StdShMemMessageFileWriter::from_shmem(shmem).ok()?;
    Some(TracingRuntime::new(writer))
}

export_runtime!(
    NoFloat => NoFloat;
    CallStackCoverage::default() => CallStackCoverage;
    OptionalRuntime::new(create_tracer())
        => OptionalRuntime<tracing::TracingRuntime>
);
