extern crate bindgen;

use std::env;
use std::path::PathBuf;

use bindgen::{
    callbacks::{
        DeriveTrait, EnumVariantCustomBehavior, EnumVariantValue, ImplementsTrait, IntKind,
        MacroParsingBehavior, ParseCallbacks,
    },
    CargoCallbacks,
};
use symcc_libafl::clone_symcc;

#[derive(Debug)]
struct MyCallbacks<W: ParseCallbacks> {
    wrapped: W,
}

impl<W: ParseCallbacks> MyCallbacks<W> {
    fn new(wrapped: W) -> Self {
        Self { wrapped }
    }
}

impl<W: ParseCallbacks> ParseCallbacks for MyCallbacks<W> {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        self.wrapped.will_parse_macro(name)
    }

    fn int_macro(&self, name: &str, value: i64) -> Option<IntKind> {
        self.wrapped.int_macro(name, value)
    }

    fn str_macro(&self, name: &str, value: &[u8]) {
        self.wrapped.str_macro(name, value);
    }

    fn func_macro(&self, name: &str, value: &[&[u8]]) {
        self.wrapped.func_macro(name, value);
    }

    fn enum_variant_behavior(
        &self,
        enum_name: Option<&str>,
        original_variant_name: &str,
        variant_value: EnumVariantValue,
    ) -> Option<EnumVariantCustomBehavior> {
        self.wrapped
            .enum_variant_behavior(enum_name, original_variant_name, variant_value)
    }

    fn enum_variant_name(
        &self,
        enum_name: Option<&str>,
        original_variant_name: &str,
        variant_value: EnumVariantValue,
    ) -> Option<String> {
        self.wrapped
            .enum_variant_name(enum_name, original_variant_name, variant_value)
    }

    fn item_name(&self, original_item_name: &str) -> Option<String> {
        if original_item_name == "SymExpr" {
            Some("SymExprRef".to_string())
        } else {
            self.wrapped.item_name(original_item_name)
        }
    }

    fn include_file(&self, filename: &str) {
        self.wrapped.include_file(filename);
    }

    fn blocklisted_type_implements_trait(
        &self,
        name: &str,
        derive_trait: DeriveTrait,
    ) -> Option<ImplementsTrait> {
        self.wrapped
            .blocklisted_type_implements_trait(name, derive_trait)
    }
}

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let symcc_path = out_path.join("symcc");
    if !symcc_path.exists() {
        clone_symcc(&symcc_path);
    }
    let bindings = bindgen::Builder::default()
        .clang_arg(format!(
            "-I{}",
            symcc_path.join("runtime").to_str().unwrap()
        ))
        .header("wrapper.h")
        .dynamic_library_name("SymRuntime")
        .allowlist_function("_sym.*")
        .size_t_is_usize(true)
        .blocklist_type("SymExpr")
        .parse_callbacks(Box::new(MyCallbacks::new(CargoCallbacks)))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
