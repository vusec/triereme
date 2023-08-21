use std::os::raw::c_int;

pub trait HighLevelRuntime {
    type Expr;
    fn build_integer(&mut self, value: u64, bits: u8) -> Option<Self::Expr>;
    fn build_integer128(&mut self, high: u64, low: u64) -> Option<Self::Expr>;
    fn build_float(&mut self, value: f64, is_double: c_int) -> Option<Self::Expr>;
    fn build_null_pointer(&mut self) -> Option<Self::Expr>;
    fn build_true(&mut self) -> Option<Self::Expr>;
    fn build_false(&mut self) -> Option<Self::Expr>;
    fn build_bool(&mut self, value: bool) -> Option<Self::Expr>;
    fn build_neg(&mut self, expr: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_add(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_sub(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_mul(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_unsigned_div(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_signed_div(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_unsigned_rem(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_signed_rem(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_shift_left(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_logical_shift_right(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_arithmetic_shift_right(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_fp_add(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_fp_sub(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_fp_mul(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_fp_div(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_fp_rem(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_fp_abs(&mut self, a: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_not(&mut self, expr: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_signed_less_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_signed_less_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_signed_greater_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_signed_greater_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_unsigned_less_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_unsigned_less_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_unsigned_greater_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_unsigned_greater_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_equal(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>)
        -> Option<Self::Expr>;
    fn build_not_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_bool_and(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_and(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_bool_or(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_or(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_bool_xor(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_xor(&mut self, a: Option<&Self::Expr>, b: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_float_ordered_greater_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_ordered_greater_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_ordered_less_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_ordered_less_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_ordered_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_ordered_not_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_ordered(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_unordered(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_unordered_greater_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_unordered_greater_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_unordered_less_than(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_unordered_less_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_unordered_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_float_unordered_not_equal(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn build_sext(&mut self, expr: Option<&Self::Expr>, bits: u8) -> Option<Self::Expr>;
    fn build_zext(&mut self, expr: Option<&Self::Expr>, bits: u8) -> Option<Self::Expr>;
    fn build_trunc(&mut self, expr: Option<&Self::Expr>, bits: u8) -> Option<Self::Expr>;
    fn build_bswap(&mut self, expr: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_int_to_float(
        &mut self,
        value: Option<&Self::Expr>,
        is_double: c_int,
        is_signed: c_int,
    ) -> Option<Self::Expr>;
    fn build_float_to_float(
        &mut self,
        expr: Option<&Self::Expr>,
        to_double: c_int,
    ) -> Option<Self::Expr>;
    fn build_bits_to_float(
        &mut self,
        expr: Option<&Self::Expr>,
        to_double: c_int,
    ) -> Option<Self::Expr>;
    fn build_float_to_bits(&mut self, expr: Option<&Self::Expr>) -> Option<Self::Expr>;
    fn build_float_to_signed_integer(
        &mut self,
        expr: Option<&Self::Expr>,
        bits: u8,
    ) -> Option<Self::Expr>;
    fn build_float_to_unsigned_integer(
        &mut self,
        expr: Option<&Self::Expr>,
        bits: u8,
    ) -> Option<Self::Expr>;
    fn build_bool_to_bits(&mut self, expr: Option<&Self::Expr>, bits: u8) -> Option<Self::Expr>;
    fn concat_helper(
        &mut self,
        a: Option<&Self::Expr>,
        b: Option<&Self::Expr>,
    ) -> Option<Self::Expr>;
    fn extract_helper(
        &mut self,
        expr: Option<&Self::Expr>,
        first_bit: usize,
        last_bit: usize,
    ) -> Option<Self::Expr>;
    fn push_path_constraint(
        &mut self,
        constraint: Option<&Self::Expr>,
        taken: c_int,
        site_id: usize,
    );
    fn get_input_byte(&mut self, offset: usize) -> Option<Self::Expr>;
    fn expression_unreachable(&mut self, expr: Option<Self::Expr>);
}
