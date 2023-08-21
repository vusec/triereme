use libafl::observers::concolic::{serialization_format::MessageFileReader, SymExpr};
use rustc_hash::FxHashMap;
use symcc_runtime::Runtime;

use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use crate::runtimes::high_level::HighLevelRuntime;

pub fn reader_from_file(path: &Path) -> io::Result<MessageFileReader<BufReader<File>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut buf = 0u64.to_be_bytes();
    reader.read_exact(&mut buf)?;
    Ok(MessageFileReader::from_reader(reader))
}

pub fn replay_trace_hl<RT, R>(rt: &mut RT, reader: &mut MessageFileReader<R>)
where
    RT: HighLevelRuntime,
    R: Read,
{
    let mut translation = FxHashMap::default();
    while let Some(msg) = reader.next_message() {
        if let Ok((id, msg)) = msg {
            match msg {
                SymExpr::InputByte { offset } => {
                    if let Some(e) = rt.get_input_byte(offset) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Integer { value, bits } => {
                    if let Some(e) = rt.build_integer(value, bits) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Integer128 { high, low } => {
                    if let Some(e) = rt.build_integer128(high, low) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::NullPointer => {
                    if let Some(e) = rt.build_null_pointer() {
                        translation.insert(id, e);
                    }
                }
                SymExpr::True => {
                    if let Some(e) = rt.build_true() {
                        translation.insert(id, e);
                    }
                }
                SymExpr::False => {
                    if let Some(e) = rt.build_false() {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Bool { value } => {
                    if let Some(e) = rt.build_bool(value) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Neg { op } => {
                    if let Some(e) = rt.build_neg(translation.get(&op)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Add { a, b } => {
                    if let Some(e) = rt.build_add(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Sub { a, b } => {
                    if let Some(e) = rt.build_sub(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Mul { a, b } => {
                    if let Some(e) = rt.build_mul(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::UnsignedDiv { a, b } => {
                    if let Some(e) = rt.build_unsigned_div(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::SignedDiv { a, b } => {
                    if let Some(e) = rt.build_signed_div(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::UnsignedRem { a, b } => {
                    if let Some(e) = rt.build_unsigned_rem(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::SignedRem { a, b } => {
                    if let Some(e) = rt.build_signed_rem(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::ShiftLeft { a, b } => {
                    if let Some(e) = rt.build_shift_left(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::LogicalShiftRight { a, b } => {
                    if let Some(e) =
                        rt.build_logical_shift_right(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::ArithmeticShiftRight { a, b } => {
                    if let Some(e) =
                        rt.build_arithmetic_shift_right(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::SignedLessThan { a, b } => {
                    if let Some(e) =
                        rt.build_signed_less_than(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::SignedLessEqual { a, b } => {
                    if let Some(e) =
                        rt.build_signed_less_equal(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::SignedGreaterThan { a, b } => {
                    if let Some(e) =
                        rt.build_signed_greater_than(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::SignedGreaterEqual { a, b } => {
                    if let Some(e) =
                        rt.build_signed_greater_equal(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::UnsignedLessThan { a, b } => {
                    if let Some(e) =
                        rt.build_unsigned_less_than(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::UnsignedLessEqual { a, b } => {
                    if let Some(e) =
                        rt.build_unsigned_less_equal(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::UnsignedGreaterThan { a, b } => {
                    if let Some(e) =
                        rt.build_unsigned_greater_than(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::UnsignedGreaterEqual { a, b } => {
                    if let Some(e) =
                        rt.build_unsigned_greater_equal(translation.get(&a), translation.get(&b))
                    {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Not { .. } => todo!(),
                SymExpr::Equal { a, b } => {
                    if let Some(e) = rt.build_equal(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::NotEqual { a, b } => {
                    if let Some(e) = rt.build_not_equal(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::BoolAnd { a, b } => {
                    if let Some(e) = rt.build_bool_and(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::BoolOr { a, b } => {
                    if let Some(e) = rt.build_bool_or(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::BoolXor { a, b } => {
                    if let Some(e) = rt.build_bool_xor(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::And { a, b } => {
                    if let Some(e) = rt.build_and(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Or { a, b } => {
                    if let Some(e) = rt.build_or(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Xor { a, b } => {
                    if let Some(e) = rt.build_xor(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Float { .. }
                | SymExpr::FloatOrdered { .. }
                | SymExpr::FloatOrderedGreaterThan { .. }
                | SymExpr::FloatOrderedGreaterEqual { .. }
                | SymExpr::FloatOrderedLessThan { .. }
                | SymExpr::FloatOrderedLessEqual { .. }
                | SymExpr::FloatOrderedEqual { .. }
                | SymExpr::FloatOrderedNotEqual { .. }
                | SymExpr::FloatUnordered { .. }
                | SymExpr::FloatUnorderedGreaterThan { .. }
                | SymExpr::FloatUnorderedGreaterEqual { .. }
                | SymExpr::FloatUnorderedLessThan { .. }
                | SymExpr::FloatUnorderedLessEqual { .. }
                | SymExpr::FloatUnorderedEqual { .. }
                | SymExpr::FloatUnorderedNotEqual { .. }
                | SymExpr::FloatAdd { .. }
                | SymExpr::FloatSub { .. }
                | SymExpr::FloatMul { .. }
                | SymExpr::FloatDiv { .. }
                | SymExpr::FloatRem { .. }
                | SymExpr::FloatAbs { .. }
                | SymExpr::FloatToFloat { .. }
                | SymExpr::BitsToFloat { .. }
                | SymExpr::FloatToBits { .. }
                | SymExpr::FloatToSignedInteger { .. }
                | SymExpr::FloatToUnsignedInteger { .. }
                | SymExpr::IntToFloat { .. } => {
                    unreachable!();
                }
                SymExpr::Sext { op, bits } => {
                    if let Some(e) = rt.build_sext(translation.get(&op), bits) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Zext { op, bits } => {
                    if let Some(e) = rt.build_zext(translation.get(&op), bits) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Trunc { op, bits } => {
                    if let Some(e) = rt.build_trunc(translation.get(&op), bits) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::BoolToBits { op, bits } => {
                    if let Some(e) = rt.build_bool_to_bits(translation.get(&op), bits) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Concat { a, b } => {
                    if let Some(e) = rt.concat_helper(translation.get(&a), translation.get(&b)) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Extract {
                    op,
                    first_bit,
                    last_bit,
                } => {
                    if let Some(e) = rt.extract_helper(translation.get(&op), first_bit, last_bit) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Insert { .. } => todo!(),
                SymExpr::PathConstraint {
                    constraint,
                    taken,
                    site_id,
                } => {
                    rt.push_path_constraint(
                        translation.get(&constraint),
                        if taken { 1 } else { 0 },
                        site_id,
                    );
                }
                SymExpr::ExpressionsUnreachable { exprs } => {
                    for drained in exprs.into_iter().map(|e| translation.remove(&e)).flatten() {
                        rt.expression_unreachable(Some(drained));
                    }
                }
            }
        } else {
            break;
        }
    }
}

pub fn replay_trace<RT, R>(rt: &mut RT, mut reader: MessageFileReader<R>)
where
    RT: Runtime,
    R: Read,
{
    let mut translation = FxHashMap::default();
    while let Some(msg) = reader.next_message() {
        if let Ok((id, msg)) = msg {
            match msg {
                SymExpr::InputByte { offset } => {
                    if let Some(e) = rt.get_input_byte(offset) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Integer { value, bits } => {
                    if let Some(e) = rt.build_integer(value, bits) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Integer128 { high, low } => {
                    if let Some(e) = rt.build_integer128(high, low) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::NullPointer => {
                    if let Some(e) = rt.build_null_pointer() {
                        translation.insert(id, e);
                    }
                }
                SymExpr::True => {
                    if let Some(e) = rt.build_true() {
                        translation.insert(id, e);
                    }
                }
                SymExpr::False => {
                    if let Some(e) = rt.build_false() {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Bool { value } => {
                    if let Some(e) = rt.build_bool(value) {
                        translation.insert(id, e);
                    }
                }
                SymExpr::Neg { op } => {
                    if let Some(op) = translation.get(&op) {
                        if let Some(e) = rt.build_neg(*op) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Add { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_add(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Sub { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_sub(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Mul { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_mul(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::UnsignedDiv { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_unsigned_div(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::SignedDiv { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_signed_div(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::UnsignedRem { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_unsigned_rem(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::SignedRem { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_signed_rem(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::ShiftLeft { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_shift_left(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::LogicalShiftRight { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_logical_shift_right(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::ArithmeticShiftRight { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_arithmetic_shift_right(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::SignedLessThan { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_signed_less_than(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::SignedLessEqual { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_signed_less_equal(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::SignedGreaterThan { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_signed_greater_than(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::SignedGreaterEqual { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_signed_greater_equal(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::UnsignedLessThan { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_unsigned_less_than(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::UnsignedLessEqual { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_unsigned_less_equal(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::UnsignedGreaterThan { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_unsigned_greater_than(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::UnsignedGreaterEqual { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_unsigned_greater_equal(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Not { .. } => todo!(),
                SymExpr::Equal { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_equal(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::NotEqual { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_not_equal(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::BoolAnd { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_bool_and(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::BoolOr { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_bool_or(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::BoolXor { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_bool_xor(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::And { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_and(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Or { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_or(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Xor { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.build_xor(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Float { .. }
                | SymExpr::FloatOrdered { .. }
                | SymExpr::FloatOrderedGreaterThan { .. }
                | SymExpr::FloatOrderedGreaterEqual { .. }
                | SymExpr::FloatOrderedLessThan { .. }
                | SymExpr::FloatOrderedLessEqual { .. }
                | SymExpr::FloatOrderedEqual { .. }
                | SymExpr::FloatOrderedNotEqual { .. }
                | SymExpr::FloatUnordered { .. }
                | SymExpr::FloatUnorderedGreaterThan { .. }
                | SymExpr::FloatUnorderedGreaterEqual { .. }
                | SymExpr::FloatUnorderedLessThan { .. }
                | SymExpr::FloatUnorderedLessEqual { .. }
                | SymExpr::FloatUnorderedEqual { .. }
                | SymExpr::FloatUnorderedNotEqual { .. }
                | SymExpr::FloatAdd { .. }
                | SymExpr::FloatSub { .. }
                | SymExpr::FloatMul { .. }
                | SymExpr::FloatDiv { .. }
                | SymExpr::FloatRem { .. }
                | SymExpr::FloatAbs { .. }
                | SymExpr::FloatToFloat { .. }
                | SymExpr::BitsToFloat { .. }
                | SymExpr::FloatToBits { .. }
                | SymExpr::FloatToSignedInteger { .. }
                | SymExpr::FloatToUnsignedInteger { .. }
                | SymExpr::IntToFloat { .. } => {
                    unreachable!("unexpected symexpr {:?}", msg);
                }
                SymExpr::Sext { op, bits } => {
                    if let Some(op) = translation.get(&op) {
                        if let Some(e) = rt.build_sext(*op, bits) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Zext { op, bits } => {
                    if let Some(op) = translation.get(&op) {
                        if let Some(e) = rt.build_zext(*op, bits) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Trunc { op, bits } => {
                    if let Some(op) = translation.get(&op) {
                        if let Some(e) = rt.build_trunc(*op, bits) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::BoolToBits { op, bits } => {
                    if let Some(op) = translation.get(&op) {
                        if let Some(e) = rt.build_bool_to_bits(*op, bits) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Concat { a, b } => {
                    if let (Some(a), Some(b)) = (translation.get(&a), translation.get(&b)) {
                        if let Some(e) = rt.concat_helper(*a, *b) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Extract {
                    op,
                    first_bit,
                    last_bit,
                } => {
                    if let Some(op) = translation.get(&op) {
                        if let Some(e) = rt.extract_helper(*op, first_bit, last_bit) {
                            translation.insert(id, e);
                        }
                    }
                }
                SymExpr::Insert { .. } => {
                    todo!("insert not implemented yet: {:?}", msg);
                }
                SymExpr::PathConstraint {
                    constraint,
                    taken,
                    site_id,
                } => {
                    if let Some(constraint) = translation.get(&constraint) {
                        rt.push_path_constraint(*constraint, taken, site_id);
                    }
                }
                SymExpr::ExpressionsUnreachable { exprs } => {
                    rt.expression_unreachable(&exprs);
                    exprs
                        .into_iter()
                        .map(|e| translation.remove(&e))
                        .flatten()
                        .count();
                }
            }
        } else {
            break;
        }
    }
}
