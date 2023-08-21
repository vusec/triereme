use std::ops::{Add, BitAnd, BitOr, BitXor, Deref, Mul, Neg, Not, Sub};

use apint::{ApInt, Bit, Width};

use super::{constraint_language::ApIntOrd, Expr};

fn set_bit_apint(v: &mut ApInt, pos: usize, b: Bit) {
    match b {
        Bit::Set => v.set_bit_at(pos).unwrap(),
        Bit::Unset => v.unset_bit_at(pos).unwrap(),
    }
}

fn swap_bit_apint(v: &mut ApInt, pos_a: usize, pos_b: usize) {
    let bit_a = v.get_bit_at(pos_a).unwrap();
    let bit_b = v.get_bit_at(pos_b).unwrap();
    set_bit_apint(v, pos_a, bit_b);
    set_bit_apint(v, pos_b, bit_a);
}

fn bswap_apint(mut v: ApIntOrd) -> ApIntOrd {
    let width: usize = v.width().to_usize();
    assert!(width % 16 == 0);
    for byte_index in 0..(width / 8 / 2) {
        for bit_offset in 0..8 {
            swap_bit_apint(
                &mut v,
                byte_index * 8 + bit_offset,
                (width - byte_index * 8 - 8) + bit_offset,
            );
        }
    }
    v
}

fn concat_apint(l: &ApIntOrd, r: &ApIntOrd) -> ApIntOrd {
    let l_width = l.width().to_usize();
    let r_width = r.width().to_usize();
    let mut res = r
        .deref()
        .clone()
        .into_zero_extend(l_width + r_width)
        .unwrap();
    for i in 0..l_width {
        set_bit_apint(&mut res, r_width + i, l.get_bit_at(i).unwrap());
    }
    res.into()
}

fn extract_apint(v: &mut ApIntOrd, first_bit: usize, last_bit: usize) {
    v.checked_ashr_assign(last_bit).unwrap();
    v.truncate(first_bit - last_bit + 1).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bswap_apinty() {
        let test: ApIntOrd = ApInt::from_u64(0x0123456789ABCDEF).into();
        let swapped = bswap_apint(test);
        assert_eq!(swapped.try_to_u64().unwrap(), 0xEFCDAB8967452301);
        let swapped_back = bswap_apint(swapped);
        assert_eq!(swapped_back.try_to_u64().unwrap(), 0x0123456789ABCDEF);
    }

    #[test]
    fn concat_apinty() {
        let test_l: ApIntOrd = ApInt::from_u32(0x01234567).into();
        let test_r: ApIntOrd = ApInt::from_u32(0x89ABCDEF).into();
        let concatenated = concat_apint(&test_l, &test_r);
        assert_eq!(concatenated.try_to_u64().unwrap(), 0x0123456789ABCDEF);
    }

    #[test]
    fn extract_apinty() {
        let mut test: ApIntOrd = ApInt::from_u64(0x0123456789ABCDEF).into();
        extract_apint(&mut test, 7, 0);
        assert_eq!(test, ApInt::from_u8(0xEF).into());
        let mut test: ApIntOrd = ApInt::from_u64(0x0123456789ABCDEF).into();
        extract_apint(&mut test, 63, 56);
        assert_eq!(test, ApInt::from_u8(0x01).into());
        let mut test: ApIntOrd = ApInt::from_u64(0x0123456789ABCDEF).into();
        extract_apint(&mut test, 63, 0);
        assert_eq!(test, ApInt::from_u64(0x0123456789ABCDEF).into());
        let mut test: ApIntOrd = ApInt::from_u64(0x0123456789ABCDEF).into();
        extract_apint(&mut test, 47, 16);
        assert_eq!(test, ApInt::from_u32(0x456789AB).into());
    }
}

impl Expr {
    pub(super) fn const_eval(self) -> Self {
        match &self {
            Expr::Variable { .. } | Expr::Integer { .. } | Expr::True | Expr::False => self,
            Expr::Neg(e) => match &***e {
                Expr::Integer(i) => Expr::Integer(i.deref().neg().into()),
                Expr::True | Expr::False => todo!("is this possible?"),
                _ => self,
            },
            Expr::Add(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(l.deref().add(r).into()),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported add involving booleans")
                }
                _ => self,
            },
            Expr::Sub(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(l.deref().sub(r).into()),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Mul(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(l.deref().mul(r).into()),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::UDiv(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(
                    if r.is_zero() {
                        ApInt::zero(l.width())
                    } else {
                        l.deref().clone().into_checked_udiv(r).unwrap()
                    }
                    .into(),
                ),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::SDiv(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(
                    if r.is_zero() {
                        ApInt::zero(l.width())
                    } else {
                        l.deref().clone().into_checked_sdiv(r).unwrap()
                    }
                    .into(),
                ),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::URem(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(
                    if r.is_zero() {
                        ApInt::zero(l.width())
                    } else {
                        l.deref().clone().into_checked_urem(r).unwrap()
                    }
                    .into(),
                ),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::SRem(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(
                    if r.is_zero() {
                        ApInt::zero(l.width())
                    } else {
                        l.deref().clone().into_checked_srem(r).unwrap()
                    }
                    .into(),
                ),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Shl(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer({
                    let shift_amount = r.try_to_u64().unwrap() as usize;
                    if shift_amount >= l.width().to_usize() {
                        ApInt::zero(l.width()).into()
                    } else {
                        l.deref()
                            .clone()
                            .into_checked_shl(shift_amount)
                            .unwrap()
                            .into()
                    }
                }),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Lshr(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer({
                    let shift_amount = r.try_to_u64().unwrap() as usize;
                    if shift_amount >= l.width().to_usize() {
                        ApInt::zero(l.width()).into()
                    } else {
                        l.deref()
                            .clone()
                            .into_checked_lshr(shift_amount)
                            .unwrap()
                            .into()
                    }
                }),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Ashr(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer({
                    let shift_amount = r.try_to_u64().unwrap() as usize;
                    if shift_amount >= l.width().to_usize() {
                        ApInt::zero(l.width()).into()
                    } else {
                        l.deref()
                            .clone()
                            .into_checked_ashr(shift_amount)
                            .unwrap()
                            .into()
                    }
                }),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Not(v) => match &***v {
                Expr::Integer(v) => Expr::Integer(v.deref().clone().not().into()),
                Expr::True => Expr::False,
                Expr::False => Expr::True,
                _ => self,
            },
            Expr::Slt(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.checked_slt(r).unwrap().into(),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Sle(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.checked_sle(r).unwrap().into(),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Sgt(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.checked_sgt(r).unwrap().into(),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Sge(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.checked_sge(r).unwrap().into(),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Ult(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.checked_ult(r).unwrap().into(),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Ule(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.checked_ule(r).unwrap().into(),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Ugt(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.checked_ugt(r).unwrap().into(),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Uge(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.checked_uge(r).unwrap().into(),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Equal(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => l.eq(r).into(),
                (Expr::True, Expr::True) | (Expr::False, Expr::False) => Expr::True,
                (Expr::True, Expr::False) | (Expr::False, Expr::True) => Expr::False,
                _ => self,
            },
            Expr::NotEqual(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => (!l.eq(r)).into(),
                (Expr::True, Expr::True) | (Expr::False, Expr::False) => Expr::False,
                (Expr::True, Expr::False) | (Expr::False, Expr::True) => Expr::True,
                _ => self,
            },
            Expr::BoolOr(l, r) => match (&***l, &***r) {
                (Expr::Integer { .. }, _) | (_, Expr::Integer { .. }) => {
                    panic!("unsupported bool or involving integers")
                }
                (Expr::True, _) | (_, Expr::True) => Expr::True,
                (Expr::False, Expr::False) => Expr::False,
                (Expr::False, other) | (other, Expr::False) => other.clone(),
                _ => self,
            },
            Expr::BoolAnd(l, r) => match (&***l, &***r) {
                (Expr::Integer { .. }, _) | (_, Expr::Integer { .. }) => {
                    panic!("unsupported bool and involving integers")
                }
                (Expr::True, Expr::True) => Expr::True,
                (Expr::True, other) | (other, Expr::True) => other.clone(),
                (Expr::False, _) | (_, Expr::False) => Expr::False,
                _ => self,
            },
            Expr::BoolXor(l, r) => match (&***l, &***r) {
                (Expr::Integer { .. }, _) | (_, Expr::Integer { .. }) => {
                    panic!("unsupported bool and involving integers")
                }
                (Expr::True, Expr::False) | (Expr::False, Expr::True) => Expr::True,
                (Expr::True, Expr::True) | (Expr::False, Expr::False) => Expr::False,
                // TODO there are more possible simplifications
                _ => self,
            },
            Expr::And(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(l.deref().bitand(r).into()),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Or(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(l.deref().bitor(r).into()),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Xor(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(l.deref().bitxor(r).into()),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported op involving booleans")
                }
                _ => self,
            },
            Expr::Sext {
                op,
                bits: target_bits,
            } => match &***op {
                Expr::Integer(op) => Expr::Integer(
                    op.deref()
                        .clone()
                        .into_sign_extend(op.width().to_usize() + *target_bits as usize)
                        .unwrap()
                        .into(),
                ),
                Expr::True | Expr::False => panic!("unsupported sext involving booleans"),
                _ => self,
            },
            Expr::Zext {
                op,
                bits: target_bits,
            } => match &***op {
                Expr::Integer(op) => Expr::Integer(
                    op.deref()
                        .clone()
                        .into_zero_extend(op.width().to_usize() + *target_bits as usize)
                        .unwrap()
                        .into(),
                ),
                Expr::True | Expr::False => panic!("unsupported sext involving booleans"),
                _ => self,
            },
            Expr::Trunc {
                op,
                bits: target_bits,
            } => match &***op {
                Expr::Integer(op) => Expr::Integer(
                    op.deref()
                        .clone()
                        .into_truncate(*target_bits as usize)
                        .unwrap()
                        .into(),
                ),
                Expr::True | Expr::False => panic!("unsupported sext involving booleans"),
                _ => self,
            },
            Expr::Bswap(op) => match &***op {
                Expr::Integer(int) => Expr::Integer(bswap_apint(int.clone())),
                Expr::True | Expr::False => todo!("is this possible?"),
                _ => self,
            },
            Expr::BoolToBits { op, bits } => match &***op {
                Expr::Integer { .. } => todo!("is this possible?"),
                Expr::True => Expr::Integer(ApInt::one((*bits as usize).into()).into()),
                Expr::False => Expr::Integer(ApInt::zero((*bits as usize).into()).into()),
                _ => self,
            },
            Expr::Concat(l, r) => match (&***l, &***r) {
                (Expr::Integer(l), Expr::Integer(r)) => Expr::Integer(concat_apint(l, r)),
                (Expr::True | Expr::False, _) | (_, Expr::True | Expr::False) => {
                    panic!("unsupported concat involving booleans")
                }
                _ => self,
            },
            Expr::Extract {
                op,
                first_bit,
                last_bit,
            } => match &***op {
                Expr::Integer(v) => {
                    let mut v = v.clone();
                    extract_apint(&mut v, *first_bit, *last_bit);
                    Expr::Integer(v)
                }
                Expr::True | Expr::False => todo!("is this possible?"),
                _ => self,
            },
        }
    }
}
