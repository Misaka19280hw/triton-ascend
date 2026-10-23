# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Portable conversions for Triton's software-only FP8 E4M3B15 type."""

from triton.language import core


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    """Implement the legacy ``fp8e4b15`` conversion without target assembly.

    Match ``third_party/nvidia/language/cuda/utils.py``: decoding embeds the
    sign, four exponent bits and three fraction bits into FP16 (bias 15).
    Every byte has a finite value, including 0x7f/0xff, which decode to
    +/-1.875. Encoding instead saturates at +/-1.75, as the upstream software
    converter does, and preserves the input sign, including signed zero/NaN.

    The legacy encoder rounds halfway cases away from zero, regardless of
    ``fp_downcast_rounding``; FP32 inputs first truncate to FP16. This is not
    an IEEE round-to-nearest-even encoder, despite the generic cast API's
    default rounding argument. Keeping this behavior matches existing
    Triton kernels using this software format.
    """
    src_scalar = arg.type.scalar
    dst_scalar = dst_ty.scalar

    if src_scalar.is_fp8e4b15():
        bits = _semantic.cast(_semantic.bitcast(arg, core.uint8), core.uint16)
        magnitude = _semantic.shl(_semantic.and_(bits, 0x7f), 7)
        sign = _semantic.shl(_semantic.and_(bits, 0x80), 8)
        half = _semantic.bitcast(_semantic.or_(magnitude, sign), core.float16)
        # Keep the requested scalar element type and the original block shape.
        return _semantic.cast(half, dst_ty)

    if dst_scalar.is_fp8e4b15() and src_scalar.is_bf16():
        # Widen the representation without canonicalizing a signed NaN.
        bits = _semantic.cast(_semantic.bitcast(arg, core.uint16), core.uint32)
        arg = _semantic.bitcast(_semantic.shl(bits, 16), core.float32)
        src_scalar = arg.type.scalar

    if dst_scalar.is_fp8e4b15() and src_scalar.is_int():
        # tl.load(..., other=0) casts its integer padding value to the loaded
        # dtype. Integers 0 and +/-1 are exact, and every
        # other integer is outside the encoder's +/-1.75 saturation range.
        arg = _semantic.cast(arg, core.float32)
        src_scalar = arg.type.scalar

    if not dst_scalar.is_fp8e4b15() or not (src_scalar.is_fp16() or src_scalar.is_fp32()):
        raise ValueError("Ascend fp8e4b15 encoding supports integer, float16, bfloat16 and float32 inputs; "
                         f"got {src_scalar} -> {dst_scalar}")

    if src_scalar.is_fp16():
        bits = _semantic.bitcast(arg, core.uint16)
        sign = _semantic.lshr(_semantic.and_(bits, 0x8000), 8)
        magnitude = _semantic.and_(bits, 0x7fff)
        # Nonnegative FP16 bit patterns are ordered as unsigned integers.
        # Infinity and every NaN payload saturate, like upstream min.f16x2
        # without the .NaN modifier.
        magnitude = _semantic.minimum(magnitude, 0x3f00, core.PropagateNan.NONE)
        magnitude = _semantic.lshr(_semantic.add(magnitude, 0x40, sanitize_overflow=False), 7)
    else:
        # Every FP8 rounding midpoint is exactly representable in FP16.
        # Direct rounding is therefore equivalent to the upstream FP32 ->
        # FP16 RTZ -> FP8 ties-away sequence. Integer operations also avoid
        # a tensor-only HFusion cast when the argument is a scalar.
        bits = _semantic.bitcast(arg, core.uint32)
        sign = _semantic.and_(_semantic.lshr(bits, 24), 0x80)
        magnitude = _semantic.and_(bits, 0x7fffffff)
        magnitude = _semantic.minimum(magnitude, 0x3fe00000, core.PropagateNan.NONE)
        exponent = _semantic.lshr(magnitude, 23)

        rounded = _semantic.lshr(_semantic.add(magnitude, 0x80000, sanitize_overflow=False), 20)
        normal = _semantic.sub(rounded, (127 - 15) << 3, sanitize_overflow=False)

        # Subnormals are multiples of 2**-17. FP32 exponents 109..112
        # contain values that round into FP8 subnormals (or the first normal).
        significand = _semantic.or_(_semantic.and_(magnitude, 0x7fffff), 0x800000)
        shift = _semantic.sub(133, exponent, sanitize_overflow=False)
        # Keep every shift in range even on lanes that select another branch.
        shift = _semantic.minimum(shift, 24, core.PropagateNan.NONE)
        halfway = _semantic.shl(1, _semantic.sub(shift, 1, sanitize_overflow=False))
        subnormal = _semantic.lshr(_semantic.add(significand, halfway, sanitize_overflow=False), shift)
        subnormal = _semantic.where(_semantic.greater_equal(exponent, 109), subnormal, 0)
        magnitude = _semantic.where(_semantic.greater_equal(exponent, 113), normal, subnormal)

    encoded = _semantic.cast(_semantic.or_(magnitude, sign), core.uint8)
    return _semantic.bitcast(encoded, dst_ty)
