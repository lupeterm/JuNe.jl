# Float8 E4M3 (OCP FP8: 1 sign bit, 4 exponent bits with bias 7, 3 mantissa bits, no infinities,
# NaN = S.1111.111, largest finite 448) as a storage type for the mixed-precision benchmark. The H200
# has no fp8 arithmetic outside the tensor cores: every operation converts to Float16 and back. On the
# GPU the conversions are the hardware's cvt.rn.f16x2.e4m3x2 / cvt.rn.satfinite.e4m3x2.f16x2; the CPU
# methods compute the same results (round to nearest even, saturation to +-448, NaN stays NaN), which
# the device overrides require.
import CUDA.CUDACore: @device_override

primitive type Float8E4M3 <: AbstractFloat 8 end

const F8_NAN = 0x7f
const F8_MAX = 0x7e  # 448

function f8_to_f16(b::UInt8)
    sign = UInt16(b & 0x80) << 8
    e, m = (b >> 3) & 0x0f, b & 0x07
    e == 0x0f && m == 0x07 && return reinterpret(Float16, sign | 0x7e00)  # NaN
    if e == 0  # subnormal: m * 2^-9, exact in Float16
        v = Float16(m) * Float16(0.001953125)
        return reinterpret(Float16, sign | reinterpret(UInt16, v))
    end
    return reinterpret(Float16, sign | (UInt16(e + 0x08) << 10) | (UInt16(m) << 7))
end

function f16_to_f8(x::Float16)
    h = reinterpret(UInt16, x)
    sign = UInt8((h >> 8) & 0x80)
    a = h & 0x7fff
    a > 0x7c00 && return sign | F8_NAN                        # NaN
    a >= 0x5f40 && return sign | F8_MAX                       # >= 464 (incl. Inf): saturated to 448
    if a < 0x2400                                             # below 2^-6, the smallest normal
        # multiples of 2^-9, rounded to nearest even; 8 * 2^-9 is the smallest normal, code 0x08
        v = Float32(reinterpret(Float16, a)) * 512.0f0
        q = unsafe_trunc(UInt8, round(v))                     # round: nearest, ties to even
        return sign | q
    end
    e = UInt8((a >> 10) - 0x08)                               # 1 .. 15
    m10 = a & 0x03ff
    m, rest = UInt8(m10 >> 7), m10 & 0x7f
    if rest > 0x40 || (rest == 0x40 && isodd(m))
        m += 0x01
        if m == 0x08
            m, e = 0x00, e + 0x01
        end
    end
    (e > 0x0f || (e == 0x0f && m == 0x07)) && return sign | F8_MAX
    return sign | (e << 3) | m
end

# the hardware conversions (sm_89 and newer), one value in the low half of the packed registers
@inline f8_to_f16_hw(b::UInt8) = reinterpret(Float16, Base.llvmcall("""
    %r = call i32 asm "cvt.rn.f16x2.e4m3x2 \$0, \$1;", "=r,h"(i16 %0)
    %h = trunc i32 %r to i16
    ret i16 %h""", UInt16, Tuple{UInt16}, UInt16(b)))
@inline f16_to_f8_hw(x::Float16) = Base.llvmcall("""
    %r = call i16 asm "cvt.rn.satfinite.e4m3x2.f16x2 \$0, \$1;", "=h,r"(i32 %0)
    %b = trunc i16 %r to i8
    ret i8 %b""", UInt8, Tuple{UInt32}, UInt32(reinterpret(UInt16, x)))

Base.Float16(x::Float8E4M3) = f8_to_f16(reinterpret(UInt8, x))
Float8E4M3(x::Float16) = reinterpret(Float8E4M3, f16_to_f8(x))
@device_override Base.Float16(x::Float8E4M3) = f8_to_f16_hw(reinterpret(UInt8, x))
@device_override Float8E4M3(x::Float16) = reinterpret(Float8E4M3, f16_to_f8_hw(x))

# through Float16 (from Float32/Float64 rounded twice)
Float8E4M3(x::Union{Float32,Float64,Core.BFloat16}) = Float8E4M3(Float16(x))
Float8E4M3(x::Integer) = Float8E4M3(Float16(x))
Float8E4M3(x::Float8E4M3) = x
Base.Float32(x::Float8E4M3) = Float32(Float16(x))
Base.Float64(x::Float8E4M3) = Float64(Float16(x))
Base.convert(::Type{Float8E4M3}, x::Real) = Float8E4M3(x)
Base.zero(::Type{Float8E4M3}) = reinterpret(Float8E4M3, 0x00)
Base.one(::Type{Float8E4M3}) = reinterpret(Float8E4M3, 0x38)
Base.:-(x::Float8E4M3) = reinterpret(Float8E4M3, reinterpret(UInt8, x) ⊻ 0x80)
for op in (:+, :-, :*, :/)
    @eval Base.$op(a::Float8E4M3, b::Float8E4M3) = Float8E4M3($op(Float16(a), Float16(b)))
end
Base.isnan(x::Float8E4M3) = reinterpret(UInt8, x) & 0x7f == F8_NAN
Base.show(io::IO, x::Float8E4M3) = print(io, "Float8E4M3(", Float16(x), ")")

# relaxed atomic add for the boundary kernel: a compare-and-swap of the 32-bit word that holds the byte
@inline function relaxed_add!(ptr::Core.LLVMPtr{Float8E4M3,1}, val::Float8E4M3)
    addr = reinterpret(UInt64, ptr)
    word = reinterpret(Core.LLVMPtr{UInt32,1}, addr & ~UInt64(3))
    shift = UInt32(addr & 3) * 0x08
    old = UnsafeAtomics.load(word, UnsafeAtomics.monotonic, UnsafeAtomics.device)
    while true
        byte = reinterpret(Float8E4M3, UInt8((old >> shift) & 0xff))
        new = (old & ~(UInt32(0xff) << shift)) | (UInt32(reinterpret(UInt8, byte + val)) << shift)
        r = UnsafeAtomics.cas!(word, old, new, UnsafeAtomics.monotonic, UnsafeAtomics.monotonic, UnsafeAtomics.device)
        r.success && return nothing
        old = r.old
    end
end
