# The matrix value type of the linear system JuNe assembles into: NeoN's coupled Vec3 matrix holds
# three identical components per entry, the segregated scalar matrix one. The kernels only know a
# VectorType and its number of components, so the same code assembles both.
abstract type ValueType end

struct Vec3 <: ValueType end
struct Scalar <: ValueType end

ncomp(::Type{Vec3}) = 3
ncomp(::Type{Scalar}) = 1

# The matrix (or boundary matrix) values with their value type. The storage A is a host Vector, a
# CuArray, the CuDeviceArray a kernel sees, or the raw device pointer NeoN hands over for the GPU
# (see wrapGpu).
struct VectorType{T <: ValueType, A}
    data::A
end

VectorType{T}(data::A) where {T <: ValueType, A} = VectorType{T, A}(data)

Adapt.adapt_structure(to, v::VectorType{T}) where {T} = VectorType{T}(Adapt.adapt(to, v.data))

@inline ncomp(::VectorType{T}) where {T} = ncomp(T)
@inline valuetype(::VectorType{T}) where {T} = T

# idx is the 1-based index of the first component of a matrix entry: entry * ncomp + 1
@inline function pluseq!(vec::VectorType{T}, idx, val) where {T}
    @inbounds for c in 0:ncomp(T)-1
        vec.data[idx+c] += val
    end
end

@inline function minuseq!(vec::VectorType{T}, idx, val) where {T}
    @inbounds for c in 0:ncomp(T)-1
        vec.data[idx+c] -= val
    end
end

@inline function eq!(vec::VectorType{T}, idx, val) where {T}
    @inbounds for c in 0:ncomp(T)-1
        vec.data[idx+c] = val
    end
end

# Relaxed (monotonic), device-scope atomic add, the ordering of Kokkos' atomic_add/sub: the assembly
# only needs every update to be atomic, nothing reads the entries before the kernel has finished.
# Atomix ignores the ordering on CUDA (its CUDACore extension calls CUDACore.atomic_add!, which is
# hard-wired to acquire-release) and puts MEMBAR.GPU ... ATOMG ... CCTL.IVALL around every update
# (Nsight Compute: 29 % of the face kernel's stalls on the barrier). UnsafeAtomics with the monotonic
# ordering drops the barriers, on the CPU it is the same atomic add as before.
@inline relaxed_add!(a, i, val) = relaxed_add!(pointer(a, i), convert(eltype(a), val))
@inline function relaxed_add!(ptr, val)
    UnsafeAtomics.modify!(ptr, +, val, UnsafeAtomics.monotonic, UnsafeAtomics.device)
    return nothing
end
# fp64 in GPU global memory: LLVM emits atom.relaxed.gpu.global.add.f64 with a result register, which
# ptxas keeps as an ATOMG (L2 sends the old value back) instead of the fire-and-forget REDG that
# Kokkos' atomic_add/sub compile to. The PTX red instruction states that the result is not needed.
@inline relaxed_add!(ptr::Core.LLVMPtr{Float64,1}, val::Float64) = Base.llvmcall("""
    call void asm sideeffect "red.relaxed.gpu.global.add.f64 [\$0], \$1;", "l,d,~{memory}"(i64 %0, double %1)
    ret void""", Cvoid, Tuple{UInt64,Float64}, reinterpret(UInt64, ptr), val)

@inline relaxed_add!(ptr::Core.LLVMPtr{Float32,1}, val::Float32) = Base.llvmcall("""
    call void asm sideeffect "red.relaxed.gpu.global.add.f32 [\$0], \$1;", "l,f,~{memory}"(i64 %0, float %1)
    ret void""", Cvoid, Tuple{UInt64,Float32}, reinterpret(UInt64, ptr), val)

@inline function pluseq_atomic!(vec::VectorType{T}, idx, val) where {T}
    for c in 0:ncomp(T)-1
        relaxed_add!(vec.data, idx+c, val)
    end
end

@inline function minuseq_atomic!(vec::VectorType{T}, idx, val) where {T}
    for c in 0:ncomp(T)-1
        relaxed_add!(vec.data, idx+c, -val)
    end
end
