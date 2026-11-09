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

@inline function pluseq_atomic!(vec::VectorType{T}, idx, val) where {T}
    for c in 0:ncomp(T)-1
        Atomix.@atomic vec.data[idx+c] += val
    end
end

@inline function minuseq_atomic!(vec::VectorType{T}, idx, val) where {T}
    for c in 0:ncomp(T)-1
        Atomix.@atomic vec.data[idx+c] -= val
    end
end
