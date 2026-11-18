# Precision study of the GPU assembly. NeoN calls benchmark_precision(:<entry>, <the arguments of that
# entry>...) instead of the entry itself (june::assemble with a Julia wrapper) and the system it hands
# over is assembled in a series of precision configurations:
#
#   whole   all operators and the matrix in fp64, fp32, fp16, bf16            labels fp64, fp32, ...
#   reduce  one operator in fp32 or bf16, the others and the matrix in fp64   labels ddt-fp32, div-bf16
#   alone   one operator alone, it and the matrix in each precision           labels only-ddt-fp64, ...
#
# An operator in precision P computes in P from its own inputs stored in P (ddt: volumes and old
# velocity; div: flux, boundary flux, cell-based face signs; laplacian: gamma, delta coefficients,
# face areas). The boundary inputs that div and laplacian share (boundary values, gradients, value
# fractions, boundary delta coefficients) stay in fp64 when only one of them is reduced and are
# converted when read; the operators add into the matrix in its own type.
#
# Measured for every configuration and strategy, each `runs` times, synchronised before and after:
#   Assembly time         reset of matrix and rhs + the JuNe GPU entry
#   Allocation time       the device arrays of the reduced-precision inputs and outputs
#   Conversion time       Float64 -> P of the reduced inputs, on the device
#   Back-conversion time  matrix and rhs -> Float64, if the matrix is not in fp64
# and the deviation of matrix and rhs from the fp64 result of the same expression. fp64 "whole" runs
# on NeoN's own arrays, so NeoN's system holds the fp64 result afterwards.

using Printf

# BFloat16 is Core.BFloat16 (Julia >= 1.11); its arithmetic comes with CUDA.jl through BFloat16s
const PRECISIONS = (Float64, Float32, Float16, Core.BFloat16)
const REDUCED = (Float32, Core.BFloat16)  # one operator at a time
const OPERATORS = (:ddt, :div, :lap)
precision_name(T) = T === Float64 ? "fp64" : T === Float32 ? "fp32" : T === Float16 ? "fp16" : "bf16"

# which operator reads an input; :shared inputs are the boundary data of div and laplacian
const OWNER = Dict(:volumes => :ddt, :oldVectors => :ddt,
                   :faceFlux => :div, :bfaceFlux => :div, :faceSignV => :div,
                   :gamma => :lap, :bgamma => :lap, :deltaCoeffs => :lap, :magFaceArea => :lap,
                   :bdeltaCoeffs => :shared, :valueFractions => :shared, :refValue => :shared,
                   :refGradient => :shared)

# entry -> strategy name, positions of matrix, boundary matrix, rhs, boundary rhs and the
# floating-point inputs (position => (name, length)). The names are the variable names in the entry
# points (ftype), the lengths those of NeoN's arrays: the entries wrap some arrays with a larger length
# than they have (value fractions: boundary faces; face signs: 2 internal faces), harmless there
# because the kernels only read the valid part, but copies read the whole array.
function precision_layout(entry::Symbol, args)
    if entry === :cellBased_gpu
        nCells, nInt, nTot = args[1], args[28], args[29]
        nB = nTot - nInt
        nnz = nCells + 2nInt
        return (name = "cell-based", vals = (8, nnz), bValues = (20, nB), rhs = (26, 3nCells), bRhs = (21, 3nB),
                inputs = [7 => (:faceSignV, 2nInt), 10 => (:faceFlux, nInt), 11 => (:bfaceFlux, nB),
                          12 => (:gamma, nInt), 13 => (:bgamma, nB), 14 => (:deltaCoeffs, nInt),
                          15 => (:bdeltaCoeffs, nB), 17 => (:magFaceArea, nTot), 18 => (:valueFractions, nB),
                          22 => (:volumes, nCells), 23 => (:oldVectors, 3nCells), 24 => (:refValue, 3nB),
                          25 => (:refGradient, 3nB)])
    end
    nInt = args[1]
    nCells, nTot = entry === :assemble_batched_cuda ? (args[30], args[31]) : (args[27], args[28])
    nB = nTot - nInt
    nnz = nCells + 2nInt
    name = entry === :assemble_gpu ? "face-based" : entry === :globalassemble_gpu ? "global face-based" :
           entry === :assemble_batched_cuda ? "batched" : error("benchmark_precision: unsupported entry $entry")
    return (name = name, vals = (8, nnz), bValues = (22, nB), rhs = (20, 3nCells), bRhs = (23, 3nB),
            inputs = [10 => (:faceFlux, nInt), 11 => (:bfaceFlux, nB), 12 => (:gamma, nInt), 13 => (:bgamma, nB),
                      14 => (:deltaCoeffs, nInt), 15 => (:bdeltaCoeffs, nB), 16 => (:magFaceArea, nTot),
                      17 => (:valueFractions, nB), 18 => (:refValue, 3nB), 19 => (:refGradient, 3nB),
                      24 => (:volumes, nCells), 25 => (:oldVectors, 3nCells)])
end

devptr(a::CuArray) = Ptr{Cvoid}(UInt(pointer(a)))

# Float64 -> BFloat16 by hand, through Float32 with round-to-nearest-even on the bits (on the host,
# LLVM 16 cannot select the vectorised x86 truncation that BFloat16s' array conversion compiles to:
# "Cannot select: v8bf16 = X86ISD::VFPROUND"); integer operations only, so it runs on the device as
# well. The inputs are finite, so NaN is not handled.
function to_bf16(x::Float64)
    bits = reinterpret(UInt32, Float32(x))
    return reinterpret(Core.BFloat16, UInt16((bits + 0x00007fff + ((bits >> 16) & 0x00000001)) >> 16))
end
to_f64(x) = Float64(x)
to_f64(x::Core.BFloat16) = Float64(reinterpret(Float32, UInt32(reinterpret(UInt16, x)) << 16))
@inline to_type(::Type{T}, x::Float64) where {T} = T(x)
@inline to_type(::Type{Core.BFloat16}, x::Float64) = to_bf16(x)

operator_of(term) = startswith(term, "Ddt") ? :ddt : startswith(term, "Div") ? :div :
                    startswith(term, "Laplace") ? :lap : error("benchmark_precision: unknown operator $term")

# the configurations for the operators present in the expression
function precision_configs(present)
    configs = Any[(label = precision_name(T), mode = :whole, op = nothing, T = T, mat = T) for T in PRECISIONS]
    for op in OPERATORS, T in REDUCED
        op in present && length(present) > 1 &&
            push!(configs, (label = "$(op)-$(precision_name(T))", mode = :reduce, op = op, T = T, mat = Float64))
    end
    for op in OPERATORS, T in PRECISIONS
        op in present && push!(configs, (label = "only-$(op)-$(precision_name(T))", mode = :alone, op = op, T = T, mat = T))
    end
    return configs
end

# the type an input is stored in for a configuration; Float64 means NeoN's array, no copy
function input_type(c, name)
    owner = OWNER[name]
    c.mode === :whole && return c.T
    owner === c.op && return c.T
    owner === :shared && c.mode === :alone && c.op in (:div, :lap) && return c.T
    return Float64
end

# the operator string of a configuration: every term with its type, or only the operator alone
function config_opstring(opstring, c)
    terms = String[]
    for term in split(opstring, " + ")
        op = operator_of(term)
        c.mode === :alone && op !== c.op && continue
        T = c.mode === :reduce && op !== c.op ? Float64 : c.T
        push!(terms, replace(term, "Float64" => string(T)))  # e.g. Float32, Core.BFloat16
    end
    return join(terms, " + ")
end

# The arguments of a configuration: reduced inputs as device copies, outputs of the matrix type. The
# allocation and the conversion are timed `runs` times; before every allocation the previous copies
# are freed and CUDA.jl's memory pool is trimmed, so the allocation is not served from its cache.
function config_args(args, layout, c, runs)
    c.mode === :whole && c.T === Float64 && return collect(Any, args), Any[], Float64
    outlen(i, n) = n * (args[i] isa VectorType ? ncomp(valuetype(args[i])) : 1)
    outputs = [(key, getfield(layout, key)...) for key in (:vals, :bValues, :rhs, :bRhs)]
    reduced = [(i, name, n) for (i, (name, n)) in layout.inputs if input_type(c, name) !== Float64 && args[i] != C_NULL]
    sources = [ptrToGpu(args[i], n, Float64) for (i, name, n) in reduced]
    timedOutputs = c.mat !== Float64
    inputs, outs = CuArray{c.T,1}[], CuArray[]
    for run in 0:runs-1
        foreach(CUDA.unsafe_free!, inputs); foreach(CUDA.unsafe_free!, outs)
        GC.gc(false); CUDA.reclaim(); CUDA.synchronize()
        t0 = time_ns()
        inputs = [CuArray{c.T}(undef, n) for (i, name, n) in reduced]
        timedOutputs && (outs = [CuArray{c.mat}(undef, outlen(i, n)) for (key, i, n) in outputs])
        CUDA.synchronize()
        t1 = time_ns()
        for (src, dst) in zip(sources, inputs)
            dst .= to_type.(c.T, src)
        end
        CUDA.synchronize()
        t2 = time_ns()
        if !isempty(reduced) || timedOutputs
            println("Allocation time (JuNe $(layout.name) $(c.label) $run) = $((t1 - t0) ÷ 1000) [us]")
            println("Conversion time (JuNe $(layout.name) $(c.label) $run) = $((t2 - t1) ÷ 1000) [us]")
            flush(stdout)
        end
    end
    # fp64 outputs of the other configurations: not part of a reduced-precision cost (NeoN's fp64
    # system exists anyway), but separate arrays, so NeoN's system keeps the fp64 reference
    timedOutputs || (outs = [CuArray{Float64}(undef, outlen(i, n)) for (key, i, n) in outputs])
    newargs = collect(Any, args)
    types = Dict{Symbol,Any}(:default => c.mat)
    for (name, n) in last.(layout.inputs)
        types[name] = Float64
    end
    for ((i, name, n), a) in zip(reduced, inputs)
        newargs[i] = devptr(a)
        types[name] = c.T
    end
    for ((key, i, n), a) in zip(outputs, outs)
        newargs[i] = args[i] isa VectorType ? VectorType{valuetype(args[i])}(devptr(a)) : devptr(a)
    end
    newargs[9] = config_opstring(args[9], c)
    # an operator alone: without ddt the cell-based kernel must not read the old velocity, which it
    # only skips for an empty array
    if c.mode === :alone && c.op !== :ddt
        for (i, (name, n)) in layout.inputs
            name in (:volumes, :oldVectors) && (newargs[i] = C_NULL)
        end
    end
    return newargs, Any[inputs, outs], (; types...)
end

# Back-conversion time: matrix and rhs -> Float64 on the device, into preallocated Float64 arrays,
# the way a Float64 solver would receive them
function time_back_conversion(args, layout, c, runs)
    c.mat === Float64 && return
    vi, vn = layout.vals
    ri, rn = layout.rhs
    vals = ptrToGpu(args[vi].data, vn * ncomp(valuetype(args[vi])), c.mat)
    rhs = ptrToGpu(args[ri], rn, c.mat)
    vals64, rhs64 = CuArray{Float64}(undef, length(vals)), CuArray{Float64}(undef, length(rhs))
    for run in 0:runs-1
        CUDA.synchronize()
        t0 = time_ns()
        vals64 .= to_f64.(vals)
        rhs64 .= to_f64.(rhs)
        CUDA.synchronize()
        println("Back-conversion time (JuNe $(layout.name) $(c.label) $run) = $((time_ns() - t0) ÷ 1000) [us]")
    end
    flush(stdout)
    CUDA.unsafe_free!(vals64); CUDA.unsafe_free!(rhs64)
end

# matrix and rhs of an assembly as Float64 host arrays
function precision_result(args, layout, ::Type{T}) where {T}
    vi, vn = layout.vals
    ri, rn = layout.rhs
    S = valuetype(args[vi])
    vals = to_f64.(Array(ptrToGpu(args[vi].data, vn * ncomp(S), T)))
    rhs = to_f64.(Array(ptrToGpu(args[ri], rn, T)))
    return vals, rhs
end

# largest deviation from the Float64 result relative to the largest Float64 entry
normwise(a, ref) = maximum(abs.(a .- ref)) / max(maximum(abs.(ref)), floatmin(Float64))

function reset!(args, layout, ::Type{T}) where {T}
    for key in (:vals, :bValues)
        i, n = getfield(layout, key)
        fill!(ptrToGpu(args[i].data, n * ncomp(valuetype(args[i])), T), zero(T))
    end
    for key in (:rhs, :bRhs)
        i, n = getfield(layout, key)
        fill!(ptrToGpu(args[i], n, T), zero(T))
    end
end

"""
    benchmark_precision(entry::Symbol, args...; runs = 5)

Assembles the system of the GPU entry point `entry` (`assemble_gpu`, `globalassemble_gpu`,
`cellBased_gpu`, `assemble_batched_cuda`) with the arguments NeoN passes to it in every precision
configuration (see the top of this file), `runs` times each. Prints the assembly, allocation,
conversion and back-conversion times and the deviation of matrix and rhs from fp64.
"""
function benchmark_precision(entry::Symbol, args...; runs = parse(Int, get(ENV, "CAVITY_RUNS", "5")))
    f = getfield(JuNe, entry)
    layout = precision_layout(entry, args)
    present = unique(operator_of.(split(args[9], " + ")))
    references = Dict{Any,Any}()
    for c in precision_configs(present)
        newargs, keep, types = config_args(args, layout, c, runs)
        # newargs only holds raw pointers into the copies: keep them from being collected
        vals, rhs = GC.@preserve keep begin
            for run in 0:runs-1
                CUDA.synchronize()
                t0 = time_ns()
                reset!(newargs, layout, c.mat)
                f(newargs...; FT = types)
                CUDA.synchronize()
                us = (time_ns() - t0) ÷ 1000
                println("Assembly time (JuNe $(layout.name) $(c.label) $run) = $us [us]")
                flush(stdout)  # NeoN aborts the process on a Julia error
            end
            time_back_conversion(newargs, layout, c, runs)
            precision_result(newargs, layout, c.mat)
        end
        key = c.mode === :alone ? c.op : :all
        if c.T === Float64
            references[key] = (vals, rhs)
        else
            @printf("Precision deviation (JuNe %s %s): matrix %.3e rhs %.3e\n", layout.name, c.label,
                    normwise(vals, references[key][1]), normwise(rhs, references[key][2]))
        end
        keep = nothing
        GC.gc(false)
    end
    flush(stdout)
    return nothing
end
