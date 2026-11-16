using Test
using JuNe
using JuNe: VectorType, Vec3, Scalar
using CUDA

# A 1D chain of n cells: internal face f (zero-based) joins cell f and cell f+1, the two boundary
# faces belong to cell 0 and cell n-1. The CSR addressing is what NeoN hands to JuNe: rowOffs per
# cell, the position of the diagonal in its row, and per face the positions of the owner's and the
# neighbour's off-diagonal entry in their rows.
function chainAddressing(n)
    nInt = n - 1
    owner = Int32.(0:nInt-1)
    neighbour = Int32.(1:nInt)
    rowOffs = zeros(Int32, n + 1)
    for c in 0:n-1
        rowOffs[c+2] = rowOffs[c+1] + 1 + (c > 0) + (c < n - 1)
    end
    diagOffs = UInt8[c > 0 ? 1 : 0 for c in 0:n-1]
    ownOffs = UInt8[diagOffs[f+1] + 1 for f in 0:nInt-1] # column f+1 sits right of the diagonal
    neiOffs = zeros(UInt8, nInt)                         # column f is the first entry of row f+1
    return owner, neighbour, rowOffs, diagOffs, ownOffs, neiOffs
end

# Deterministic input fields; the matrix, rhs and boundary accumulators start at zero.
function chainInputs(n)
    nInt = n - 1
    nTotal = nInt + 2
    owner, neighbour, rowOffs, diagOffs, ownOffs, neiOffs = chainAddressing(n)
    nnz = rowOffs[end]
    return (
        numInteriorFaces = Int32(nInt),
        owner = owner,
        neighbour = neighbour,
        diagOffs = diagOffs,
        ownOffs = ownOffs,
        neiOffs = neiOffs,
        rowOffs = rowOffs,
        vals = VectorType{Vec3}(zeros(3 * nnz)),
        opString = "Ddt{Float64, BDF1{Float64}}(BDF1(DELTAT), 1.000000) + " *
                   "Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.000000, true) + " *
                   "Laplace{Float64}(-1.000000)",
        faceFlux = [sin(1.3k) for k in 1:nInt],
        bfaceFlux = [-0.4, 0.6],
        gamma = [1.0 + 0.1cos(0.7k) for k in 1:nInt],
        bgamma = [1.1, 0.9],
        deltaCoeffs = [1.0 + 0.2sin(0.4k) for k in 1:nInt],
        bdeltaCoeffs = [2.0, 2.5],
        magFaceArea = [1.0 + 0.05cos(1.1k) for k in 1:nTotal],
        valueFractions = [1.0, 0.0, zeros(nTotal - 2)...],   # fixedValue at cell 0, zeroGradient at n-1
        refValue = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        refGradient = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
        RHS = zeros(3n),
        surfaceCells = Int32[0, n - 1],
        bValues = VectorType{Vec3}(zeros(6)),
        bRhs = zeros(6),
        volumes = [1.0 + 0.1sin(0.3k) for k in 1:n],
        oldVectors = [sin(0.9k) for k in 1:3n],
        dt = 0.01,
    )
end

faceArgs(in) = (
    in.numInteriorFaces, in.owner, in.neighbour, in.diagOffs, in.ownOffs, in.neiOffs, in.rowOffs,
    in.vals, in.opString, in.faceFlux, in.bfaceFlux, in.gamma, in.bgamma, in.deltaCoeffs,
    in.bdeltaCoeffs, in.magFaceArea, in.valueFractions, in.refValue, in.refGradient, in.RHS,
    in.surfaceCells, in.bValues, in.bRhs, in.volumes, in.oldVectors, in.dt,
)

# Faces f and f+1 share cell f+1, so even and odd faces are the two batches of the chain.
function chainBatches(nInt)
    even = Int32.(0:2:nInt-1)
    odd = Int32.(1:2:nInt-1)
    return (
        nBatches = Int32(2),
        batchSegments = Int32[0, length(even), nInt],
        batchFaces = vcat(even, odd),
    )
end

batchArgs(c) = (c.nBatches, c.batchSegments, c.batchFaces)

@testset "batched face-based assembly" begin
    n = 17
    reference = chainInputs(n)
    JuNe.assemble(faceArgs(reference)...)

    batches = chainBatches(n - 1)
    batched = chainInputs(n)
    JuNe.assemble_batched(faceArgs(batched)..., batchArgs(batches)...)

    @testset "equals the face-based assembly" begin
        @test batched.vals.data ≈ reference.vals.data atol = 1e-12
        @test batched.RHS ≈ reference.RHS atol = 1e-12
        @test batched.bValues.data ≈ reference.bValues.data atol = 1e-12
        @test batched.bRhs ≈ reference.bRhs atol = 1e-12
    end

    @testset "is bitwise reproducible" begin
        again = chainInputs(n)
        JuNe.assemble_batched(faceArgs(again)..., batchArgs(batches)...)
        @test again.vals.data == batched.vals.data
        @test again.RHS == batched.RHS
    end

    if CUDA.functional()
        @testset "gpu equals the face-based assembly" begin
            gpu = chainInputs(n)
            dev = Dict(k => CuArray(v isa VectorType ? v.data : v) for (k, v) in pairs(gpu) if v isa Union{AbstractVector, VectorType})
            ptr(x) = Ptr{Cvoid}(UInt(pointer(x)))
            devBatches = Dict(k => CuArray(v) for (k, v) in pairs(batches) if v isa AbstractVector)
            JuNe.assemble_batched_gpu(
                gpu.numInteriorFaces, ptr(dev[:owner]), ptr(dev[:neighbour]), ptr(dev[:diagOffs]),
                ptr(dev[:ownOffs]), ptr(dev[:neiOffs]), ptr(dev[:rowOffs]), VectorType{Vec3}(ptr(dev[:vals])),
                gpu.opString, ptr(dev[:faceFlux]), ptr(dev[:bfaceFlux]), ptr(dev[:gamma]),
                ptr(dev[:bgamma]), ptr(dev[:deltaCoeffs]), ptr(dev[:bdeltaCoeffs]),
                ptr(dev[:magFaceArea]), ptr(dev[:valueFractions]), ptr(dev[:refValue]),
                ptr(dev[:refGradient]), ptr(dev[:RHS]), ptr(dev[:surfaceCells]), VectorType{Vec3}(ptr(dev[:bValues])),
                ptr(dev[:bRhs]), ptr(dev[:volumes]), ptr(dev[:oldVectors]), gpu.dt,
                batches.nBatches, ptr(devBatches[:batchSegments]), ptr(devBatches[:batchFaces]),
                Int32(n), Int32(n + 1),
            )
            @test Array(dev[:vals]) ≈ reference.vals.data atol = 1e-12
            @test Array(dev[:RHS]) ≈ reference.RHS atol = 1e-12
        end
    else
        @info "CUDA not functional: skipping the GPU test"
    end
end
