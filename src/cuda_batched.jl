using Printf
# Batched face-based GPU assembly written with CUDA.jl kernels directly instead of
# KernelAbstractions: same kernels, same launch sequence, same per-batch synchronisation as
# assemble_batched_gpu, to compare the two libraries' host overhead.

function cuda_ddt_kernel!(temporals, volumes, oldVectors, diagOffs, rowOffs, vals, RHS, numCells)
	z = zero(eltype(RHS)) # accumulators in the matrix type
	celli = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	celli > numCells && return nothing
	idx2D = (celli - 1) * 3 + 1
	valueDiag, rx, ry, rz = temporals(
		oldVectors[idx2D], oldVectors[idx2D+1], oldVectors[idx2D+2], volumes[celli], z, z, z, z
	)
	idx = (rowOffs[celli] + diagOffs[celli]) * ncomp(vals) + 1
	pluseq!(vals, idx, valueDiag)
	RHS[idx2D] += rx
	RHS[idx2D+1] += ry
	RHS[idx2D+2] += rz
	return nothing
end

function cuda_batched_innerFace_kernel!(
	batchFaces, offset, owner, neighbour, diagOffs, ownOffs, neiOffs, rowOffs, vals,
	faceFlux, gamma, deltaCoeffs, magFaceArea, fused_pde, n
)
	z = zero(eltype(vals.data)) # accumulators in the matrix type
	i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	i > n && return nothing
	iFace = batchFaces[offset + i] + 1
	iOwner = owner[iFace] + 1
	iNeighbor = neighbour[iFace] + 1
	rowNeiStart = rowOffs[iNeighbor]
	rowOwnStart = rowOffs[iOwner]
	valueUpper, valueLower = fused_pde(
		faceFlux[iFace], gamma[iFace], deltaCoeffs[iFace], magFaceArea[iFace], z, z
	)
	idx = (rowNeiStart + neiOffs[iFace]) * ncomp(vals) + 1
	eq!(vals, idx, valueUpper)
	idx = (rowOwnStart + diagOffs[iOwner]) * ncomp(vals) + 1
	minuseq!(vals, idx, valueUpper)
	idx = (rowOwnStart + ownOffs[iFace]) * ncomp(vals) + 1
	eq!(vals, idx, valueLower)
	idx = (rowNeiStart + diagOffs[iNeighbor]) * ncomp(vals) + 1
	minuseq!(vals, idx, valueLower)
	bs = boundedScale(fused_pde)
	if bs != 0.0
		ϕs = faceFlux[iFace] * bs
		idx = (rowOwnStart + diagOffs[iOwner]) * ncomp(vals) + 1
		minuseq!(vals, idx, ϕs)
		idx = (rowNeiStart + diagOffs[iNeighbor]) * ncomp(vals) + 1
		pluseq!(vals, idx, ϕs)
	end
	return nothing
end

function cuda_face_boundaryKernel!(
	numInteriorFaces, surfaceCells, diagOffs, rowOffs, vals, spatialOperators, bfaceFlux, bgamma,
	bdeltaCoeffs, magFaceArea, valueFractions, refValue, refGradient, RHS, bValues, bRhs, nBoundary
)
	z = zero(eltype(RHS)) # accumulators in the matrix type
	bcfacei = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	bcfacei > nBoundary && return nothing
	iFace = bcfacei + numInteriorFaces
	start = bcfacei * 3 - 2
	valueDiag, valueRHSx, valueRHSy, valueRHSz = spatialOperators(
		refValue[start], refValue[start+1], refValue[start+2],
		refGradient[start], refGradient[start+1], refGradient[start+2],
		bfaceFlux[bcfacei], valueFractions[bcfacei], bdeltaCoeffs[bcfacei], bgamma[bcfacei],
		magFaceArea[iFace], z, z, z, z
	)
	own = surfaceCells[bcfacei] + 1
	vIdx = (rowOffs[own] + diagOffs[own]) * ncomp(vals) + 1
	pluseq!(bValues, (bcfacei-1)*ncomp(bValues)+1, valueDiag)
	pluseq_atomic!(vals, vIdx, valueDiag)
	CUDA.@atomic RHS[own*3-2] += valueRHSx
	CUDA.@atomic RHS[own*3-1] += valueRHSy
	CUDA.@atomic RHS[own*3]   += valueRHSz
	bRhs[bcfacei*3-2] += valueRHSx
	bRhs[bcfacei*3-1] += valueRHSy
	bRhs[bcfacei*3]   += valueRHSz
	return nothing
end

const CUDA_THREADS = 256

# JUNE_TRACE=1: record the host time at which every step of assemble_batched_cuda completes.
# Read at load time in __init__, not at precompile time.
const TRACE_STEPS = Ref(false)
function __init__()
	TRACE_STEPS[] = get(ENV, "JUNE_TRACE", "0") == "1"
end
struct StepTrace
	t0::UInt64
	steps::Vector{Tuple{String,UInt64}}
end
StepTrace() = StepTrace(time_ns(), Tuple{String,UInt64}[])
@inline step!(tr::StepTrace, name) = TRACE_STEPS[] && push!(tr.steps, (name, time_ns()))
function report(tr::StepTrace)
	TRACE_STEPS[] || return
	prev = tr.t0
	println("JUNE_TRACE assemble_batched_cuda: step, time since previous step, time since call start [ms]")
	for (name, t) in tr.steps
		@printf("  %-30s %9.3f %9.3f\n", name, (t - prev) / 1e6, (t - tr.t0) / 1e6)
		prev = t
	end
end

function assemble_batched_cuda(
    numInteriorFaces::Int32,
	_owner::Ptr{Cvoid},
	_neighbour::Ptr{Cvoid},
	_diagOffs::Ptr{Cvoid},
	_ownOffs::Ptr{Cvoid},
	_neiOffs::Ptr{Cvoid},
	_rowOffs::Ptr{Cvoid},
	_vals::VectorType,
	opString::String,
	_faceFlux::Ptr{Cvoid},
	_bfaceFlux::Ptr{Cvoid},
	_gamma::Ptr{Cvoid},
	_bgamma::Ptr{Cvoid},
	_deltaCoeffs::Ptr{Cvoid},
	_bdeltaCoeffs::Ptr{Cvoid},
	_magFaceArea::Ptr{Cvoid},
	_valueFractions::Ptr{Cvoid},
	_refValue::Ptr{Cvoid},
	_refGradient::Ptr{Cvoid},
	_RHS::Ptr{Cvoid},
	_surfaceCells::Ptr{Cvoid},
	_bValues::VectorType,
	_bRhs::Ptr{Cvoid},
	_volumes::Ptr{Cvoid},
	_oldVectors::Ptr{Cvoid},
	_dt::Float64,
	nBatches::Int32,
	_batchSegments::Ptr{Cvoid},
	_batchFaces::Ptr{Cvoid},
	numCells::Int32,
	numTotalFaces::Int32;
	FT = Float64 # precision of the floating-point arrays behind the pointers: a type, or a NamedTuple name => type with a default
)
	tr = StepTrace()
	boundaryFaces = numTotalFaces - numInteriorFaces
	owner = ptrToGpu(_owner, numInteriorFaces, Int32)
	neighbour = ptrToGpu(_neighbour, numInteriorFaces, Int32)
	diagOffs = ptrToGpu(_diagOffs, numCells, UInt8)
	ownOffs = ptrToGpu(_ownOffs, numInteriorFaces, UInt8)
	neiOffs = ptrToGpu(_neiOffs, numInteriorFaces, UInt8)
	rowOffs = ptrToGpu(_rowOffs, numCells, Int32)
	vals = wrapGpu(_vals, numCells + 2*numInteriorFaces, ftype(FT, :vals))
	faceFlux = ptrToGpu(_faceFlux, numInteriorFaces, ftype(FT, :faceFlux))
	bfaceFlux = ptrToGpu(_bfaceFlux, boundaryFaces, ftype(FT, :bfaceFlux))
	gamma = ptrToGpu(_gamma, numInteriorFaces, ftype(FT, :gamma))
	bgamma = ptrToGpu(_bgamma, boundaryFaces, ftype(FT, :bgamma))
	deltaCoeffs = ptrToGpu(_deltaCoeffs, numInteriorFaces, ftype(FT, :deltaCoeffs))
	bdeltaCoeffs = ptrToGpu(_bdeltaCoeffs, boundaryFaces, ftype(FT, :bdeltaCoeffs))
	magFaceArea = ptrToGpu(_magFaceArea, numTotalFaces, ftype(FT, :magFaceArea))
	valueFractions = ptrToGpu(_valueFractions, numTotalFaces, ftype(FT, :valueFractions))
	refValue = ptrToGpu(_refValue, boundaryFaces*3, ftype(FT, :refValue))
	refGradient = ptrToGpu(_refGradient, boundaryFaces*3, ftype(FT, :refGradient))
	RHS = ptrToGpu(_RHS, numCells*3, ftype(FT, :RHS))
	surfaceCells = ptrToGpu(_surfaceCells, boundaryFaces, Int32)
	bValues = wrapGpu(_bValues, boundaryFaces, ftype(FT, :bValues))
	bRhs = ptrToGpu(_bRhs, boundaryFaces*3, ftype(FT, :bRhs))
	volumes = ptrToGpu(_volumes, numCells, ftype(FT, :volumes))
	oldVectors = ptrToGpu(_oldVectors, numCells*3, ftype(FT, :oldVectors))
	step!(tr, "wrap pointers")
	batchSegments = Array(ptrToGpu(_batchSegments, nBatches + 1, Int32))
	batchFaces = ptrToGpu(_batchFaces, batchSegments[end], Int32)
	step!(tr, "segments to host")
	dt = "$_dt"
	opstring2 = replace(opString, "DELTAT" => dt)
	opstring3 = replace(opstring2, "BDF2" => "BDF1")
	fused_pde = eval(Meta.parse(opstring3))
	step!(tr, "eval op string")
	ddt, spatials = JuNe.splitTempSpat(fused_pde)
	step!(tr, "splitTempSpat")

	if !isnothing(ddt)
		@cuda threads=CUDA_THREADS blocks=cld(numCells, CUDA_THREADS) cuda_ddt_kernel!(
			ddt, volumes, oldVectors, diagOffs, rowOffs, vals, RHS, numCells)
		step!(tr, "launch ddt")
	end
	if !isnothing(spatials)
		for batch in 1:nBatches
			n = batchSegments[batch+1] - batchSegments[batch]
			n == 0 && continue
			@cuda threads=CUDA_THREADS blocks=cld(n, CUDA_THREADS) cuda_batched_innerFace_kernel!(
				batchFaces, batchSegments[batch], owner, neighbour, diagOffs, ownOffs, neiOffs, rowOffs,
				vals, faceFlux, gamma, deltaCoeffs, magFaceArea, spatials, n)
			step!(tr, "launch batch $batch")
		end
		@cuda threads=CUDA_THREADS blocks=cld(boundaryFaces, CUDA_THREADS) cuda_face_boundaryKernel!(
			numInteriorFaces, surfaceCells, diagOffs, rowOffs, vals, spatials, bfaceFlux, bgamma,
			bdeltaCoeffs, magFaceArea, valueFractions, refValue, refGradient, RHS, bValues, bRhs,
			boundaryFaces)
		step!(tr, "launch boundary")
		# the only wait: every kernel sits on the task's stream in launch order, the caller reads
		# the matrix after the call returns
		CUDA.synchronize()
		step!(tr, "sync all")
	end
	report(tr)
end
