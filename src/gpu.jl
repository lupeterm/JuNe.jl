using CUDA
using KernelAbstractions
using Atomix
using StaticArrays

# NeoN hands the matrix values over as a VectorType around the raw device pointer; this makes it a
# VectorType around a CuArray of n entries times the components of the value type
wrapGpu(w::VectorType{T}, n, FT = Float64) where {T} = VectorType{T}(ptrToGpu(w.data, n * ncomp(T), FT))

# the element type of the array `name` (its variable name in the entry points): FT is a type for all
# arrays, or a NamedTuple of types per array with an entry `default` for the others
ftype(FT::Type, ::Symbol) = FT
ftype(FT::NamedTuple, name::Symbol) = haskey(FT, name) ? FT[name] : FT.default

function ptrToGpu(ptr::Ptr{Cvoid}, size, dType)
	# inputs of absent operators arrive as empty NeoN vectors, i.e. a null pointer
	ptr == C_NULL && return CuArray{dType}(undef, 0)
	p = reinterpret(CuPtr{dType}, ptr)
    arr = unsafe_wrap(CuArray, p, Int64(size); own=false)
	return arr
end

function globalassemble_gpu(
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
	numCells::Int32,
	numTotalFaces::Int32;
	FT = Float64 # precision of the floating-point arrays behind the pointers: a type, or a NamedTuple name => type with a default
)
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
	bRhs= ptrToGpu(_bRhs, boundaryFaces*3, ftype(FT, :bRhs))
	volumes= ptrToGpu(_volumes, numCells, ftype(FT, :volumes))
	oldVectors= ptrToGpu(_oldVectors, numCells*3, ftype(FT, :oldVectors))
	
	dt = "$_dt"
    opstring2 = replace(opString, "DELTAT" => dt)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")
    fused_pde = eval(Meta.parse(opstring3))

    ddt, spatials = JuNe.splitTempSpat(fused_pde)
	backend = get_backend(vals.data)
	if !isnothing(ddt)
		ddt_kernel(backend, 64)(
			ddt,
			volumes,
			oldVectors,
			diagOffs,
			rowOffs,
			vals,
			RHS;
			ndrange=numCells
		)		
		KernelAbstractions.synchronize(backend)
	end
	if !isnothing(spatials)
		globalface_kernel(backend, 64)(
			numInteriorFaces,
			owner,
			neighbour,
			diagOffs,
			ownOffs,
			neiOffs,
			rowOffs,
			vals,
			faceFlux,
			bfaceFlux,
			gamma,
			bgamma,
			deltaCoeffs,
			bdeltaCoeffs,
			magFaceArea,
			valueFractions,
			refValue,
			refGradient,
			RHS,
			surfaceCells,
			bValues,
			bRhs,
			spatials;
			ndrange=numTotalFaces
		)
		KernelAbstractions.synchronize(backend)
	end
end

@kernel function ddt_kernel(
	@Const(temporals),
    @Const(volumes),
    @Const(oldVectors),
    @Const(diagOffs),
    @Const(rowOffs),
    vals,
    RHS
)
	z = zero(eltype(RHS)) # accumulators in the matrix type
	celli = @index(Global)
	idx2D = (celli - 1) * 3 + 1
	valueDiag, rx, ry, rz = temporals(
		oldVectors[idx2D],
		oldVectors[idx2D+1],
		oldVectors[idx2D+2],
		volumes[celli],
		z,
		z,
		z,
		z
	)
	idx = (rowOffs[celli] + diagOffs[celli]) * ncomp(vals) + 1
	pluseq!(vals, idx, valueDiag)
	RHS[idx2D] += rx
	RHS[idx2D+1] += ry
	RHS[idx2D+2] += rz
end

@kernel function globalface_kernel(
    numInteriorFaces,
    @Const(owner),
	@Const(neighbour),
    @Const(diagOffs),
    @Const(ownOffs),    
	@Const(neiOffs),
    @Const(rowOffs),
    vals,
    @Const(faceFlux),  
    @Const(bfaceFlux),  
    @Const(gamma),   
    @Const(bgamma),   
    @Const(deltaCoeffs),
    @Const(bdeltaCoeffs),
    @Const(magFaceArea),
    @Const(valueFractions),
    @Const(refValue),
    @Const(refGradient),
    RHS,    
    @Const(surfaceCells),
    bValues,
    bRhs,
	@Const(fused_pde)
)
	z = zero(eltype(RHS)) # accumulators in the matrix type
	iFace = @index(Global)

	if iFace <= numInteriorFaces        
		iOwner = owner[iFace] + 1
		iNeighbor = neighbour[iFace] + 1

        rowNeiStart = rowOffs[iNeighbor]
      	
		rowOwnStart = rowOffs[iOwner]

        valueUpper, valueLower = fused_pde(
            faceFlux[iFace], 
            gamma[iFace], 
            deltaCoeffs[iFace], 
            magFaceArea[iFace], 
            z, 
            z
        )

		idx = (rowNeiStart + neiOffs[iFace]) * ncomp(vals) + 1
        pluseq!(vals, idx, valueUpper)
		idx = (rowOwnStart + diagOffs[iOwner]) * ncomp(vals) + 1
        minuseq_atomic!(vals, idx, valueUpper)
		idx = (rowOwnStart + ownOffs[iFace]) * ncomp(vals) + 1
		pluseq!(vals, idx, valueLower)
		idx = (rowNeiStart + diagOffs[iNeighbor]) * ncomp(vals) + 1
        # FIXME when going back to scalar values

        minuseq_atomic!(vals, idx, valueLower)
        bs = boundedScale(fused_pde)
        if bs != 0.0
            ϕs = faceFlux[iFace] * bs
            idx = (rowOwnStart + diagOffs[iOwner]) * ncomp(vals) + 1
            minuseq_atomic!(vals, idx, ϕs)
            idx = (rowNeiStart + diagOffs[iNeighbor]) * ncomp(vals) + 1
            pluseq_atomic!(vals, idx, ϕs)
        end
	else
		bcfacei = iFace - numInteriorFaces
        start = bcfacei * 3 - 2
        fluxContrib, valueRHSx, valueRHSy, valueRHSz = fused_pde(
			refValue[start],
			refValue[start+1],
			refValue[start+2],
			refGradient[start],
			refGradient[start+1],
			refGradient[start+2],
			bfaceFlux[bcfacei],
			valueFractions[bcfacei],
			bdeltaCoeffs[bcfacei],
			bgamma[bcfacei],
			magFaceArea[iFace],
			z, z, z, z
        )


		own = surfaceCells[bcfacei] + 1
		vIdx = (rowOffs[own] + diagOffs[own]) * ncomp(vals) + 1
        pluseq!(bValues, (bcfacei-1)*ncomp(bValues)+1, fluxContrib)
		pluseq_atomic!(vals, vIdx, fluxContrib)
        # # FIXME dont forget, changed this back to [xyzxyzxyz] instead of [xxxyyyzzz] for now
        Atomix.@atomic RHS[own*3-2] += valueRHSx
        Atomix.@atomic RHS[own*3-1] += valueRHSy
        Atomix.@atomic RHS[own*3] += valueRHSz
        
		bRhs[bcfacei*3-2] += valueRHSx
        bRhs[bcfacei*3-1] += valueRHSy
        bRhs[bcfacei*3] += valueRHSz
    end
end

function checkindex(index, arr)
	if index > length(arr)
		@print("vidx: $index, $(length(arr))")
	end
end

function cellBased_gpu(
    numCells::Int32,
    owners_::Ptr{Cvoid},
    cellFacesSegments_::Ptr{Cvoid},
    diagOffs_::Ptr{Cvoid},
    rowOffs_::Ptr{Cvoid},
    cellFacesValues_::Ptr{Cvoid},
    faceSignV_::Ptr{Cvoid},
    vals_::VectorType,
    opString_::String,
    faceFlux_::Ptr{Cvoid},
    bfaceFlux_::Ptr{Cvoid},
    gamma_::Ptr{Cvoid},
    bgamma_::Ptr{Cvoid},
    deltaCoeffs_::Ptr{Cvoid},
    bdeltaCoeffs_::Ptr{Cvoid},
    matrixColumnIdxV_::Ptr{Cvoid},
    magFaceArea_::Ptr{Cvoid},
    valueFractions_::Ptr{Cvoid},
    surfaceCells_::Ptr{Cvoid},
    bValues_::VectorType,
    bRhs_::Ptr{Cvoid},
    volumes_::Ptr{Cvoid},
    oldVectors_::Ptr{Cvoid},
    refValue_::Ptr{Cvoid},
    refGradient_::Ptr{Cvoid},
    RHS_::Ptr{Cvoid},
    _dt::Float64,
    numInteriorFaces::Int32,
	numTotalFaces::Int32;
	FT = Float64 # precision of the floating-point arrays behind the pointers: a type, or a NamedTuple name => type with a default
)
	numBoundaryFaces = numTotalFaces - numInteriorFaces
	cellFacesSegments = ptrToGpu(cellFacesSegments_, numCells+1, Int32)
	diagOffs = ptrToGpu(diagOffs_, numCells, UInt8)
	rowOffs = ptrToGpu(rowOffs_, numCells, Int32)
	cellFacesValues = ptrToGpu(cellFacesValues_, (numCells + 2*numInteriorFaces), Int32)
	faceSignV = ptrToGpu(faceSignV_, (numCells + 2*numInteriorFaces), ftype(FT, :faceSignV))
	vals = wrapGpu(vals_, numCells + 2*numInteriorFaces, ftype(FT, :vals))
	faceFlux = ptrToGpu(faceFlux_, numInteriorFaces, ftype(FT, :faceFlux))
	bfaceFlux = ptrToGpu(bfaceFlux_, numBoundaryFaces, ftype(FT, :bfaceFlux))
	gamma = ptrToGpu(gamma_, numInteriorFaces, ftype(FT, :gamma))
	bgamma = ptrToGpu(bgamma_, numBoundaryFaces, ftype(FT, :bgamma))
	deltaCoeffs = ptrToGpu(deltaCoeffs_, numInteriorFaces, ftype(FT, :deltaCoeffs))
	bdeltaCoeffs = ptrToGpu(bdeltaCoeffs_, numBoundaryFaces, ftype(FT, :bdeltaCoeffs))
	matrixColumnIdxV = ptrToGpu(matrixColumnIdxV_, (numCells + 2*numInteriorFaces), Int32)
	magFaceArea = ptrToGpu(magFaceArea_, numTotalFaces, ftype(FT, :magFaceArea))
	valueFractions = ptrToGpu(valueFractions_, numTotalFaces, ftype(FT, :valueFractions))
	surfaceCells = ptrToGpu(surfaceCells_, numBoundaryFaces, Int32)
	bValues = wrapGpu(bValues_, numBoundaryFaces, ftype(FT, :bValues))
	bRhs = ptrToGpu(bRhs_, numBoundaryFaces*3, ftype(FT, :bRhs))
	volumes = ptrToGpu(volumes_, numCells, ftype(FT, :volumes))
	oldVectors = ptrToGpu(oldVectors_, numCells*3, ftype(FT, :oldVectors))
	refValue = ptrToGpu(refValue_, numBoundaryFaces*3, ftype(FT, :refValue))
	refGradient = ptrToGpu(refGradient_, numBoundaryFaces*3, ftype(FT, :refGradient))
	RHS = ptrToGpu(RHS_, numCells*3, ftype(FT, :RHS))

	dt = "$_dt"
    opstring2 = replace(opString_, "DELTAT" => dt)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")
    fused_pde = eval(Meta.parse(opstring3))

    ddt, spatials = JuNe.splitTempSpat(fused_pde)
	backend = get_backend(vals.data)
	# the fallbacks for absent operators capture a zero of the precision, not FT itself: a closure
	# over a type is not isbits and cannot be passed to a kernel
	zFT = zero(ftype(FT, :vals))
	hasSpatials = !isnothing(spatials)
	temporals = !isnothing(ddt) ? ddt : f(args...) = (zFT, zFT, zFT, zFT)
    spatials = hasSpatials ? spatials : g(args...) = (zFT, zFT)
	cellbased_kernel(backend, 64)(
		cellFacesSegments,
		diagOffs,
		rowOffs,
		cellFacesValues,
		faceSignV,
		vals,
		temporals,
		spatials,
		faceFlux,
		gamma,
		deltaCoeffs,
		matrixColumnIdxV,
		magFaceArea,
		volumes,
		oldVectors,
		RHS;
		ndrange=numCells
	)
	hasSpatials && face_boundaryKernel(backend, 64)(
		numInteriorFaces,
		surfaceCells,
		diagOffs,
		rowOffs,
		vals,
		spatials,
		bfaceFlux,
		bgamma,
		bdeltaCoeffs,
		magFaceArea,
		valueFractions,
		refValue,
		refGradient,
		RHS,
		bValues,
		bRhs;
		ndrange=numBoundaryFaces
	)
	KernelAbstractions.synchronize(backend)
end

@kernel function cellbased_kernel(
    @Const(cellFacesSegments),
    @Const(diagOffs),
    @Const(rowOffs),
    @Const(cellFacesValues),
    @Const(faceSignV),
    vals,
    @Const(temporals),
    @Const(spatials),
    @Const(faceFlux),
    @Const(gamma),
    @Const(deltaCoeffs),
    @Const(matrixColumnIdxV),
    @Const(magFaceArea),
    @Const(volumes),
    @Const(oldVectors),
    RHS,
)
	z = zero(eltype(RHS)) # accumulators in the matrix type
	celli = @index(Global)
	numInternalFaces = cellFacesSegments[celli+1] - cellFacesSegments[celli]

	diagValue = z
	startIdx = cellFacesSegments[celli]
	for i in 1:numInternalFaces
		faceIdx = cellFacesValues[startIdx+i] + 1
		diagValue, offDiagValue = spatials(
			faceFlux[faceIdx],
			gamma[faceIdx],
			deltaCoeffs[faceIdx],
			magFaceArea[faceIdx],
			diagValue,
			z,
			faceSignV[startIdx+i]
		)
		fIdx = matrixColumnIdxV[startIdx+i] * ncomp(vals) + 1
		pluseq!(vals, fIdx, offDiagValue)
		# diagValue -= dVal
	end
	idx2D = celli * 3 -2
	dv, rx, ry, rz = z, z, z, z
	if length(oldVectors) > 0 # empty without a ddt operator
		dv, rx, ry, rz = temporals(
			oldVectors[idx2D],
			oldVectors[idx2D+1],
			oldVectors[idx2D+2],
			volumes[celli],
			z,
			z,
			z,
			z
		)
	end
	diagValue += dv

	diagIdx = (rowOffs[celli] + diagOffs[celli]) * ncomp(vals) + 1
	pluseq!(vals, diagIdx, diagValue)
	RHS[idx2D]   += rx
	RHS[idx2D+1] += ry 
	RHS[idx2D+2] += rz
end

@kernel function face_boundaryKernel(
	@Const(numInteriorFaces),
    @Const(surfaceCells),
    @Const(diagOffs),
    @Const(rowOffs),
    vals,
    @Const(spatialOperators),
    @Const(bfaceFlux),
    @Const(bgamma),
    @Const(bdeltaCoeffs),
    @Const(magFaceArea),
    @Const(valueFractions),
    @Const(refValue),
    @Const(refGradient),
    RHS,
    bValues,
    bRhs
)
	z = zero(eltype(RHS)) # accumulators in the matrix type
	bcfacei = @index(Global)
	iFace = bcfacei + numInteriorFaces
	start = bcfacei * 3 - 2
	valueDiag, valueRHSx, valueRHSy, valueRHSz = spatialOperators(
		refValue[start],
		refValue[start+1],
		refValue[start+2],
		refGradient[start],
		refGradient[start+1],
		refGradient[start+2],
		bfaceFlux[bcfacei],
		valueFractions[bcfacei],
		bdeltaCoeffs[bcfacei],
		bgamma[bcfacei],
		magFaceArea[iFace],
		z, z, z, z
	)
	
	own = surfaceCells[bcfacei] + 1
	vIdx = (rowOffs[own] + diagOffs[own]) * ncomp(vals) + 1
	pluseq!(bValues, (bcfacei-1)*ncomp(bValues)+1, valueDiag)
	pluseq_atomic!(vals, vIdx, valueDiag)
	# # FIXME dont forget, changed this back to [xyzxyzxyz] instead of [xxxyyyzzz] for now
	Atomix.@atomic RHS[own*3-2] += valueRHSx
	Atomix.@atomic RHS[own*3-1] += valueRHSy
	Atomix.@atomic RHS[own*3] += valueRHSz

	bRhs[bcfacei*3-2] += valueRHSx
	bRhs[bcfacei*3-1] += valueRHSy
	bRhs[bcfacei*3] += valueRHSz
end



function assemble_gpu(
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
	numCells::Int32,
	numTotalFaces::Int32;
	FT = Float64 # precision of the floating-point arrays behind the pointers: a type, or a NamedTuple name => type with a default
)
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
	bRhs= ptrToGpu(_bRhs, boundaryFaces*3, ftype(FT, :bRhs))
	volumes= ptrToGpu(_volumes, numCells, ftype(FT, :volumes))
	oldVectors= ptrToGpu(_oldVectors, numCells*3, ftype(FT, :oldVectors))
	dt = "$_dt"
    opstring2 = replace(opString, "DELTAT" => dt)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")
    fused_pde = eval(Meta.parse(opstring3))

    ddt, spatials = JuNe.splitTempSpat(fused_pde)
	backend = get_backend(vals.data)
	if !isnothing(ddt)
		ddt_kernel(backend, 64)(
			ddt,
			volumes,
			oldVectors,
			diagOffs,
			rowOffs,
			vals,
			RHS;
			ndrange=numCells
		)		
		KernelAbstractions.synchronize(backend)
	end
	if !isnothing(spatials)
		innerFace_kernel(backend, 64)(
			owner,
			neighbour,
			diagOffs,
			ownOffs,
			neiOffs,
			rowOffs,
			vals,
			faceFlux,
			gamma,
			deltaCoeffs,
			magFaceArea,
			spatials;
			ndrange=numInteriorFaces
		)
		face_boundaryKernel(backend, 64)(
			numInteriorFaces,
			surfaceCells,
			diagOffs,
			rowOffs,
			vals,
			spatials,
			bfaceFlux,
			bgamma,
			bdeltaCoeffs,
			magFaceArea,
			valueFractions,
			refValue,
			refGradient,
			RHS,
			bValues,
			bRhs;
			ndrange=boundaryFaces
		)
		KernelAbstractions.synchronize(backend)
	end
end

# Batched face-based assembly on the GPU, see assemble_batched: one launch per batch of internal
# faces without atomics, then the face-based boundary kernel.
function assemble_batched_gpu(
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
	# the batch segments are needed on the host for the launch sizes; they are tiny
	batchSegments = Array(ptrToGpu(_batchSegments, nBatches + 1, Int32))
	batchFaces = ptrToGpu(_batchFaces, batchSegments[end], Int32)
	dt = "$_dt"
	opstring2 = replace(opString, "DELTAT" => dt)
	opstring3 = replace(opstring2, "BDF2" => "BDF1")
	fused_pde = eval(Meta.parse(opstring3))

	ddt, spatials = JuNe.splitTempSpat(fused_pde)
	backend = get_backend(vals.data)
	if !isnothing(ddt)
		ddt_kernel(backend, 64)(
			ddt,
			volumes,
			oldVectors,
			diagOffs,
			rowOffs,
			vals,
			RHS;
			ndrange=numCells
		)
	end
	if !isnothing(spatials)
		for batch in 1:nBatches
			n = batchSegments[batch+1] - batchSegments[batch]
			n == 0 && continue
			batched_innerFace_kernel(backend, 64)(
				batchFaces,
				batchSegments[batch],
				owner,
				neighbour,
				diagOffs,
				ownOffs,
				neiOffs,
				rowOffs,
				vals,
				faceFlux,
				gamma,
				deltaCoeffs,
				magFaceArea,
				spatials;
				ndrange=n
			)
		end
		# boundary faces see a single cell each: assembled like in the face-based strategy
		face_boundaryKernel(backend, 64)(
			numInteriorFaces,
			surfaceCells,
			diagOffs,
			rowOffs,
			vals,
			spatials,
			bfaceFlux,
			bgamma,
			bdeltaCoeffs,
			magFaceArea,
			valueFractions,
			refValue,
			refGradient,
			RHS,
			bValues,
			bRhs;
			ndrange=boundaryFaces
		)
		KernelAbstractions.synchronize(backend)
	end
end

# innerFace_kernel over the faces of one batch: no two of them share a cell, so the diagonal
# updates are plain writes
@kernel function batched_innerFace_kernel(
	@Const(batchFaces),
	offset,
	@Const(owner),
	@Const(neighbour),
	@Const(diagOffs),
	@Const(ownOffs),
	@Const(neiOffs),
	@Const(rowOffs),
	vals,
	@Const(faceFlux),
	@Const(gamma),
	@Const(deltaCoeffs),
	@Const(magFaceArea),
	@Const(fused_pde)
)
	z = zero(eltype(vals.data)) # accumulators in the matrix type
	i = @index(Global)
	iFace = batchFaces[offset + i] + 1
	iOwner = owner[iFace] + 1
	iNeighbor = neighbour[iFace] + 1
	rowNeiStart = rowOffs[iNeighbor]
	rowOwnStart = rowOffs[iOwner]

	valueUpper, valueLower = fused_pde(
		faceFlux[iFace],
		gamma[iFace],
		deltaCoeffs[iFace],
		magFaceArea[iFace],
		z,
		z
	)

	idx = (rowNeiStart + neiOffs[iFace]) * ncomp(vals) + 1
	pluseq!(vals, idx, valueUpper)
	idx = (rowOwnStart + diagOffs[iOwner]) * ncomp(vals) + 1
	minuseq!(vals, idx, valueUpper)
	idx = (rowOwnStart + ownOffs[iFace]) * ncomp(vals) + 1
	pluseq!(vals, idx, valueLower)
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
end

@kernel function innerFace_kernel(
    @Const(owner),
	@Const(neighbour),
    @Const(diagOffs),
    @Const(ownOffs),    
	@Const(neiOffs),
    @Const(rowOffs),
    vals,
    @Const(faceFlux),  
    @Const(gamma),   
    @Const(deltaCoeffs),
    @Const(magFaceArea),
	@Const(fused_pde)
)
	z = zero(eltype(vals.data)) # accumulators in the matrix type
	iFace = @index(Global)
	iOwner = owner[iFace] + 1
	iNeighbor = neighbour[iFace] + 1

	rowNeiStart = rowOffs[iNeighbor]
	
	rowOwnStart = rowOffs[iOwner]

	valueUpper, valueLower = fused_pde(
		faceFlux[iFace], 
		gamma[iFace], 
		deltaCoeffs[iFace], 
		magFaceArea[iFace], 
		z, 
		z
	)

	idx = (rowNeiStart + neiOffs[iFace]) * ncomp(vals) + 1
	pluseq!(vals, idx, valueUpper)
	idx = (rowOwnStart + diagOffs[iOwner]) * ncomp(vals) + 1
	minuseq_atomic!(vals, idx, valueUpper)
	idx = (rowOwnStart + ownOffs[iFace]) * ncomp(vals) + 1
	pluseq!(vals, idx, valueLower)
	idx = (rowNeiStart + diagOffs[iNeighbor]) * ncomp(vals) + 1
	# FIXME when going back to scalar values

	minuseq_atomic!(vals, idx, valueLower)
	bs = boundedScale(fused_pde)
	if bs != 0.0
	    ϕs = faceFlux[iFace] * bs
	    idx = (rowOwnStart + diagOffs[iOwner]) * ncomp(vals) + 1
	    minuseq_atomic!(vals, idx, ϕs)
	    idx = (rowNeiStart + diagOffs[iNeighbor]) * ncomp(vals) + 1
	    pluseq_atomic!(vals, idx, ϕs)
	end
end
"""
    warmup_gpu(op::String)

Compiles the GPU kernels of every GPU assembly entry point for the operator string `op` on the
smallest mesh: two cells, one internal face between them and one boundary face on the first cell.
Called once before a simulation, so that no time step pays for the compilation. The arguments are
passed exactly as NeoN passes them (device pointers and Int32 counts), so the compiled kernels are
the ones the simulation launches.
"""
function warmup_gpu(op::String)
	nCells, nInt, nB = Int32(2), Int32(1), Int32(1)
	nTotal = nInt + nB
	nnz = nCells + 2nInt
	for T in (Vec3, Scalar)
		nc = ncomp(T)
		dev = Dict{Symbol,CuArray}(
			:owner => CuArray(Int32[0]), :neighbour => CuArray(Int32[1]),
			:diagOffs => CuArray(UInt8[0, 1]), :ownOffs => CuArray(UInt8[1]), :neiOffs => CuArray(UInt8[0]),
			:rowOffs => CuArray(Int32[0, 2, 4]),
			:vals => CUDA.zeros(Float64, nc * nnz),
			:faceFlux => CuArray([0.5]), :bfaceFlux => CuArray([0.0]),
			:gamma => CuArray([1.0]), :bgamma => CuArray([1.0]),
			:deltaCoeffs => CuArray([1.0]), :bdeltaCoeffs => CuArray([2.0]),
			:magFaceArea => CuArray(ones(nTotal)), :valueFractions => CuArray(ones(nTotal)),
			:refValue => CUDA.zeros(Float64, 3nB), :refGradient => CUDA.zeros(Float64, 3nB),
			:RHS => CUDA.zeros(Float64, 3nCells), :surfaceCells => CuArray(Int32[0]),
			:bValues => CUDA.zeros(Float64, nc * nB), :bRhs => CUDA.zeros(Float64, 3nB),
			:volumes => CuArray(ones(nCells)), :oldVectors => CUDA.zeros(Float64, 3nCells),
			:batchSegments => CuArray(Int32[0, 1]), :batchFaces => CuArray(Int32[0]),
		)
		ptr(k) = Ptr{Cvoid}(UInt(pointer(dev[k])))
		# the matrix and boundary matrix arrive as NeoN hands them over: a VectorType around the pointer
		faceArgs = (nInt, ptr(:owner), ptr(:neighbour), ptr(:diagOffs), ptr(:ownOffs), ptr(:neiOffs),
			ptr(:rowOffs), VectorType{T}(ptr(:vals)), op, ptr(:faceFlux), ptr(:bfaceFlux), ptr(:gamma),
			ptr(:bgamma), ptr(:deltaCoeffs), ptr(:bdeltaCoeffs), ptr(:magFaceArea), ptr(:valueFractions),
			ptr(:refValue), ptr(:refGradient), ptr(:RHS), ptr(:surfaceCells), VectorType{T}(ptr(:bValues)),
			ptr(:bRhs), ptr(:volumes), ptr(:oldVectors), 0.001)
		batchArgs = (Int32(1), ptr(:batchSegments), ptr(:batchFaces))
		GC.@preserve dev begin
			assemble_gpu(faceArgs..., nCells, nTotal)
			globalassemble_gpu(faceArgs..., nCells, nTotal)
			assemble_batched_gpu(faceArgs..., batchArgs..., nCells, nTotal)
			assemble_batched_cuda(faceArgs..., batchArgs..., nCells, nTotal)
			CUDA.synchronize()
		end
	end
	return nothing
end
