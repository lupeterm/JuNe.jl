using CUDA
using KernelAbstractions
using Atomix
using StaticArrays

function ptrToGpu(ptr::Ptr{Cvoid}, size, dType)
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
	_vals::Ptr{Cvoid},
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
	_bValues::Ptr{Cvoid},
	_bRhs::Ptr{Cvoid},
	_volumes::Ptr{Cvoid},
	_oldVectors::Ptr{Cvoid},
	_dt::Float64,
	numCells::Int32,
	numTotalFaces::Int32
)
	boundaryFaces = numTotalFaces - numInteriorFaces
	owner = ptrToGpu(_owner, numInteriorFaces, Int32)	
	neighbour = ptrToGpu(_neighbour, numInteriorFaces, Int32)
	diagOffs = ptrToGpu(_diagOffs, numCells, UInt8)
	ownOffs = ptrToGpu(_ownOffs, numInteriorFaces, UInt8)
	neiOffs = ptrToGpu(_neiOffs, numInteriorFaces, UInt8)
	rowOffs = ptrToGpu(_rowOffs, numCells, Int32)
	vals = ptrToGpu(_vals, (numCells + 2*numInteriorFaces)*3, Float64)
	faceFlux = ptrToGpu(_faceFlux, numInteriorFaces, Float64)
	bfaceFlux = ptrToGpu(_bfaceFlux, boundaryFaces, Float64)
	gamma = ptrToGpu(_gamma, numInteriorFaces, Float64)
	bgamma = ptrToGpu(_bgamma, boundaryFaces, Float64)
	deltaCoeffs = ptrToGpu(_deltaCoeffs, numInteriorFaces, Float64)
	bdeltaCoeffs = ptrToGpu(_bdeltaCoeffs, boundaryFaces, Float64)
	magFaceArea = ptrToGpu(_magFaceArea, numTotalFaces, Float64)
	valueFractions = ptrToGpu(_valueFractions, numTotalFaces, Float64)
	refValue = ptrToGpu(_refValue, boundaryFaces*3, Float64)
	refGradient = ptrToGpu(_refGradient, boundaryFaces*3, Float64)
	RHS = ptrToGpu(_RHS, numCells*3, Float64)
	surfaceCells = ptrToGpu(_surfaceCells, boundaryFaces, Int32)
	bValues= ptrToGpu(_bValues, boundaryFaces*3, Float64)
	bRhs= ptrToGpu(_bRhs, boundaryFaces*3, Float64)
	volumes= ptrToGpu(_volumes, numCells, Float64)
	oldVectors= ptrToGpu(_oldVectors, numCells*3, Float64)
	
	dt = "$_dt"
    opstring2 = replace(opString, "DELTAT" => dt)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")
    fused_pde = eval(Meta.parse(opstring3))

    ddt, spatials = JuNe.splitTempSpat(fused_pde)
	backend = get_backend(vals)
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
	celli = @index(Global)
	idx2D = (celli - 1) * 3 + 1
	valueDiag, rx, ry, rz = temporals(
		oldVectors[idx2D],
		oldVectors[idx2D+1],
		oldVectors[idx2D+2],
		volumes[celli],
		0.0,
		0.0,
		0.0,
		0.0
	)
	idx = (rowOffs[celli] + diagOffs[celli]) * 3 + 1
	vals[idx] += valueDiag
	vals[idx+1] += valueDiag
	vals[idx+2] += valueDiag
	RHS[idx2D] += rx
	RHS[idx2D+1] += rz
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
            0.0, 
            0.0
        )

		idx = (rowNeiStart + neiOffs[iFace]) * 3 + 1
        vals[idx]   += valueUpper
        vals[idx+1] += valueUpper
        vals[idx+2] += valueUpper
        
		idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
        Atomix.@atomic vals[idx] -= valueUpper
        Atomix.@atomic vals[idx+1] -= valueUpper
        Atomix.@atomic vals[idx+2] -= valueUpper
        
		idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1        
		vals[idx]   += valueLower
        vals[idx+1] += valueLower
        vals[idx+2] += valueLower
        
		idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
        # FIXME when going back to scalar values

        Atomix.@atomic vals[idx]   -= valueLower
        Atomix.@atomic vals[idx+1] -= valueLower
        Atomix.@atomic vals[idx+2] -= valueLower
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
			0.0, 0.0, 0.0, 0.0
        )


		own = surfaceCells[bcfacei] + 1
		vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1

        bValues[bcfacei*3-2] += fluxContrib
        bValues[bcfacei*3-1] += fluxContrib
        bValues[bcfacei*3] 	 += fluxContrib

		Atomix.@atomic vals[vIdx]   += fluxContrib
        Atomix.@atomic vals[vIdx+1] += fluxContrib
        Atomix.@atomic vals[vIdx+2] += fluxContrib
		
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
    vals_::Ptr{Cvoid},
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
    bValues_::Ptr{Cvoid},
    bRhs_::Ptr{Cvoid},
    volumes_::Ptr{Cvoid},
    oldVectors_::Ptr{Cvoid},
    refValue_::Ptr{Cvoid},
    refGradient_::Ptr{Cvoid},
    RHS_::Ptr{Cvoid},
    _dt::Float64,
    numInteriorFaces::Int32,
	numTotalFaces::Int32
)
	numBoundaryFaces = numTotalFaces - numInteriorFaces
	cellFacesSegments = ptrToGpu(cellFacesSegments_, numCells+1, Int32)
	diagOffs = ptrToGpu(diagOffs_, numCells, UInt8)
	rowOffs = ptrToGpu(rowOffs_, numCells, Int32)
	cellFacesValues = ptrToGpu(cellFacesValues_, (numCells + 2*numInteriorFaces), Int32)
	faceSignV = ptrToGpu(faceSignV_, (numCells + 2*numInteriorFaces), Float64)
	vals = ptrToGpu(vals_, (numCells + 2*numInteriorFaces)*3, Float64)
	faceFlux = ptrToGpu(faceFlux_, numInteriorFaces, Float64)
	bfaceFlux = ptrToGpu(bfaceFlux_, numBoundaryFaces, Float64)
	gamma = ptrToGpu(gamma_, numInteriorFaces, Float64)
	bgamma = ptrToGpu(bgamma_, numBoundaryFaces, Float64)
	deltaCoeffs = ptrToGpu(deltaCoeffs_, numInteriorFaces, Float64)
	bdeltaCoeffs = ptrToGpu(bdeltaCoeffs_, numBoundaryFaces, Float64)
	matrixColumnIdxV = ptrToGpu(matrixColumnIdxV_, (numCells + 2*numInteriorFaces), Int32)
	magFaceArea = ptrToGpu(magFaceArea_, numTotalFaces, Float64)
	valueFractions = ptrToGpu(valueFractions_, numTotalFaces, Float64)
	surfaceCells = ptrToGpu(surfaceCells_, numBoundaryFaces, Int32)
	bValues = ptrToGpu(bValues_, numBoundaryFaces *3, Float64)
	bRhs = ptrToGpu(bRhs_, numBoundaryFaces*3, Float64)
	volumes = ptrToGpu(volumes_, numCells, Float64)
	oldVectors = ptrToGpu(oldVectors_, numCells*3, Float64)
	refValue = ptrToGpu(refValue_, numBoundaryFaces*3, Float64)
	refGradient = ptrToGpu(refGradient_, numBoundaryFaces*3, Float64)
	RHS = ptrToGpu(RHS_, numCells*3, Float64)

	dt = "$_dt"
    opstring2 = replace(opString_, "DELTAT" => dt)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")
    fused_pde = eval(Meta.parse(opstring3))

    ddt, spatials = JuNe.splitTempSpat(fused_pde)
	backend = get_backend(vals)
	temporals = !isnothing(ddt) ? ddt : f(args...) = (0.0, 0.0, 0.0, 0.0)
    spatials = !isnothing(spatials) ? spatials : g(args...) = (0.0, 0.0)
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
	celli = @index(Global)
	numInternalFaces = cellFacesSegments[celli+1] - cellFacesSegments[celli]

	diagValue = 0.0
	startIdx = cellFacesSegments[celli]
	for i in 1:numInternalFaces
		faceIdx = cellFacesValues[startIdx+i] + 1
		diagValue, offDiagValue = spatials(
			faceFlux[faceIdx],
			gamma[faceIdx],
			deltaCoeffs[faceIdx],
			magFaceArea[faceIdx],
			diagValue,
			0.0,
			faceSignV[startIdx+i]
		)
		fIdx = matrixColumnIdxV[startIdx+i] *3 +1
		vals[fIdx] += offDiagValue 
		vals[fIdx+1] += offDiagValue 
		vals[fIdx+2] += offDiagValue 
		# diagValue -= dVal
	end
	idx2D = celli * 3 -2
	dv, rx, ry, rz = temporals(
		oldVectors[idx2D],
		oldVectors[idx2D+1],
		oldVectors[idx2D+2],
		volumes[celli],
		0.0,
		0.0,
		0.0,
		0.0
	)
	diagValue += dv

	diagIdx = (rowOffs[celli] + diagOffs[celli]) * 3 +1
	vals[diagIdx]   += diagValue
	vals[diagIdx+1] += diagValue
	vals[diagIdx+2] += diagValue
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
		0.0, 0.0, 0.0, 0.0
	)
	
	own = surfaceCells[bcfacei] + 1
	vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1

	bValues[bcfacei*3-2] += valueDiag
	bValues[bcfacei*3-1] += valueDiag
	bValues[bcfacei*3] 	 += valueDiag

	Atomix.@atomic vals[vIdx]   += valueDiag
	Atomix.@atomic vals[vIdx+1] += valueDiag
	Atomix.@atomic vals[vIdx+2] += valueDiag

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
	_vals::Ptr{Cvoid},
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
	_bValues::Ptr{Cvoid},
	_bRhs::Ptr{Cvoid},
	_volumes::Ptr{Cvoid},
	_oldVectors::Ptr{Cvoid},
	_dt::Float64,
	numCells::Int32,
	numTotalFaces::Int32
)
	boundaryFaces = numTotalFaces - numInteriorFaces
	owner = ptrToGpu(_owner, numInteriorFaces, Int32)	
	neighbour = ptrToGpu(_neighbour, numInteriorFaces, Int32)
	diagOffs = ptrToGpu(_diagOffs, numCells, UInt8)
	ownOffs = ptrToGpu(_ownOffs, numInteriorFaces, UInt8)
	neiOffs = ptrToGpu(_neiOffs, numInteriorFaces, UInt8)
	rowOffs = ptrToGpu(_rowOffs, numCells, Int32)
	vals = ptrToGpu(_vals, (numCells + 2*numInteriorFaces)*3, Float64)
	faceFlux = ptrToGpu(_faceFlux, numInteriorFaces, Float64)
	bfaceFlux = ptrToGpu(_bfaceFlux, boundaryFaces, Float64)
	gamma = ptrToGpu(_gamma, numInteriorFaces, Float64)
	bgamma = ptrToGpu(_bgamma, boundaryFaces, Float64)
	deltaCoeffs = ptrToGpu(_deltaCoeffs, numInteriorFaces, Float64)
	bdeltaCoeffs = ptrToGpu(_bdeltaCoeffs, boundaryFaces, Float64)
	magFaceArea = ptrToGpu(_magFaceArea, numTotalFaces, Float64)
	valueFractions = ptrToGpu(_valueFractions, numTotalFaces, Float64)
	refValue = ptrToGpu(_refValue, boundaryFaces*3, Float64)
	refGradient = ptrToGpu(_refGradient, boundaryFaces*3, Float64)
	RHS = ptrToGpu(_RHS, numCells*3, Float64)
	surfaceCells = ptrToGpu(_surfaceCells, boundaryFaces, Int32)
	bValues= ptrToGpu(_bValues, boundaryFaces*3, Float64)
	bRhs= ptrToGpu(_bRhs, boundaryFaces*3, Float64)
	volumes= ptrToGpu(_volumes, numCells, Float64)
	oldVectors= ptrToGpu(_oldVectors, numCells*3, Float64)
	dt = "$_dt"
    opstring2 = replace(opString, "DELTAT" => dt)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")
    fused_pde = eval(Meta.parse(opstring3))

    ddt, spatials = JuNe.splitTempSpat(fused_pde)
	backend = get_backend(vals)
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
		0.0, 
		0.0
	)

	idx = (rowNeiStart + neiOffs[iFace]) * 3 + 1
	vals[idx]   += valueUpper
	vals[idx+1] += valueUpper
	vals[idx+2] += valueUpper
	
	idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
	Atomix.@atomic vals[idx] -= valueUpper
	Atomix.@atomic vals[idx+1] -= valueUpper
	Atomix.@atomic vals[idx+2] -= valueUpper
	
	idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1        
	vals[idx]   += valueLower
	vals[idx+1] += valueLower
	vals[idx+2] += valueLower
	
	idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
	# FIXME when going back to scalar values

	Atomix.@atomic vals[idx]   -= valueLower
	Atomix.@atomic vals[idx+1] -= valueLower
	Atomix.@atomic vals[idx+2] -= valueLower
end