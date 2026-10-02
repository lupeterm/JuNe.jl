module JuNe

include("operators.jl")
include("gpu.jl")
using PrecompileTools
using Atomix 
using Polyester

function ddtloop(
    temporals::Union{DiffEq,Ddt},
    volumes::Vector{P},
    oldVectors::Vector{P},
    oldOldVectors::Vector{P},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    RHS::Vector{Float64}
) where {P<:AbstractFloat}
    nCells = Int32(length(oldVectors) / 3)
    for celli in 1:nCells
        idx2D = (celli - 1) * 3 + 1
        diagValue, rx, ry, rz = temporals(
            oldVectors[idx2D:idx2D+2],
            oldOldVectors[idx2D:idx2D+2],
            volumes[celli],
            0.0,
            0.0,
            0.0,
            0.0
        )

        diagIdx = (rowOffs[celli] + diagOffs[celli]) * 3 +1
        vals[diagIdx:diagIdx+2] .+= diagValue
        RHS[idx2D:idx2D+2] += [rx, ry, rz]
    end
end

function ddtloop_threaded(
    temporals::Union{DiffEq,Ddt},
    volumes::Vector{P},
    oldVectors::Vector{P},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    RHS::Vector{Float64}
) where {P<:AbstractFloat}
    @batch for celli in eachindex(volumes)
        diagValue = 0.0
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
        vals[diagIdx:diagIdx+2] .+= diagValue
        RHS[idx2D:idx2D+2] += [rx, ry, rz]
    end
end

function ddtloop(
    temporals::Union{DiffEq,Ddt},
    volumes::Vector{P},
    oldVectors::Vector{P},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    RHS::Vector{Float64}
) where {P<:AbstractFloat}
    nCells = Int32(length(oldVectors) / 3)
    for celli in 1:nCells
        idx2D = (celli - 1) * 3 + 1
        diagValue, rx, ry, rz = temporals(
            oldVectors[idx2D:idx2D+2],
            volumes[celli],
            0.0,
            0.0,
            0.0,
            0.0
        )

        diagIdx = (rowOffs[celli] + diagOffs[celli]) * 3 +1
        vals[diagIdx:diagIdx+2] .+= diagValue
        RHS[idx2D:idx2D+2] += [rx, ry, rz]
    end
end

function cellBased_threaded(
    temporals::Union{DiffEq,Ddt},
    volumes::Vector{P},
    oldVectors::Vector{P},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    RHS::Vector{Float64}
) where {P<:AbstractFloat}
    nCells = Int32(length(oldVectors) / 3)
    @batch for celli in 1:nCells
        idx2D = (celli - 1) * 3 + 1
        valueDiag, rx, ry, rz = temporals(
            oldVectors[idx2D:idx2D+2],
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
end

function cellBased_(
    numCells::Int32,
    owners::Vector{Int32},
    cellFacesSegments::Vector{Int32},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    cellFacesValues::Vector{Int32},
    faceSignV::Vector{Float64},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    matrixColumnIdxV::Vector{Int32},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    dt::Float64
)
    dts = "$dt"
    opstring2 = replace(opString, "DELTAT" => dts)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")
    fused_pde = eval(Meta.parse(opstring3))
    if Threads.nthreads() == 1
        cellBased_serial(
            numCells,
            owners,
            cellFacesSegments,
            diagOffs,
            rowOffs,
            cellFacesValues,
            faceSignV,
            vals,
            fused_pde,
            faceFlux,
            bfaceFlux,
            gamma,
            bgamma,
            deltaCoeffs,
            bdeltaCoeffs,
            matrixColumnIdxV,
            magFaceArea,
            valueFractions,
            surfaceCells,
            bValues,
            bRhs,
            volumes,
            oldVectors,
            refValue,
            refGradient,
            RHS,
            dt
        )
    else
        cellBased_threaded(
            numCells,
            owners,
            cellFacesSegments,
            diagOffs,
            rowOffs,
            cellFacesValues,
            faceSignV,
            vals,
            fused_pde,
            faceFlux,
            bfaceFlux,
            gamma,
            bgamma,
            deltaCoeffs,
            bdeltaCoeffs,
            matrixColumnIdxV,
            magFaceArea,
            valueFractions,
            surfaceCells,
            bValues,
            bRhs,
            volumes,
            oldVectors,
            refValue,
            refGradient,
            RHS,
            dt
        )
    end
end

function cellBased_serial(
    numCells::Int32,
    owners::Vector{Int32},
    cellFacesSegments::Vector{Int32},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    cellFacesValues::Vector{Int32},
    faceSignV::Vector{Float64},
    vals::Vector{Float64},
    fused_pde::PTERM,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    matrixColumnIdxV::Vector{Int32},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    dt::Float64
)
    ddt, spatials = splitTempSpat(fused_pde)
    temporals = !isnothing(ddt) ? ddt : f(args...) = (0.0, 0.0, 0.0, 0.0)
    spatials = !isnothing(spatials) ? spatials : g(args...) = (0.0, 0.0)
    for celli in 1:numCells
        diagValue = 0.0
        numInternalFaces = cellFacesSegments[celli+1] - cellFacesSegments[celli]
        startIdx = cellFacesSegments[celli]
        for i in 1:numInternalFaces
            faceIdx = cellFacesValues[startIdx+i] + 1
            dVal, offDiagValue = spatials(
                faceFlux[faceIdx],
                gamma[faceIdx],
                deltaCoeffs[faceIdx],
                magFaceArea[faceIdx],
                0.0,
                0.0,
                faceSignV[startIdx+i]
            )
            fIdx = matrixColumnIdxV[startIdx+i] *3 +1
            vals[fIdx:fIdx+2] .+= offDiagValue 
            diagValue += dVal
        end
        idx2D = (celli - 1) * 3 + 1
        dv, rx, ry, rz = temporals(
            oldVectors[idx2D:idx2D+2],
            volumes[celli],
            0.0,
            0.0,
            0.0,
            0.0
        )
        diagValue += dv

        diagIdx = (rowOffs[celli] + diagOffs[celli]) * 3 +1
        vals[diagIdx:diagIdx+2] .+= diagValue
        RHS[idx2D:idx2D+2] += [rx, ry, rz]
    end

    faceBasedBoundary(
        Int32(length(gamma)),
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
        bRhs
    )
end


function cellBased_threaded(
    numCells::Int32,
    owners::Vector{Int32},
    cellFacesSegments::Vector{Int32},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    cellFacesValues::Vector{Int32},
    faceSignV::Vector{Float64},
    vals::Vector{Float64},
    fused_pde::PTERM,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    matrixColumnIdxV::Vector{Int32},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    dt::Float64
)
    ddt, spatials = splitTempSpat(fused_pde)
    temporals = !isnothing(ddt) ? ddt : f(args...) = (0.0, 0.0, 0.0, 0.0)
    spatials = !isnothing(spatials) ? spatials : g(args...) = (0.0, 0.0)
    numCells = Int(numCells)
    @batch for celli in 1:numCells
        numInternalFaces = cellFacesSegments[celli+1] - cellFacesSegments[celli]
        diagValue = 0.0
        startIdx = cellFacesSegments[celli]
        for i in 1:numInternalFaces
            faceIdx = cellFacesValues[startIdx+i] + 1
            dVal, offDiagValue = spatials(
                faceFlux[faceIdx],
                gamma[faceIdx],
                deltaCoeffs[faceIdx],
                magFaceArea[faceIdx],
                0.0,
                0.0,
                faceSignV[startIdx+i]
            )
            fIdx = matrixColumnIdxV[startIdx+i] *3 +1
            vals[fIdx] += offDiagValue 
            vals[fIdx+1] += offDiagValue 
            vals[fIdx+2] += offDiagValue 
            diagValue -= dVal
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
        vals[diagIdx:diagIdx+2] .+= diagValue
        RHS[idx2D:idx2D+2] += [rx, ry, rz]
    end

    faceBasedBoundary_threaded(
        Int32(length(gamma)),
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
        bRhs
    )
end



# no temporals
function assemble(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    fused_pde = eval(Meta.parse(opString))
    ddt, spatials = splitTempSpat(fused_pde)
    if !isnothing(ddt)
        println("[ Tried assembly for $(typeof(ddt)), but no oldVectors were passed.")
    end
    if !isnothing(spatials)
        faceBasedAll_(numInteriorFaces,
            owner,
            neighbour,
            diagOffs,
            ownOffs,
            neiOffs,
            rowOffs,
            vals,
            spatials,
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
            bRhs
        )
    end
end

function assemble(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    dt::Float64
)
    dts = "$dt"
    opstring2 = replace(opString, "DELTAT" => dts)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")

    fused_pde = eval(Meta.parse(opstring3))
    ddt, spatials = splitTempSpat(fused_pde)
    if !isnothing(ddt)
        if Threads.nthreads() == 1
            ddtloop(
                ddt,
                volumes,
                oldVectors,
                diagOffs,
                rowOffs,
                vals,
                RHS
            )
        else
            ddtloop_threaded(
                ddt,
                volumes,
                oldVectors,
                diagOffs,
                rowOffs,
                vals,
                RHS
            )
        end
    end
    if !isnothing(spatials)
        if Threads.nthreads() == 1
            faceBasedAll_(
                numInteriorFaces,
                owner,
                neighbour,
                diagOffs,
                ownOffs,
                neiOffs,
                rowOffs,
                vals,
                spatials,
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
                bRhs
            )
        else
            faceBasedAll_threaded(
                numInteriorFaces,
                owner,
                neighbour,
                diagOffs,
                ownOffs,
                neiOffs,
                rowOffs,
                vals,
                spatials,
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
                bRhs
            )
        end
    end
end


function globalassemble(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    dt::Float64
)
    dts = "$dt"
    opstring2 = replace(opString, "DELTAT" => dts)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")

    fused_pde = eval(Meta.parse(opstring3))
    ddt, spatials = splitTempSpat(fused_pde)
    if !isnothing(ddt)
        if Threads.nthreads() == 1
            ddtloop(
                ddt,
                volumes,
                oldVectors,
                diagOffs,
                rowOffs,
                vals,
                RHS
            )
        else
            ddtloop_threaded(
                ddt,
                volumes,
                oldVectors,
                diagOffs,
                rowOffs,
                vals,
                RHS
            )
        end
    end
    if !isnothing(spatials)
        if Threads.nthreads() == 1
            globalfaceBasedAll_(
                numInteriorFaces,
                owner,
                neighbour,
                diagOffs,
                ownOffs,
                neiOffs,
                rowOffs,
                vals,
                spatials,
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
                bRhs
            )
        else
            globalfaceBased_threaded(
                numInteriorFaces,
                owner,
                neighbour,
                diagOffs,
                ownOffs,
                neiOffs,
                rowOffs,
                vals,
                spatials,
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
                bRhs
            )
        end
    end
end

function assemble(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    oldOldVectors::Vector{Float64},
    dt::Float64
)
    dts = "$dt"
    opstring = replace(opString, "DELTAT" => dts)
    fused_pde = eval(Meta.parse(opString))
    ddt, spatials = splitTempSpat(fused_pde)
    if !isnothing(ddt)
        ddtloop(
            temporals,
            volumes,
            oldVectors,
            oldOldVectors,
            diagOffs,
            rowOffs,
            vals,
            RHS
        )
    end
    if !isnothing(spatials)
        faceBasedAll_(
            numInteriorFaces,
            owner,
            neighbour,
            diagOffs,
            ownOffs,
            neiOffs,
            rowOffs,
            vals,
            spatials,
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
            bRhs
        )
    end
end


function faceBasedAll(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    fused_pde = eval(Meta.parse(opString))
    faceBasedAll_(
        numInteriorFaces,
        owner,
        neighbour,
        diagOffs,
        ownOffs,
        neiOffs,
        rowOffs,
        vals,
        fused_pde,
        faceFlux,
        bfaceFlux,
        gamma,
        deltaCoeffs,
        bdeltaCoeffs,
        magFaceArea,
        valueFractions,
        refValue,
        refGradient,
        RHS,
        surfaceCells,
        bValues,
        bRhs
    )
end

function faceBasedAll_(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    fused_pde::PTERM,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    numCells = length(rowOffs) - 1
    for iFace in 1:numInteriorFaces
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
        vals[idx:idx+2] .+= valueUpper
        
        idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
        vals[idx:idx+2] .-= valueUpper

        idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1
        vals[idx:idx+2] .+= valueLower

        idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
        vals[idx:idx+2] .-= valueLower
    end
    for facei in numInteriorFaces+1:numInteriorFaces+length(bfaceFlux)
        bcfacei = facei - numInteriorFaces
        start = bcfacei * 3 - 2
        end_ = start + 2
        valueDiag, valueRHSx, valueRHSy, valueRHSz = fused_pde(
            refValue[start:end_],
            refGradient[start:end_],
            bfaceFlux[bcfacei],
            valueFractions[bcfacei],
            bdeltaCoeffs[bcfacei],
            bgamma[bcfacei],
            magFaceArea[facei],
            0.0, 0.0, 0.0, 0.0
        )
        own = surfaceCells[bcfacei] + 1

        vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1
        vals[vIdx:vIdx+2] .+= valueDiag

        bValues[bcfacei*3-2:bcfacei*3] .+= valueDiag

        # FIXME dont forget, changed this back to [vec3, vec3] instead of [xxxyyyzzz] for now
        RHS[own*3-2:own*3] += [valueRHSx, valueRHSy, valueRHSz]

        bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
    end
end


function faceBasedAll_threaded(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    fused_pde::PTERM,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    numCells = length(rowOffs) - 1
    @batch for iFace in 1:numInteriorFaces
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
        vals[idx:idx+2] .+= valueUpper
        idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
        Atomix.@atomic vals[idx] -= valueUpper
        Atomix.@atomic vals[idx+1] -= valueUpper
        Atomix.@atomic vals[idx+2] -= valueUpper
        idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1
        vals[idx:idx+2] .+= valueLower
        idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
        Atomix.@atomic vals[idx] -= valueLower
        Atomix.@atomic vals[idx+1] -= valueLower
        Atomix.@atomic vals[idx+2] -= valueLower
    end
    @batch for facei in numInteriorFaces+1:numInteriorFaces+length(bfaceFlux)
        bcfacei = facei - numInteriorFaces
        start = bcfacei * 3 - 2
        end_ = start + 2
        valueDiag, valueRHSx, valueRHSy, valueRHSz = fused_pde(
            refValue[start:end_],
            refGradient[start:end_],
            bfaceFlux[bcfacei],
            valueFractions[bcfacei],
            bdeltaCoeffs[bcfacei],
            bgamma[bcfacei],
            magFaceArea[facei],
            0.0, 0.0, 0.0, 0.0
        )
        own = surfaceCells[bcfacei] + 1

        vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1
        Atomix.@atomic vals[vIdx] += valueDiag
        Atomix.@atomic vals[vIdx+1] += valueDiag
        Atomix.@atomic vals[vIdx+2] += valueDiag

        bValues[bcfacei*3-2:bcfacei*3] .+= valueDiag

        # FIXME dont forget, changed this back to [vec3, vec3] instead of [xxxyyyzzz] for now
        Atomix.@atomic RHS[own*3-2] += valueRHSx
        Atomix.@atomic RHS[own*3-1] += valueRHSy
        Atomix.@atomic RHS[own*3] += valueRHSz

        bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
    end
end


function globalfaceBasedAll_(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    fused_pde::PTERM,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    numCells = length(rowOffs) - 1
    numTotalFaces = numInteriorFaces + length(bfaceFlux)
    for iFace in 1:numTotalFaces
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
            vals[idx:idx+2] .+= valueUpper
            
            idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
            vals[idx:idx+2] .-= valueUpper

            idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1
            vals[idx:idx+2] .+= valueLower

            idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
            vals[idx:idx+2] .-= valueLower
        else
            bcfacei = iFace - numInteriorFaces
            start = bcfacei * 3 - 2
            end_ = start + 2
            valueDiag, valueRHSx, valueRHSy, valueRHSz = fused_pde(
                refValue[start:end_],
                refGradient[start:end_],
                bfaceFlux[bcfacei],
                valueFractions[bcfacei],
                bdeltaCoeffs[bcfacei],
                bgamma[bcfacei],
                magFaceArea[iFace],
                0.0, 0.0, 0.0, 0.0
            )
            own = surfaceCells[bcfacei] + 1

            vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1
            vals[vIdx:vIdx+2] .+= valueDiag

            bValues[bcfacei*3-2:bcfacei*3] .+= valueDiag

            # FIXME dont forget, changed this back to [vec3, vec3] instead of [xxxyyyzzz] for now
            RHS[own*3-2:own*3] += [valueRHSx, valueRHSy, valueRHSz]

            bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
        end
    end
end


function globalfaceBased_threaded(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    fused_pde::PTERM,
    faceFlux::Vector{Float64},
    bfaceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bgamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    numCells = length(rowOffs) - 1
    numTotalFaces = numInteriorFaces + length(bfaceFlux)
    @batch for iFace in 1:numTotalFaces
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
        vals[idx:idx+2] .+= valueUpper
        idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
        Atomix.@atomic vals[idx] -= valueUpper
        Atomix.@atomic vals[idx+1] -= valueUpper
        Atomix.@atomic vals[idx+2] -= valueUpper
        idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1
        vals[idx:idx+2] .+= valueLower
        idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
        Atomix.@atomic vals[idx] -= valueLower
        Atomix.@atomic vals[idx+1] -= valueLower
        Atomix.@atomic vals[idx+2] -= valueLower
        else
            bcfacei = iFace - numInteriorFaces
            start = bcfacei * 3 - 2
            end_ = start + 2
            valueDiag, valueRHSx, valueRHSy, valueRHSz = fused_pde(
                refValue[start:end_],
                refGradient[start:end_],
                bfaceFlux[bcfacei],
                valueFractions[bcfacei],
                bdeltaCoeffs[bcfacei],
                bgamma[bcfacei],
                magFaceArea[iFace],
                0.0, 0.0, 0.0, 0.0
            )
            own = surfaceCells[bcfacei] + 1

            vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1
            Atomix.@atomic vals[vIdx] += valueDiag
            Atomix.@atomic vals[vIdx+1] += valueDiag
            Atomix.@atomic vals[vIdx+2] += valueDiag

            bValues[bcfacei*3-2:bcfacei*3] .+= valueDiag

            # FIXME dont forget, changed this back to [vec3, vec3] instead of [xxxyyyzzz] for now
            Atomix.@atomic RHS[own*3-2] += valueRHSx
            Atomix.@atomic RHS[own*3-1] += valueRHSy
            Atomix.@atomic RHS[own*3] += valueRHSz

            bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
        end
    end
end


function faceBased(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    fused_pde = eval(Meta.parse(opString))
    for iFace in 1:numInteriorFaces
        iOwner = owner[iFace] + 1
        iNeighbor = neighbour[iFace] + 1

        rowNeiStart = rowOffs[iNeighbor] + 1
        rowOwnStart = rowOffs[iOwner] + 1

        valueUpper, valueLower = fused_pde(
            faceFlux[iFace], 
            gamma[iFace], 
            deltaCoeffs[iFace], 
            magFaceArea[iFace],
            0.0, 
            0.0
        )        
        vals[rowNeiStart+neiOffs[iFace]] += valueUpper
        vals[rowOwnStart+diagOffs[iOwner]] -= valueUpper
        vals[rowOwnStart+ownOffs[iFace]] += valueLower
        vals[rowNeiStart+diagOffs[iNeighbor]] -= valueLower
    end
end

function faceBasedBoundary(
    numInteriorFaces::Int32,
    surfaceCells::Vector{Int32},  # surfaceCells 
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient_::Vector{Float64},
    RHS::Vector{Float64},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    fused_pde = eval(Meta.parse(opString))
    ddt, spatials = splitTempSpat(fused_pde)

    faceBasedBoundary(
        numInteriorFaces,
        surfaceCells,
        diagOffs,
        rowOffs,
        vals,
        spatials,
        faceFlux,
        gamma,
        bdeltaCoeffs,
        magFaceArea,
        valueFractions,
        refValue,
        refGradient_,
        RHS,
        bValues,
        bRhs
    )
end

function faceBasedBoundary(
    numInteriorFaces::Int32,
    surfaceCells::Vector{Int32},  # surfaceCells 
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    spatialOperators::PTERM,
    bfaceFlux::Vector{Float64},
    bgamma::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    for facei in numInteriorFaces+1:numInteriorFaces+length(bfaceFlux)
        bcfacei = facei - numInteriorFaces
        start = bcfacei * 3 - 2
        end_ = start + 2
        valueDiag, valueRHSx, valueRHSy, valueRHSz = spatialOperators(
            refValue[start:end_],
            refGradient[start:end_],
            bfaceFlux[bcfacei],
            valueFractions[bcfacei],
            bdeltaCoeffs[bcfacei],
            bgamma[bcfacei],
            magFaceArea[facei],
            0.0, 0.0, 0.0, 0.0
        )
        own = surfaceCells[bcfacei] + 1

        vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1
        vals[vIdx:vIdx+2] .+= valueDiag

        bValues[bcfacei*3-2:bcfacei*3] .+= valueDiag

        # FIXME dont forget, changed this back to [vec3, vec3] instead of [xxxyyyzzz] for now
        RHS[own*3-2:own*3] += [valueRHSx, valueRHSy, valueRHSz]

        bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
    end
end


function faceBasedBoundary_threaded(
    numInteriorFaces::Int32,
    surfaceCells::Vector{Int32},  # surfaceCells 
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    spatialOperators::PTERM,
    bfaceFlux::Vector{Float64},
    bgamma::Vector{Float64},
    bdeltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    @batch for facei in numInteriorFaces+1:numInteriorFaces+length(bfaceFlux)
        bcfacei = facei - numInteriorFaces
        start = bcfacei * 3 - 2
        end_ = start + 2
        valueDiag, valueRHSx, valueRHSy, valueRHSz = spatialOperators(
            refValue[start:end_],
            refGradient[start:end_],
            bfaceFlux[bcfacei],
            valueFractions[bcfacei],
            bdeltaCoeffs[bcfacei],
            bgamma[bcfacei],
            magFaceArea[facei],
            0.0, 0.0, 0.0, 0.0
        )
        own = surfaceCells[bcfacei] + 1

        vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1
        Atomix.@atomic vals[vIdx] += valueDiag
        Atomix.@atomic vals[vIdx+1] += valueDiag
        Atomix.@atomic vals[vIdx+2] += valueDiag

        bValues[bcfacei*3-2:bcfacei*3] .+= valueDiag

        # FIXME dont forget, changed this back to [vec3, vec3] instead of [xxxyyyzzz] for now
        Atomix.@atomic RHS[own*3-2] += valueRHSx
        Atomix.@atomic RHS[own*3-1] += valueRHSy
        Atomix.@atomic RHS[own*3] += valueRHSz

        bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
    end
end



function warmup(op::String)
    assemble(
        Int32(1),
        ones(Int32, 2),
        ones(Int32, 2),
        ones(UInt8, 2),
        ones(UInt8, 2),
        ones(UInt8, 2),
        ones(Int32, 2),
        zeros(Float64, 12),
        op,
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 3),
        ones(Float64, 3),
        zeros(Float64, 12),
        zeros(Int32, 2),
        zeros(Float64, 12),
        zeros(Float64, 12),
        zeros(Float64, 2),
        zeros(Float64, 12),
        0.001,
    )
end


export faceBased, Div, Noop, linear, upwind, BDF1, Laplace, test, faceBasedBoundary, faceBasedAll, Ddt, hasTransient, splitTempSpat, BDF2, DELTAT, assemble_gpu

end # module JuNe
