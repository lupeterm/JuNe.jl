include("schemes.jl")
using StaticArrays
#### Transient Term 
# ddt(ϕ)

struct Ddt{P,S}
    scheme::S
    scale::P
end

# cellbased
function (d::Ddt{P,S})(
    oldVector::Vector{P},
    oldOldVector::Vector{P},
    volume::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat,S}
    ret = d.scale * d.scheme(volume, oldVector, oldOldVector)
    return valueDiag + ret[1], valueRHSx +ret[2][1], valueRHSy +ret[2][2], valueRHSz + ret[2][3]
end
# cellbased inner 
function (d::Ddt{P,BDF1{P}})(
    oldVector::Vector{P},
    volume::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat}
    ret = d.scale * d.scheme(volume, oldVector)
    return valueDiag + ret[1], valueRHSx +ret[2][1], valueRHSy +ret[2][2], valueRHSz + ret[2][3]
end

# gpu version BDF1
function (d::Ddt{P,BDF1{P}})(
    oldVectorx::P,
    oldVectory::P,
    oldVectorz::P,
    volume::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat}
    r1,r2,r3,r4 = d.scale * d.scheme(volume, oldVectorx, oldVectory, oldVectorz)
    return valueDiag + r1, valueRHSx +r2, valueRHSy +r3, valueRHSz + r4
end

# gpu version BDF2
function (d::Ddt{P,BDF1{P}})(
    oldVectorx::P,
    oldVectory::P,
    oldVectorz::P,
    oldOldVectorx::P,
    oldOldVectory::P,
    oldOldVectorz::P,
    volume::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat}
    r1,r2,r3,r4 = d.scale * d.scheme(volume, oldVectorx, oldVectory, oldVectorz, oldOldVectorx, oldOldVectory, oldOldVectorz)
    return valueDiag + r1, valueRHSx +r2, valueRHSy +r3, valueRHSz + r4
end

#### Convection
# ∇(ϕ,U) (in momentum equation)

struct Div{P,S}
    scheme::S
    scale::P
end

# cellbased inner
function (d::Div{P,S})(
    faceFlux::P,
    _::P,
    _::P,
    _::P,
    valueUpper::P,
    valueLower::P,
    sign::P
) where {P<:AbstractFloat,S}
    weights_f = d.scheme(faceFlux)
    if sign > 0
        offdiag = faceFlux * (1.0 - weights_f) * d.scale
        diagContrib = faceFlux * weights_f * d.scale
    else 
        offdiag = -faceFlux * weights_f * d.scale
        diagContrib = -faceFlux * (1.0 - weights_f) * d.scale
    end
    return valueUpper + diagContrib, valueLower + offdiag
end

# facebased inner
function (d::Div{P,S})(
    faceFlux::P,
    _::P,
    _::P,
    _::P,
    valueUpper::P,
    valueLower::P
) where {P<:AbstractFloat,S}
    weights_f = d.scheme(faceFlux)
    ownFluxContrib = -faceFlux * weights_f * d.scale
    neiFluxContrib = +faceFlux * (1.0 - weights_f) * d.scale
    return valueUpper + ownFluxContrib, valueLower + neiFluxContrib
end


# facebased boundary 
function (d::Div{P,S})(
    refValue::AbstractVector{P},
    refGradient::AbstractVector{P},
    bFaceFlux::P,
    valueFraction::P,
    bdeltaCoeff::P,
    _::P,
    _::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat,S}
    refGradFrac = 1.0 - valueFraction
    flux = bFaceFlux * refGradFrac * d.scale * -1.0
	valueRHSx -= bFaceFlux * valueFraction * refValue[1] + refGradFrac * refGradient[1] / bdeltaCoeff
	valueRHSy -= bFaceFlux * valueFraction * refValue[2] + refGradFrac * refGradient[2] / bdeltaCoeff
	valueRHSz -= bFaceFlux * valueFraction * refValue[3] + refGradFrac * refGradient[3] / bdeltaCoeff
    return valueDiag - flux, valueRHSx, valueRHSy, valueRHSz 
end

# facebased boundary 
function (d::Div{P,S})(
    refValuex::P,
    refValuey::P,
    refValuez::P,
    refGradientx::P,
    refGradienty::P,
    refGradientz::P,
    bFaceFlux::P,
    valueFraction::P,
    bdeltaCoeff::P,
    _::P,
    _::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat,S}
    refGradFrac = 1.0 - valueFraction
    flux = bFaceFlux * refGradFrac * d.scale * -1.0
	valueRHSx -= bFaceFlux * valueFraction * refValuex + refGradFrac * refGradientx / bdeltaCoeff
	valueRHSx -= bFaceFlux * valueFraction * refValuey + refGradFrac * refGradienty / bdeltaCoeff
	valueRHSx -= bFaceFlux * valueFraction * refValuez + refGradFrac * refGradientz / bdeltaCoeff
    return valueDiag - flux, valueRHSx* d.scale, valueRHSy* d.scale, valueRHSz * d.scale
end

#### Diffusion
# laplacian(ν, ϕ)

struct Laplace{P}
    scale::P
end

#cellbased
function (t::Laplace{P})(
    _::P,
    gamma::P,
    deltaCoeff::P,
    magFaceArea::P,
    valueUpper::P,
    valueLower::P,
    _::P
) where {P<:AbstractFloat}
    flux = gamma * deltaCoeff * magFaceArea * t.scale
    return valueUpper - flux, valueLower + flux
end

#facebased
function (t::Laplace{P})(
    _::P,
    gamma::P,
    deltaCoeff::P,
    magFaceArea::P,
    valueUpper::P,
    valueLower::P
) where {P<:AbstractFloat}
    flux = gamma * deltaCoeff * magFaceArea * t.scale
    return valueUpper + flux, valueLower + flux
end

# facebased boundary 
function (d::Laplace{P})(
    refValue::AbstractVector{P},
    refGradient::AbstractVector{P},
    _::P,
    refValFrac::P,
    deltaCoeff::P,
    gamma::P,
    magFaceArea::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat}
    flux = gamma * magFaceArea
    fluxContrib = flux * d.scale * refValFrac * deltaCoeff 
    refGradFrac = 1-refValFrac
    x = flux * d.scale * (refValFrac * deltaCoeff * refValue[1] + refGradFrac * refGradient[1]);
    y = flux * d.scale * (refValFrac * deltaCoeff * refValue[2] + refGradFrac * refGradient[2]);
    z = flux * d.scale * (refValFrac * deltaCoeff * refValue[3] + refGradFrac * refGradient[3]);
    return valueDiag - fluxContrib, valueRHSx - x, valueRHSy - y, valueRHSz - z
end

# GPU facebased boundary 
function (d::Laplace{P})(
    refValuex::P,
    refValuey::P,
    refValuez::P,
    refGradientx::P,
    refGradienty::P,
    refGradientz::P,
    _::P,
    refValFrac::P,
    deltaCoeff::P,
    gamma::P,
    magFaceArea::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat}
    flux = gamma * magFaceArea
    fluxContrib = flux * d.scale * refValFrac * deltaCoeff 
    refGradFrac = 1-refValFrac
    x = flux * d.scale * (refValFrac * deltaCoeff * refValuex + refGradFrac * refGradientx);
    y = flux * d.scale * (refValFrac * deltaCoeff * refValuey + refGradFrac * refGradienty);
    z = flux * d.scale * (refValFrac * deltaCoeff * refValuez + refGradFrac * refGradientz);
    return valueDiag - fluxContrib, valueRHSx - x, valueRHSy - y, valueRHSz - z
end

#### Operator Fusing
struct DiffEq{A,B}
    a::A
    b::B
end

## cellbased inner
@inline function (o::DiffEq)(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower, sign)
    valueUpper, valueLower = o.a(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower, sign)
    valueUpper, valueLower = o.b(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower, sign)
    return valueUpper, valueLower
end


## facebased inner
@inline function (o::DiffEq)(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
    valueUpper, valueLower = o.a(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
    valueUpper, valueLower = o.b(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
    return valueUpper, valueLower
end


# facebased boundary 
@inline function (o::DiffEq)(
    refValue,
    refGradient,
    flux,
    valueFraction,
    deltaCoeff,
    gamma,
    magFaceArea,
    valueDiag,
    valueRHSx,
    valueRHSy,
    valueRHSz
)
    valueDiag, valueRHSx, valueRHSy, valueRHSz = o.a(refValue, refGradient, flux, valueFraction, deltaCoeff, gamma, magFaceArea, valueDiag, valueRHSx, valueRHSy, valueRHSz)
    valueDiag, valueRHSx, valueRHSy, valueRHSz = o.b(refValue, refGradient, flux, valueFraction, deltaCoeff, gamma, magFaceArea, valueDiag, valueRHSx, valueRHSy, valueRHSz)
    return valueDiag, valueRHSx, valueRHSy, valueRHSz
end

# gpu
@inline function (o::DiffEq)(
    refValuex,refValuey,refValuez,
    refGradientx,refGradienty,refGradientz,
    flux,
    valueFraction,
    deltaCoeff,
    gamma,
    magFaceArea,
    valueDiag,
    valueRHSx, valueRHSy, valueRHSz
)
    valueDiag, valueRHSx, valueRHSy, valueRHSz = o.a(refValuex,refValuey,refValuez, refGradientx, refGradienty, refGradientz, flux, valueFraction, deltaCoeff, gamma, magFaceArea, valueDiag, valueRHSx, valueRHSy, valueRHSz)
    valueDiag, valueRHSx, valueRHSy, valueRHSz = o.b(refValuex,refValuey,refValuez, refGradientx, refGradienty, refGradientz, flux, valueFraction, deltaCoeff, gamma, magFaceArea, valueDiag, valueRHSx, valueRHSy, valueRHSz)
    return valueDiag, valueRHSx, valueRHSy, valueRHSz
end

PTERM = Union{DiffEq,Ddt,Div,Laplace}

Base.:+(a::PTERM, b::PTERM) = DiffEq(a, b)
Base.:+(a::PTERM, _::Nothing) = a
Base.:+(_::Nothing, a::PTERM) = a
Base.:+(_::Nothing, _::Nothing) = nothing
Base.:+(a::Tuple, b::Tuple) = a[1] + b[1], a[2] + b[2]
Base.:*(a::P, b::Tuple{P}) where {P<:AbstractFloat} = a * b[1], a * b[2]
Base.:*(a::P, b::Tuple{P, P, P, P}) where {P<:AbstractFloat} = a * b[1], a * b[2], a*b[3], a*b[4]
Base.:*(a::P, b::Tuple{P, Vector{P}}) where {P<:AbstractFloat} = a * b[1], a .* b[2]

hasTransient(t::DiffEq) = hasTransient(t.a) || hasTransient(t.b)
hasTransient(t::Ddt) = true
hasTransient(t::Union{Div,Laplace}) = false

splitTempSpat(t::DiffEq) = splitTempSpat(t.a) + splitTempSpat(t.b)
splitTempSpat(t::Ddt) = t, nothing
splitTempSpat(t::Union{Div,Laplace}) = nothing, t

# Base.show(io::IO, t::DiffEq) = print(io, print(t.a), print(t.b)) 
# Base.show(io::IO, t::Div) = print(io, "$(t.scale) * Div{P, $(t.scheme)}" ) 
# Base.show(io::IO, t::Ddt) = print(io, "$(t.scale) * Ddt{P, $(t.scheme)}" ) 
# Base.show(io::IO, t::Laplace) = print(io, "+ $(t.scale) * Laplace{P}" ) 
