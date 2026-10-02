#### DIV SCHEMES
abstract type DivScheme{P} end

struct Noop{P} <: DivScheme{P} end

struct upwind{P} <: DivScheme{P} end
@inline (s::upwind{P})(ϕ) where {P<:AbstractFloat} = ifelse(ϕ ≥ 0, one(P), zero(P))

struct linear{P} <: DivScheme{P} end
@inline (u::linear{P})(ϕf) where {P<:AbstractFloat} = P(0.5)

#### DDT SCHEMES
abstract type DdtScheme{P} end

struct BDF1{P} <: DdtScheme{P}
    a0a1::P
end

function BDF1(Δt::P) where {P<:AbstractFloat}
    return BDF1{P}(1 / Δt)
end

#cpu
@inline function (s::BDF1{P})(volume::P, oldVector::Vector{P}) where {P<:AbstractFloat}
    common = volume * s.a0a1
    return common, common .* oldVector # diag, rhs 
end

# gpu
@inline function (s::BDF1{P})(
    volume::P,     
    oldVectorx::P,
    oldVectory::P,
    oldVectorz::P
) where {P<:AbstractFloat}
    common = volume * s.a0a1
    return common, oldVectorx * common, oldVectory * common, oldVectorz * common
end

struct BDF2{P} <: DdtScheme{P}
    a0::P
    a1::P
    a2::P
end
DELTAT = nothing

#cpu
@inline function (s::BDF2)(volume::P, oldVector::Vector{P}, oldOldVector::Vector{P}) where {P<:AbstractFloat}
    diag = volume * s.a0
    rhs = volume * s.a1 .* oldVector + volume * s.a2 .* oldOldVector
    return diag, rhs
end

#gpu
@inline function (s::BDF2)(
    volume::P, 
    oldVectorx::P, 
    oldVectory::P, 
    oldVectorz::P, 
    oldOldVectorx::P,
    oldOldVectory::P,
    oldOldVectorz::P
) where {P<:AbstractFloat}
    diag = volume * s.a0
    rhsx = volume * s.a1 .* oldVectorx + volume * s.a2 .* oldOldVectorx
    rhsy = volume * s.a1 .* oldVectory + volume * s.a2 .* oldOldVectory
    rhsz = volume * s.a1 .* oldVectorz + volume * s.a2 .* oldOldVectorz
    return diag, rhsx, rhsy, rhsz
end

function BDF2(Δt::P)::BDF2{P} where {P<:AbstractFloat}
    return BDF2{P}(1.5 / Δt, 2.0 / Δt, -0.5 / Δt)
end


function BDF2(d::Nothing)::BDF2{Float64}
    return BDF2{Float64}(1.0, 1.0, 1.0)
end
function BDF1(d::Nothing)::BDF1{Float64}
    return BDF1{Float64}(1.0)
end
