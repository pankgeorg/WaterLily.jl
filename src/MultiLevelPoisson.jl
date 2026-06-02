@inline up(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))
@inline down(I::CartesianIndex) = CI((I+2oneunit(I)).I .÷2)
@fastmath @inline function restrict(I::CartesianIndex,b)
    s = zero(eltype(b))
    for J ∈ up(I)
     s += @inbounds(b[J])
    end
    return s
end
@fastmath @inline function restrictL(I::CartesianIndex,i,b)
    s = zero(eltype(b))
    for J ∈ up(I,i)
     s += @inbounds(b[J,i])
    end
    return 0.5s
end

function restrictML(b::Poisson)
    N,n = size_u(b.L)
    Na = map(i->1+i÷2,N)
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    restrictL!(aL,b.L,perdir=b.perdir)
    Poisson(ax,aL,copy(ax);b.perdir)
end
function restrictL!(a::AbstractArray{T,M},b;perdir=()) where {T,M}
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b) over I ∈ CartesianIndices(map(n->2:n-1,Na))
    end
    BC!(a,zero(SVector{M-1,T}),false,perdir)  # correct μ₀ @ boundaries
end
restrict!(a,b) = @inside a[I] = restrict(I,b)
prolongate!(a,b) = @inside a[I] = b[down(I)]

@inline divisible(N) = mod(N,2)==0 && N>4
@inline divisible(l::Poisson) = all(size(l.x) .|> divisible)
"""
    MultiLevelPoisson{N,M}

Composite type used to solve the pressure Poisson equation with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method.
The only variable is `levels`, a vector of nested `Poisson` systems.
"""
struct MultiLevelPoisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    x::S
    L::V
    z::S
    levels :: Vector{Poisson{T,S,V}}
    n :: Vector{Int16}
    perdir :: NTuple # direction of periodic boundary condition
    function MultiLevelPoisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};maxlevels=10,perdir=()) where T
        levels = Poisson[Poisson(x,L,z;perdir)]
        while divisible(levels[end]) && length(levels) <= maxlevels
            push!(levels,restrictML(levels[end]))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where n>2"
        @assert (length(levels)>2) text
        new{T,typeof(x),typeof(L)}(x,L,z,levels,[],perdir)
    end
end

function update!(ml::MultiLevelPoisson)
    update!(ml.levels[1])
    for l ∈ 2:length(ml.levels)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L,perdir=ml.levels[l-1].perdir)
        update!(ml.levels[l])
    end
end

# Finest-level face-coefficient array (the one the projection corrects with
# and the one `update!` restricts to coarser levels). For a bare `Poisson`
# this is `p.L`; for a `MultiLevelPoisson` it is `levels[1].L` — which for the
# default constructor aliases `ml.L`, so writing it updates both.
fineL(p::Poisson) = p.L
fineL(p::MultiLevelPoisson) = p.levels[1].L

# 1/ρ at the d-face of cell I, either read from an array (`invρ[I,d]`) or
# evaluated from a caller-supplied closure (`invρ(d,I)`). The closure keeps
# the face-averaging convention (arithmetic-mean-of-1/ρ, f-then-ρ, harmonic,
# …) in the downstream package rather than baking one choice into the core.
@inline _invρf(invρ::AbstractArray, d, I) = @inbounds invρ[I, d]
@inline _invρf(invρ, d, I) = invρ(d, I)

"""
    density_coefficient!(pois, μ₀, invρ; perdir=())

Set the pressure-Poisson face coefficients for a **variable-density**
projection: `L[I,d] = μ₀[I,d] · (1/ρ)_face(d,I)`, then refresh the solver
(`update!`, recomputing the diagonal and restricting `L` through the
multigrid levels). Call once per step after the density field changes —
the constant-density case needs nothing (`L === μ₀`).

`invρ` supplies `1/ρ` on the **faces**, either as
  * an `AbstractArray` shaped like `μ₀` (`invρ[I,d]`), or
  * a callable `invρ(d, I)` evaluated on the fly (no stored `1/ρ` array),

so the face-averaging convention stays with the caller (VoF averages
`1/ρ`; geometric-VoF packages average the volume fraction first). Folding
in the *measured* `μ₀` keeps moving immersed bodies and the density jump
consistent in one place.
"""
function density_coefficient!(pois::AbstractPoisson, μ₀::AbstractArray{T,Mp1}, invρ::F; perdir=()) where {T,Mp1,F}
    # `D` from μ₀'s type parameter (so `SVector{D}` below is concrete, not a
    # runtime-`size` value which would be type-unstable and heap-allocate),
    # and `invρ::F` forces specialization so the closure call inlines.
    D = Mp1 - 1
    L = fineL(pois)
    R = inside_u(L)   # 2:N-1 per spatial dim — the face-coefficient range
                      # (matches μ₀/`restrictL!`, NOT the convective-flux range)
    for d in 1:D
        @loop L[I,d] = μ₀[I,d] * _invρf(invρ, d, I) over I ∈ R
    end
    BC!(L, zeros(SVector{D,T}), false, perdir)  # μ₀ no-flux wall convention
    update!(pois)
    return pois
end

function Vcycle!(ml::MultiLevelPoisson;l=1,ω=1)
    fine,coarse = ml.levels[l],ml.levels[l+1]
    # set up coarse level
    Jacobi!(fine)
    restrict!(coarse.r,fine.r)
    fill!(coarse.x,0.)
    # solve coarse (with recursion if possible)
    l+1<length(ml.levels) && Vcycle!(ml,l=l+1; ω)
    smooth!(coarse;ω)
    # correct fine
    prolongate!(fine.ϵ,coarse.x)
    increment!(fine; ω)
end

mult!(ml::MultiLevelPoisson,x) = mult!(ml.levels[1],x)
residual!(ml::MultiLevelPoisson,x) = residual!(ml.levels[1],x)

smooth! = GaussSeidelRB!

function solver!(ml::MultiLevelPoisson{T};tol=1e-4,itmx=32) where T
    p = ml.levels[1]
    residual!(p); r₂ = L₂(p); ω = T(1)
    nᵖ=0; @log ", $nᵖ, $(L∞(p)), $r₂, $ω\n"
    while nᵖ<itmx
        Vcycle!(ml; ω)
        smooth!(p; ω); 
        rnew = L₂(p); nᵖ+=1
        @log ", $nᵖ, $(L∞(p)), $rnew, $ω\n"
        if     rnew ≥ r₂
            ω = max(0.2, 0.9ω) |> T
        elseif rnew < r₂
            ω = min(1.0, 1.02ω) |> T
        end
        r₂ = rnew
        r₂<tol && break
    end
    perBC!(p.x,p.perdir)
    push!(ml.n,nᵖ);
end