# Draft regression for the variable-density Poisson coefficient hook.
#
# density_coefficient!(pois, μ₀, invρ) sets L = μ₀ ⊙ (1/ρ)_face and
# refreshes the solver. `invρ` is a face 1/ρ array or a closure invρ(d,I).

using Test
using WaterLily
using StaticArrays

@testset "variable density: density_coefficient!" begin
    T = Float32; N = (8, 8); D = 2; Ng = N .+ 2
    mkμ₀() = (m = ones(T, Ng..., D); WaterLily.BC!(m, ntuple(zero, D)); m)
    x() = zeros(T, Ng); Iint = CartesianIndex(4, 4)

    # a non-uniform face 1/ρ field (water/air-ish, 1/ρ ∈ [1e-3, 1])
    invρ = ones(T, Ng..., D)
    for I in CartesianIndices((Ng...,)), d in 1:D
        invρ[I, d] = I.I[2] < Ng[2] ÷ 2 ? T(1e-3) : T(1)
    end

    @testset "constant 1/ρ ≡ 1 ⇒ L == μ₀ (interior)" begin
        μ₀ = mkμ₀()
        p = WaterLily.Poisson(x(), copy(μ₀), x())
        WaterLily.density_coefficient!(p, μ₀, ones(T, Ng..., D))
        @test all(p.L[I, d] ≈ μ₀[I, d] for I in CartesianIndices((3:Ng[1]-1, 3:Ng[2]-1)), d in 1:D)
    end

    @testset "variable 1/ρ array ⇒ L = μ₀·(1/ρ)" begin
        μ₀ = mkμ₀()
        p = WaterLily.Poisson(x(), copy(μ₀), x())
        WaterLily.density_coefficient!(p, μ₀, invρ)
        @test p.L[Iint, 1] ≈ μ₀[Iint, 1] * invρ[Iint, 1]
        @test p.L[Iint, 2] ≈ μ₀[Iint, 2] * invρ[Iint, 2]
        # diagonal was refreshed to match the new L
        @test p.D[Iint] ≈ WaterLily.diag(Iint, p.L)
    end

    @testset "closure invρ(d,I) matches the array" begin
        μ₀ = mkμ₀()
        pa = WaterLily.Poisson(x(), copy(μ₀), x())
        pc = WaterLily.Poisson(x(), copy(μ₀), x())
        WaterLily.density_coefficient!(pa, μ₀, invρ)
        WaterLily.density_coefficient!(pc, μ₀, (d, I) -> @inbounds invρ[I, d])
        @test pa.L == pc.L
    end

    @testset "MultiLevelPoisson: fine L set, levels aliased, solver runs" begin
        μ₀ = mkμ₀()
        ml = WaterLily.MultiLevelPoisson(x(), copy(μ₀), x())
        @test ml.L === ml.levels[1].L              # writing fineL updates both
        WaterLily.density_coefficient!(ml, μ₀, invρ)
        @test ml.levels[1].L[Iint, 1] ≈ μ₀[Iint, 1] * invρ[Iint, 1]
        # a solve still proceeds with the variable coefficient
        ml.z .= 0; ml.x .= 0
        WaterLily.solver!(ml; itmx = 5)
        @test all(isfinite, ml.x)
    end
end
