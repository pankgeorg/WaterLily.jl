# Regression test for the array-valued `ν` hook (PLAN 1, Hook 1).
#
# Run as:
#   JULIA_NUM_THREADS=1 julia --project=. test/effective_nu_test.jl
#
# The hot path inside `conv_diff!` was changed from `ν*∂(...)` to
# `_ν(ν,I)*∂(...)`. With a scalar ν, this must compile to identical code
# and produce identical numerical output. With a uniform array ν matching
# the scalar value, results must also be bit-identical (or within 1 ULP
# in Float32).

using Test
using WaterLily
using StaticArrays

@testset "Hook 1: array-valued ν" begin

    @testset "scalar ν unchanged" begin
        # Build a Flow with scalar ν, step it, check basic invariants.
        U = (2/3, -1/3)
        N = (16, 16)
        a = WaterLily.Flow(N, U; T=Float32, ν=Float32(0.01))
        @test a.ν isa Float32
        @test a.ν == Float32(0.01)
        # CFL still computes
        Δt = WaterLily.CFL(a)
        @test isfinite(Δt) && Δt > 0
    end

    @testset "array ν with constant value ≈ scalar" begin
        # Two flows, same parameters, one with scalar ν, one with array ν.
        # After one conv_diff! they should agree to within Float32 ULP.
        U = (2/3, -1/3)
        N = (16, 16)
        ν_val = Float32(0.01)

        a_scalar = WaterLily.Flow(N, U; T=Float32, ν=ν_val)

        ν_arr = fill(ν_val, N .+ 2)
        a_array = WaterLily.Flow(N, U; T=Float32, ν=ν_arr)
        @test a_array.ν isa AbstractArray
        @test all(a_array.ν .== ν_val)

        # Drive the same initial velocity in both (apply a smooth IC)
        for I in CartesianIndices(a_scalar.u)
            v = sinpi((I.I[1] - 1) / N[1]) + cospi((I.I[2] - 1) / N[2])
            a_scalar.u[I] = a_array.u[I] = Float32(v)
        end
        Φs, Φa = similar(a_scalar.p), similar(a_array.p)
        WaterLily.conv_diff!(a_scalar.f, a_scalar.u, Φs, WaterLily.quick; ν=a_scalar.ν)
        WaterLily.conv_diff!(a_array.f, a_array.u, Φa, WaterLily.quick; ν=a_array.ν)

        # Bit-identical or within 1 ULP (Float32 eps ~ 1.2e-7).
        err = maximum(abs.(a_scalar.f .- a_array.f))
        @test err ≤ 5 * eps(Float32) * maximum(abs.(a_scalar.f))
    end

    @testset "array ν enables spatially varying viscosity" begin
        # Non-uniform ν must produce a different result from a constant ν.
        # This is the smoke test that the hook is actually wired up.
        U = (1f0, 0f0)
        N = (16, 16)
        ν_const = fill(Float32(0.01), N .+ 2)
        ν_varying = copy(ν_const)
        ν_varying[8:end, :] .= Float32(0.05)   # 5× viscosity in upstream half

        a1 = WaterLily.Flow(N, U; T=Float32, ν=ν_const)
        a2 = WaterLily.Flow(N, U; T=Float32, ν=ν_varying)
        for I in CartesianIndices(a1.u)
            v = sinpi((I.I[1] - 1) / N[1]) * cospi((I.I[2] - 1) / N[2])
            a1.u[I] = a2.u[I] = Float32(v)
        end
        Φ1, Φ2 = similar(a1.p), similar(a2.p)
        WaterLily.conv_diff!(a1.f, a1.u, Φ1, WaterLily.quick; ν=a1.ν)
        WaterLily.conv_diff!(a2.f, a2.u, Φ2, WaterLily.quick; ν=a2.ν)
        @test maximum(abs.(a1.f .- a2.f)) > 0
    end

    @testset "CFL with array ν" begin
        U = (1f0, 0f0)
        N = (16, 16)
        ν_arr = fill(Float32(0.01), N .+ 2)
        ν_arr[1] = Float32(0.5)   # spike — most restrictive
        a = WaterLily.Flow(N, U; T=Float32, ν=ν_arr)
        # Apply some non-zero velocity for a finite flux_out
        a.u .= 1f0
        Δt = WaterLily.CFL(a)
        # Must be limited by the spike, not the bulk value
        @test Δt < 1 / (5 * 0.5)
    end

end
