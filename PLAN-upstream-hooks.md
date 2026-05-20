# PLAN 1 — Upstream hooks in WaterLily.jl

**Repo:** `pankgeorg/WaterLily.jl` (fork), branch `plan/upstream-hooks`.
**Target upstream:** `WaterLily-jl/WaterLily.jl#master`.
**Status:** plan only; no implementation yet. Open the upstream issue *with* a
reference implementation, not before.

## Scope

Three minimal API additions that unblock four downstream packages
(Turbulence.jl, VoF.jl, Propellers.jl, ShipShapes.jl) without committing
WaterLily to any specific physical model.

### Hook 1: per-cell effective viscosity

`Flow` currently carries a scalar kinematic viscosity `ν :: T` used inside
`conv_diff!` (Flow.jl:38). Turbulence closures need a spatially-varying
`ν_eff = ν + ν_t(I)`. The hook must:

- accept either a scalar `T` or a `D`-dimensional array on the same backend
  as `flow.p`;
- default to the current scalar `ν` (zero ergonomic cost for non-turbulent
  users);
- be read inside the convection–diffusion kernel without breaking GPU
  kernel inference (no closure over a heap-allocated wrapper).

**Proposed surface:**

```julia
Flow(...; ν::Union{T, AbstractArray{T,D}} = 0)
```

with `conv_diff!(...; ν)` dispatching on `ν::Number` vs `ν::AbstractArray`.
The array branch reads `ν[I]` (or `ν[ϕ(j,I,...)]` at faces if needed —
that's a sub-decision, see open questions).

### Hook 2: scalar transport helper

A generic conservative advection–diffusion update for one scalar field
`φ`. Required by VoF (advect `α`), Turbulence Tier-2/3 (transport `ν̃`, `k`,
`ω`), and any future scalar (temperature, species).

**Proposed surface:**

```julia
transport!(φ::Sf, u::Vf, σ::Sf;
           D::Union{T,Sf}=zero(T),
           S::Union{Nothing,Function,Sf}=nothing,
           λ=quick, perdir=())
```

Returns nothing, updates `φ` in place by `Δt`. `σ` is a workspace
field of the same shape as `φ` (matches the existing `Flow.σ` workspace
pattern). `λ` is the convective scheme. `S` is a source term, either a
field or a function `S(I, φ, t)`.

This becomes the building block of every Tier-2+ closure. It must reuse
the same stencil utilities as `conv_diff!` so the GPU code-gen is
identical.

### Hook 3: composable body-force protocol

Today `udf` in `mom_step!` (Flow.jl:160) takes exactly one function. To
compose gravity + actuator disk + Coriolis + custom user force without a
hand-rolled combinator, define:

```julia
abstract type BodyForce end
apply_force!(f, ::BodyForce, flow, t) = nothing   # default no-op
```

`mom_step!` accepts `forces::Tuple{Vararg{BodyForce}}` and folds them
left-to-right onto `flow.f`. The existing `udf::Function` path stays as a
deprecated alias that wraps into an anonymous `BodyForce`.

## Non-goals (deliberately excluded)

- **No** AMR / adaptive grid changes. That's a research project.
- **No** sub-grid model selection machinery (registry of `eddy_viscosity!`
  implementations). Downstream packages handle that themselves.
- **No** new abstract types unless strictly required — extend
  `AbstractFlow`, don't add a sibling.

## Validation

The hooks are *enabling infrastructure*. Correctness is proved by:

1. **All existing WaterLily tests must pass unchanged** with default
   arguments (scalar `ν`, no forces tuple, no `transport!` call). This is
   the *only* upstream-blocking criterion.
2. **An array-`ν` regression test** that sets `ν = ν_scalar * ones(...)` and
   asserts results bit-identical (in Float64) or within 1 ULP (in Float32)
   to the scalar baseline on a Taylor–Green vortex.
3. **A `transport!` MMS test**: advect–diffuse `φ(x,t) = sin(2πx)e^(-4π²Dt)`
   on a periodic 1D-in-`x` grid; check L₂ error scales as O(Δx²) for the
   `cds` scheme and O(Δx³) for `quick`.
4. **A `BodyForce` composition test**: gravity-only vs `(Gravity(),)` tuple
   must give bit-identical results; gravity + actuator-disk-stub must give
   the same result as the equivalent inlined `udf`.

## Performance budget

| Configuration                | Cost vs baseline |
|------------------------------|------------------|
| Scalar `ν`, no forces        | identical (must) |
| Array `ν`, no forces         | ≤ 5% per-step    |
| Scalar `ν`, one BodyForce    | ≤ 1% per-step    |
| `transport!` for one scalar  | ≤ 80% of one `conv_diff!` call |

Benchmarked with `BenchmarkTools` on the existing `cube` and `2D vortex`
tests over 100 steps on:

- CPU: `Threads.nthreads() = 8`, Float32, 256³
- GPU: CUDA RTX 3090 / 4090 if available, Float32, 512³

## Harness

- All four validation tests live in `test/runtests.jl` under a new
  `@testset "Hooks"` block.
- The perf check is a CI job that records timings on the upstream `master`
  baseline vs the PR branch and posts a comparison comment. Until the PR
  is opened upstream, runs locally via `julia --project=. test/perf.jl`.
- No OpenFOAM comparison at this layer — the hooks have no physics.

## Risks

- **Performance regression on GPU.** Array-`ν` read inside the hot kernel
  may inhibit some optimizations. Mitigation: dispatch on `ν` type so the
  scalar case compiles to identical code, not a generic branch.
- **API churn for `LilyPad.jl` / `BiotSavartBCs.jl`.** Both use
  `flow_ctor`/`pois_ctor`. The new hooks should be *additive* — no
  signature change to existing constructors. Verify by building both
  downstream packages against the PR branch before opening upstream.
- **Maintainer pushback.** WaterLily is intentionally small. The
  body-force protocol is the part most likely to be rejected on
  scope grounds; if so, drop it and keep `udf` — the downstream
  packages can compose forces themselves.

## Milestones

| # | Goal                                                       | Done when                                                |
|---|------------------------------------------------------------|----------------------------------------------------------|
| 1 | Hook 1 implemented + tested locally                        | `pankgeorg/WaterLily.jl@plan/upstream-hooks` green       |
| 2 | Hook 2 implemented + MMS test                              | `transport!` order-of-accuracy verified                  |
| 3 | Hook 3 implemented OR shelved                              | decision documented in this file                         |
| 4 | LilyPad.jl + BiotSavartBCs.jl still build against branch   | both pass their own tests                                |
| 5 | Open upstream issue + PR simultaneously                    | `WaterLily-jl/WaterLily.jl#NNN` filed with diff attached |

## Open questions (decide before implementation)

- Cell-centered `ν_t[I]` or face-interpolated `ν_t[I,j]`? OpenFOAM uses
  face-interpolated for the diffusion term. Cell-centered is simpler and
  sufficient for Smagorinsky. Pick face-interp if k-ω SST is on the
  near-term roadmap, otherwise start cell-centered.
- Should `transport!` advance one time step (matches `mom_step!` cadence)
  or N sub-steps for stability under tight α/k constraints? Start with
  one step; revisit when VoF.jl forces the issue.
- Float32 vs Float64 for k, ε, ω: these can become very small. WaterLily
  defaults to Float32. Consider per-field precision.
