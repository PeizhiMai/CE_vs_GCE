#!/usr/bin/env julia

using Test
using LinearAlgebra
using MPI

include(joinpath(@__DIR__, "square_lattice_geometry.jl"))
using .SquareLatticeGeometry
include(joinpath(@__DIR__, "smoqy_obc_equal_time.jl"))
using .SmoQyOBCEqualTime

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()

@testset "SmoQy OBC signed equal-time accumulator" begin
    geometry = build_square_geometry(2, 2; boundary="open")
    rho_up = Diagonal(fill(0.4, 4)) |> Matrix
    rho_dn = Diagonal(fill(0.3, 4)) |> Matrix
    Gup = Matrix{Float64}(I, 4, 4) - transpose(rho_up)
    Gdn = Matrix{Float64}(I, 4, 4) - transpose(rho_dn)

    accumulator = OBCEqualTimeAccumulator(4)
    expected = measure_equal_time_observables(geometry, rho_up, rho_dn; U=-3.0)
    recorded = record_obc_equal_time!(
        accumulator, geometry, Gup, Gdn, 1.0; U=-3.0,
    )
    @test recorded.kinetic_per_site == expected.kinetic_per_site
    @test recorded.double_occupancy_per_site ≈ expected.double_occupancy_per_site
    @test accumulator.nsamples == 1
    @test accumulator.phase_sum == 1

    pooled = SmoQyOBCEqualTime._pool(accumulator, MPI.COMM_WORLD)
    means, errors, site_mean = SmoQyOBCEqualTime._ratio_stats(pooled)
    @test means[1] == expected.kinetic_per_site
    @test means[2] ≈ expected.double_occupancy_per_site
    @test means[3] ≈ expected.nn_spin
    @test site_mean ≈ fill(0.7, 4)
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        @test all(isnan, errors)
    else
        @test all(isfinite, errors)
        @test maximum(abs, errors) < 1e-12
    end

    mktempdir() do directory
        write_obc_rank_accumulator(
            directory, accumulator, MPI.Comm_rank(MPI.COMM_WORLD), geometry,
        )
        write_obc_pooled_outputs(
            MPI.COMM_WORLD, directory, accumulator, geometry; beta=2.0, U=-3.0,
        )
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            for filename in (
                "equal_time_kinetic_per_site_qmc.tsv",
                "equal_time_double_occupancy_per_site_qmc.tsv",
                "equal_time_nn_spin_qmc.tsv",
                "equal_time_nn_connected_charge_qmc.tsv",
                "equal_time_bond_observables_qmc.tsv",
                "obc_equal_time_site_rank_pID-0.tsv",
            )
                @test isfile(joinpath(directory, filename))
            end
        end
    end

    # A zero global phase denominator must fail rather than emit a numerical
    # zero-sign observable.
    cancelling = OBCEqualTimeAccumulator(4)
    record_obc_equal_time!(cancelling, geometry, Gup, Gdn, 1.0; U=3.0)
    record_obc_equal_time!(cancelling, geometry, Gup, Gdn, -1.0; U=3.0)
    cancelled_pool = SmoQyOBCEqualTime._pool(cancelling, MPI.COMM_WORLD)
    @test_throws ErrorException SmoQyOBCEqualTime._ratio_stats(cancelled_pool)

    # Match the exact sign/phase expression in SmoQyDQMC.make_measurements!.
    @test configuration_phase((; Sb=0.0), -1.0, 1.0) == -1
    @test configuration_phase((; Sb=0.5im * pi), 1.0 + 0im, 1.0 + 0im) ≈ -1im

    provenance = smoqy_provenance()
    @test provenance.upstream_version == "2.0.12"
    @test length(provenance.fork_commit) == 40
end

MPI.Barrier(MPI.COMM_WORLD)
initialized_here && MPI.Finalize()
