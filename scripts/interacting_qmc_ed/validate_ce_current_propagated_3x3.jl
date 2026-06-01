#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Random
using CanEnsAFQMC

include(joinpath(@__DIR__, "ce_unequal_time_current_helpers.jl"))

function compare_projected_and_propagated(; refresh_interval::Int, tolerance::Float64)
    lx, ly = 3, 3
    beta, dtau = 10.0, 0.1
    nup, ndn = 4, 4
    L = round(Int, beta / dtau)
    system = GenericHubbard(
        (lx, ly, 1),
        (nup, ndn),
        hopping_matrix_Hubbard_2d(lx, ly, 1.0),
        -5.0,
        0.0,
        beta,
        L;
        sys_type=ComplexF64,
        useChargeHST=false,
        useFirstOrderTrotter=false,
    )
    qmc = QMC(
        system;
        nwarmups=20,
        nsamples=1,
        measure_interval=2,
        stab_interval=10,
        useClusterUpdate=true,
        cluster_size=3,
        num_FourierPoints=10,
        forceSymmetry=true,
        isLowrank=false,
        lrThld=1e-10,
        saveRatio=false,
    )

    Random.seed!(20260601)
    walker = Walker(system, qmc)
    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)
    sweep!(system, qmc, walker, loop_number=qmc.measure_interval)

    ρup = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    ρdn = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    update!(system, walker, ρup, 1)
    update!(system, walker, ρdn, ρup)

    Bup, Bdn = build_B_slices(system, walker)
    prefix_up, suffix_up = build_prefix_suffix(Bup)
    prefix_dn, suffix_dn = build_prefix_suffix(Bdn)
    momenta = [(2π / lx, 0.0), (0.0, 2π / ly)]

    projected = nothing
    projected_time = @elapsed begin
        projected = measure_current_responses_unequaltime(
            system,
            ρup,
            ρdn,
            prefix_up,
            suffix_up,
            prefix_dn,
            suffix_dn,
            momenta,
        )
    end
    propagated = nothing
    propagated_time = @elapsed begin
        propagated = measure_current_responses_unequaltime_propagated(
            system,
            ρup,
            ρdn,
            Bup,
            Bdn,
            prefix_up,
            suffix_up,
            prefix_dn,
            suffix_dn,
            momenta;
            refresh_interval=refresh_interval,
        )
    end

    labels = ("longitudinal", "transverse")
    maxdiff = 0.0
    for iq in eachindex(momenta)
        diff = abs(real(projected[iq]) - real(propagated[iq]))
        maxdiff = max(maxdiff, diff)
        @printf(
            "refresh=%3d %-12s projected=% .15e propagated=% .15e absdiff=%.3e\n",
            refresh_interval,
            labels[iq],
            real(projected[iq]),
            real(propagated[iq]),
            diff,
        )
    end
    @printf(
        "refresh=%3d timing projected=%.3fs propagated=%.3fs speedup=%.2fx\n",
        refresh_interval,
        projected_time,
        propagated_time,
        projected_time / propagated_time,
    )
    maxdiff <= tolerance || error("propagated current estimator mismatch: refresh=$refresh_interval maxdiff=$maxdiff tolerance=$tolerance")
    return maxdiff
end

function main()
    exact_refresh_diff = compare_projected_and_propagated(refresh_interval=1, tolerance=1e-8)
    production_refresh_diff = compare_projected_and_propagated(refresh_interval=10, tolerance=1e-6)
    @printf(
        "PASS: propagated current estimator agrees with stable projection; maxdiff(refresh=1)=%.3e maxdiff(refresh=10)=%.3e\n",
        exact_refresh_diff,
        production_refresh_diff,
    )
end

main()
