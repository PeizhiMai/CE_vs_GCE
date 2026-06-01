#!/usr/bin/env julia

using Random
using CanEnsAFQMC

include(joinpath(@__DIR__, "ce_unequal_time_current_helpers.jl"))

function build_system(; lx=3, ly=3, nup=4, ndn=4, u=-5.0, beta=10.0, dtau=0.025)
    tmat = hopping_matrix_Hubbard_2d(lx, ly, 1.0)
    L = round(Int, beta / dtau)
    return GenericHubbard(
        (lx, ly, 1), (nup, ndn), tmat, u, 0.0, beta, L,
        sys_type=Float64, useChargeHST=true, useFirstOrderTrotter=false,
    )
end

function build_qmc(system; nwarmups=32, nsamples=1, measure_interval=2, cluster_size=3, nft=10)
    return QMC(
        system,
        nwarmups=nwarmups, nsamples=nsamples, measure_interval=measure_interval,
        stab_interval=10, useClusterUpdate=true, cluster_size=cluster_size,
        num_FourierPoints=nft, forceSymmetry=true, isLowrank=true, lrThld=1e-10,
        saveRatio=false,
    )
end

function main()
    u = isempty(ARGS) ? -5.0 : parse(Float64, ARGS[1])
    system = build_system(u=u)
    qmc = build_qmc(system)
    Random.seed!(1234)

    walker = Walker(system, qmc)
    ρup = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    ρdn = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    qxmin = 2π / system.Ns[1]
    qymin = 2π / system.Ns[2]

    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)
    sweep!(system, qmc, walker, loop_number=qmc.measure_interval)
    update!(system, walker, ρup, 1)
    update!(system, walker, ρdn, ρup)

    Bup, Bdn = build_B_slices(system, walker)
    prefix_up, suffix_up = build_prefix_suffix(Bup)
    prefix_dn, suffix_dn = build_prefix_suffix(Bdn)

    λL_old = real(measure_CurrentResponse(system, ρup, ρdn, qx=qmin, qy=0.0))
    λT_old = real(measure_CurrentResponse(system, ρup, ρdn, qx=0.0, qy=qmin))
    λL_up = real(canonical_same_spin_current_response(system, ρup, prefix_up, suffix_up, qx=qmin, qy=0.0))
    λL_dn = real(canonical_same_spin_current_response(system, ρdn, prefix_dn, suffix_dn, qx=qmin, qy=0.0))
    λT_up = real(canonical_same_spin_current_response(system, ρup, prefix_up, suffix_up, qx=0.0, qy=qmin))
    λT_dn = real(canonical_same_spin_current_response(system, ρdn, prefix_dn, suffix_dn, qx=0.0, qy=qmin))
    JLq = current_operator_x(system, qx=qxmin, qy=0.0)
    JLm = current_operator_x(system, qx=-qxmin, qy=0.0)
    JTq = current_operator_x(system, qx=0.0, qy=qymin)
    JTm = current_operator_x(system, qx=0.0, qy=-qymin)
    crossL = real(system.β * (
        measure_Bilinear(system, ρup, JLq) * measure_Bilinear(system, ρdn, JLm) +
        measure_Bilinear(system, ρdn, JLq) * measure_Bilinear(system, ρup, JLm)
    ) / system.V)
    crossT = real(system.β * (
        measure_Bilinear(system, ρup, JTq) * measure_Bilinear(system, ρdn, JTm) +
        measure_Bilinear(system, ρdn, JTq) * measure_Bilinear(system, ρup, JTm)
    ) / system.V)
    λL = real(measure_current_response_unequaltime(system, ρup, ρdn, prefix_up, suffix_up, prefix_dn, suffix_dn, qx=qxmin, qy=0.0))
    λT = real(measure_current_response_unequaltime(system, ρup, ρdn, prefix_up, suffix_up, prefix_dn, suffix_dn, qx=0.0, qy=qymin))
    kx = real(measure_KxPerSite(system, ρup, ρdn))

    println("U = ", u)
    println("lambdaL_old = ", λL_old)
    println("lambdaT_old = ", λT_old)
    println("lambdaL_up = ", λL_up, " lambdaL_dn = ", λL_dn, " crossL = ", crossL)
    println("lambdaT_up = ", λT_up, " lambdaT_dn = ", λT_dn, " crossT = ", crossT)
    println("lambdaL = ", λL)
    println("lambdaT = ", λT)
    println("rho_s = ", 0.25 * (λL - λT))
    println("rho_s_dia = ", 0.25 * (-kx - λT))
end

main()
