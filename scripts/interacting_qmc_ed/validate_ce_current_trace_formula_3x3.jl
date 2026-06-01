#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Random
using CanEnsAFQMC

include(joinpath(@__DIR__, "ce_unequal_time_current_helpers.jl"))

function trace_formula(
    U::AbstractMatrix{ComplexF64},
    V::AbstractMatrix{ComplexF64},
    A::AbstractMatrix{ComplexF64},
    B::AbstractMatrix{ComplexF64},
)
    M = Matrix{ComplexF64}(I, size(U, 1), size(U, 1)) .+ V * U
    G00 = inv(M)
    dA = tr(G00 * V * A * U)
    dB = tr(G00 * V * U * B)
    connected = tr(G00 * V * A * U * B) - tr(G00 * V * U * B * G00 * V * A * U)
    return connected + dA * dB
end

function displaced_greens(
    U::Matrix{ComplexF64},
    V::Matrix{ComplexF64};
    boundary::Bool=false,
)
    n = size(U, 1)
    Iₙ = Matrix{ComplexF64}(I, n, n)
    if boundary
        G = inv(Iₙ + V * U)
        return G, G, Iₙ - G, -G
    end
    Gττ = inv(Iₙ + U * V)
    G00 = inv(Iₙ + V * U)
    Gτ0 = inv(inv(U) + V)
    G0τ = -inv(inv(V) + U)
    return Gττ, G00, Gτ0, G0τ
end

function dense_factorizations(factors)
    n = size(factors[1].L, 1)
    ws = ldr_workspace(Matrix{ComplexF64}(I, n, n))
    mats = Matrix{ComplexF64}[]
    for factor in factors
        M = zeros(ComplexF64, n, n)
        copyto!(M, factor, ws)
        push!(mats, M)
    end
    return mats
end

function main()
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

    Random.seed!(20260529)
    walker = Walker(system, qmc)
    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)
    sweep!(system, qmc, walker, loop_number=qmc.measure_interval)

    ρ = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    update!(system, walker, ρ, 1)
    Bup, _ = build_B_slices(system, walker)
    prefix, suffix = build_prefix_suffix(Bup)
    P = dense_factorizations(prefix)
    S = dense_factorizations(suffix)

    qmin = 2π / lx
    checks = (("longitudinal", qmin, 0.0), ("transverse", 0.0, qmin))
    test_slices = (1, 2, 10, 50)  # avoid the β-boundary and the most ill-conditioned final slice
    # This diagnostic intentionally uses dense/plain inverses instead of the
    # LDR kernels used in production, so mid-β slices are mildly conditioned.
    # The old bond-expanded estimator failed this check at O(1e-1)-O(1);
    # O(1e-5) dense-inverse differences are numerical noise for this purpose.
    tolerance = 1e-4
    maxdiff = 0.0

    for (label, qx, qy) in checks
        Jq = ce_current_operator_x(system, qx=qx, qy=qy)
        Jm = ce_current_operator_x(system, qx=-qx, qy=-qy)
        @printf("== %s q=(%.12g, %.12g) ==\n", label, qx, qy)
        for m in 1:ρ.Nft
            z = ComplexF64(ρ.expiφμ[m])
            for l in test_slices
                U = P[l + 1]
                V = z * S[l + 1]
                Gττ, G00, Gτ0, G0τ = displaced_greens(U, V)
                trace_val = trace_formula(U, V, Jq, Jm) / system.V
                estimator_val = current_corr_pair_q(
                    system,
                    Gττ,
                    G00;
                    Gτ0₁=Gτ0,
                    G0τ₁=G0τ,
                    qx=qx,
                    qy=qy,
                    same_spin=true,
                )
                diff = abs(trace_val - estimator_val)
                maxdiff = max(maxdiff, diff)
                if m == 1
                    @printf(
                        "slice=%3d trace=% .12e%+.3ei estimator=% .12e%+.3ei |diff|=%.3e\n",
                        l,
                        real(trace_val),
                        imag(trace_val),
                        real(estimator_val),
                        imag(estimator_val),
                        diff,
                    )
                end
                diff <= tolerance || error("current estimator trace-formula check failed: label=$label m=$m slice=$l diff=$diff")
            end
        end
    end
    @printf("PASS: max |trace - estimator| = %.3e\n", maxdiff)
end

main()
