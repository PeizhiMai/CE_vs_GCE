#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Random
using CanEnsAFQMC

include(joinpath(@__DIR__, "ce_unequal_time_current_helpers.jl"))

function gen_basis(nsites::Int, nparticles::Int)
    basis = Int[]
    function rec(start::Int, left::Int, state::Int)
        if left == 0
            push!(basis, state)
            return
        end
        for i in start:(nsites - left + 1)
            rec(i + 1, left - 1, state | (1 << (i - 1)))
        end
    end
    rec(1, nparticles, 0)
    return basis
end

@inline occ(state::Int, site::Int) = (state >> (site - 1)) & 1

function cdagc(state::Int, i::Int, j::Int)
    if occ(state, j) == 0 || occ(state, i) == 1
        return nothing
    end
    mask_j = 1 << (j - 1)
    st = state & ~mask_j
    sign1 = isodd(count_ones(state & (mask_j - 1))) ? -1 : 1
    mask_i = 1 << (i - 1)
    sign2 = isodd(count_ones(st & (mask_i - 1))) ? -1 : 1
    return sign1 * sign2, st | mask_i
end

occupied_sites(state::Int, nsites::Int) = [i for i in 1:nsites if occ(state, i) == 1]

function manybody_propagator(B::AbstractMatrix{ComplexF64}, occ_lists)
    dim = length(occ_lists)
    M = zeros(ComplexF64, dim, dim)
    for row in 1:dim
        rows = occ_lists[row]
        for col in 1:dim
            cols = occ_lists[col]
            M[row, col] = det(@view B[rows, cols])
        end
    end
    return M
end

function bilinear_sector_matrix(A::AbstractMatrix{ComplexF64}, basis::Vector{Int})
    dim = length(basis)
    index = Dict(state => idx for (idx, state) in enumerate(basis))
    M = zeros(ComplexF64, dim, dim)
    V = size(A, 1)
    for (col, state) in enumerate(basis)
        for i in 1:V, j in 1:V
            abs(A[i, j]) < 1e-14 && continue
            out = cdagc(state, i, j)
            out === nothing && continue
            sign, new_state = out
            M[index[new_state], col] += A[i, j] * sign
        end
    end
    return M
end

function prefix_suffix_dense(Ms)
    dim = size(Ms[1], 1)
    L = length(Ms)
    prefix = Vector{Matrix{ComplexF64}}(undef, L + 1)
    suffix = Vector{Matrix{ComplexF64}}(undef, L + 1)
    prefix[1] = Matrix{ComplexF64}(I, dim, dim)
    for l in 1:L
        prefix[l + 1] = Ms[l] * prefix[l]
    end
    suffix[L + 1] = Matrix{ComplexF64}(I, dim, dim)
    for l in L:-1:1
        suffix[l] = suffix[l + 1] * Ms[l]
    end
    return prefix, suffix
end

function exact_spin_parts(system, Bseq, N, qx, qy)
    basis = gen_basis(system.V, N)
    occ_lists = [occupied_sites(s, system.V) for s in basis]
    Ms = [manybody_propagator(B, occ_lists) for B in Bseq]
    prefix, suffix = prefix_suffix_dense(Ms)
    Jq = bilinear_sector_matrix(ce_current_operator_x(system, qx=qx, qy=qy), basis)
    Jm = bilinear_sector_matrix(ce_current_operator_x(system, qx=-qx, qy=-qy), basis)
    Z = tr(prefix[end])
    Δτ = system.β / system.L
    same = zero(ComplexF64)
    jτ_q = ComplexF64[]
    j0_m = ComplexF64[]
    for l in 0:(system.L - 1)
        U = prefix[l + 1]
        V = suffix[l + 1]
        same += tr(V * Jq * U * Jm) / Z
        push!(jτ_q, tr(V * Jq * U) / Z)
        push!(j0_m, tr(V * U * Jm) / Z)
    end
    return (same=Δτ * same / system.V, jτ_q=jτ_q, j0_m=j0_m)
end

function main()
    lx, ly = 3, 3
    nup, ndn = 4, 4
    β, Δτ = 1.0, 0.25
    L = round(Int, β / Δτ)
    system = GenericHubbard(
        (lx, ly, 1),
        (nup, ndn),
        hopping_matrix_Hubbard_2d(lx, ly, 1.0),
        -5.0,
        0.0,
        β,
        L;
        sys_type=ComplexF64,
        useChargeHST=false,
        useFirstOrderTrotter=false,
    )

    Random.seed!(20260530)
    aux = rand((-1, 1), system.V, system.L)
    qmc = QMC(
        system;
        nwarmups=0,
        nsamples=1,
        measure_interval=1,
        stab_interval=10,
        useClusterUpdate=false,
        cluster_size=1,
        num_FourierPoints=10,
        forceSymmetry=true,
        isLowrank=false,
        lrThld=1e-10,
        saveRatio=false,
    )
    walker = Walker(system, qmc, auxfield=aux)
    ρup = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    ρdn = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    update!(system, walker, ρup, 1)
    update!(system, walker, ρdn, ρup)

    Bup, Bdn = build_B_slices(system, walker)
    prefix_up, suffix_up = build_prefix_suffix(Bup)
    prefix_dn, suffix_dn = build_prefix_suffix(Bdn)
    qxmin = 2π / lx
    qymin = 2π / ly

    for (label, qx, qy) in (("L", qxmin, 0.0), ("T", 0.0, qymin))
        exact_up = exact_spin_parts(system, Bup, nup, qx, qy)
        exact_dn = exact_spin_parts(system, Bdn, ndn, qx, qy)
        cross = Δτ * sum(exact_up.jτ_q .* exact_dn.j0_m .+ exact_dn.jτ_q .* exact_up.j0_m) / system.V
        exact = exact_up.same + exact_dn.same + cross
        estimator = measure_current_response_unequaltime(
            system,
            ρup,
            ρdn,
            prefix_up,
            suffix_up,
            prefix_dn,
            suffix_dn;
            qx=qx,
            qy=qy,
        )
        diff_re = abs(real(exact) - real(estimator))
        @printf(
            "%s exact=% .15e%+.3ei estimator=% .15e%+.3ei real_diff=%.3e\n",
            label,
            real(exact),
            imag(exact),
            real(estimator),
            imag(estimator),
            diff_re,
        )
        diff_re < 1e-8 || error("full exact current response real-part mismatch for $label")
    end
    println("PASS: full fixed-sector exact trace agrees with estimator for physical real parts.")
end

main()
