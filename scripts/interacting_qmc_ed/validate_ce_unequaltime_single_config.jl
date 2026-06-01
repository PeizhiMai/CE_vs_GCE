#!/usr/bin/env julia

using LinearAlgebra
using Random
using CanEnsAFQMC

include(joinpath(@__DIR__, "ce_unequal_time_current_helpers.jl"))

function build_system(; lx=2, ly=2, nup=2, ndn=2, u=-5.0, beta=4.0, dtau=0.1)
    tmat = hopping_matrix_Hubbard_2d(lx, ly, 1.0)
    L = round(Int, beta / dtau)
    return GenericHubbard(
        (lx, ly, 1), (nup, ndn), tmat, u, 0.0, beta, L,
        sys_type=Float64, useChargeHST=true, useFirstOrderTrotter=false,
    )
end

function build_qmc(system; nwarmups=16, measure_interval=2, cluster_size=2, nft=system.V + 1)
    return QMC(
        system,
        nwarmups=nwarmups, nsamples=1, measure_interval=measure_interval,
        stab_interval=5, useClusterUpdate=true, cluster_size=cluster_size,
        num_FourierPoints=nft, forceSymmetry=true, isLowrank=false, lrThld=1e-10,
        saveRatio=false,
    )
end

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

all_basis(nsites::Int) = collect(0:(1 << nsites) - 1)

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

function bilinear_manybody_matrix(A::AbstractMatrix{ComplexF64}, basis::Vector{Int})
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
            row = index[new_state]
            M[row, col] += A[i, j] * sign
        end
    end
    return M
end

function bilinear_corr_general(
    A::AbstractMatrix{ComplexF64},
    B::AbstractMatrix{ComplexF64},
    Gττ::AbstractMatrix{ComplexF64},
    G00::AbstractMatrix{ComplexF64},
    Gτ0::AbstractMatrix{ComplexF64},
    G0τ::AbstractMatrix{ComplexF64},
)
    V = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, V, V)
    ρττ = Icomplex .- Gττ
    ρ00 = Icomplex .- G00
    disc = sum(A .* transpose(ρττ)) * sum(B .* transpose(ρ00))
    conn = zero(ComplexF64)
    @inbounds for a in 1:V, b in 1:V, c in 1:V, d in 1:V
        conn += A[a, b] * B[c, d] * G0τ[d, a] * Gτ0[b, c]
    end
    return disc - conn
end

function bilinear_corr_trace_formula(
    U::AbstractMatrix{ComplexF64},
    V::AbstractMatrix{ComplexF64},
    A::AbstractMatrix{ComplexF64},
    B::AbstractMatrix{ComplexF64},
)
    M = Matrix{ComplexF64}(I, size(U, 1), size(U, 1)) .+ V * U
    G00 = inv(M)
    dA = tr(G00 * V * A * U)
    dB = tr(G00 * V * U * B)
    cross = tr(G00 * V * A * U * B) - tr(G00 * V * U * B * G00 * V * A * U)
    return cross + dA * dB
end

function finite_difference_trace_corr(
    U::AbstractMatrix{ComplexF64},
    V::AbstractMatrix{ComplexF64},
    A::AbstractMatrix{ComplexF64},
    B::AbstractMatrix{ComplexF64},
    basis::Vector{Int},
    z::ComplexF64;
    h::Float64=1e-7,
)
    Zop = Diagonal(ComplexF64[z ^ count_ones(state) for state in basis])
    function F(η, ξ)
        M = V * exp(η * A) * U * exp(ξ * B)
        ΓM = manybody_propagator(M, basis, size(M, 1))
        return tr(Zop * ΓM)
    end
    fpp = log(F( h,  h))
    fpm = log(F( h, -h))
    fmp = log(F(-h,  h))
    fmm = log(F(-h, -h))
    return (fpp - fpm - fmp + fmm) / (4h^2)
end

function finite_difference_det_corr(
    U::AbstractMatrix{ComplexF64},
    V::AbstractMatrix{ComplexF64},
    A::AbstractMatrix{ComplexF64},
    B::AbstractMatrix{ComplexF64},
    z::ComplexF64;
    h::Float64=1e-6,
)
    function Z(η, ξ)
        return det(Matrix{ComplexF64}(I, size(U, 1), size(U, 1)) .+ z * V * exp(η * A) * U * exp(ξ * B))
    end
    z00 = Z(0.0, 0.0)
    return (Z(h, h) - Z(h, -h) - Z(-h, h) + Z(-h, -h)) / (4h^2 * z00)
end

function annihilation_matrix(site::Int, basis::Vector{Int})
    dim = length(basis)
    index = Dict(state => idx for (idx, state) in enumerate(basis))
    M = zeros(ComplexF64, dim, dim)
    mask = 1 << (site - 1)
    for (col, state) in enumerate(basis)
        occ(state, site) == 1 || continue
        new_state = state & ~mask
        sign = isodd(count_ones(state & (mask - 1))) ? -1 : 1
        row = index[new_state]
        M[row, col] = sign
    end
    return M
end

function occupied_sites(state::Int, nsites::Int)
    out = Int[]
    for i in 1:nsites
        occ(state, i) == 1 && push!(out, i)
    end
    return out
end

function manybody_propagator(B::AbstractMatrix{ComplexF64}, basis::Vector{Int}, nsites::Int)
    dim = length(basis)
    M = zeros(ComplexF64, dim, dim)
    occ_cache = [occupied_sites(state, nsites) for state in basis]
    for (row, occ_r) in enumerate(occ_cache), (col, occ_c) in enumerate(occ_cache)
        length(occ_r) == length(occ_c) || continue
        isempty(occ_r) && (M[row, col] = 1; continue)
        M[row, col] = det(B[occ_r, occ_c])
    end
    return M
end

function manybody_propagator_transpose(B::AbstractMatrix{ComplexF64}, basis::Vector{Int}, nsites::Int)
    return manybody_propagator(transpose(B), basis, nsites)
end

function exact_same_spin_response(system, Bseq::Vector{<:AbstractMatrix}; qx::Float64, qy::Float64, endpoint::Symbol=:right)
    V = system.V
    N = system.N[1]
    L = system.L
    Δτ = system.β / system.L
    basis = gen_basis(V, N)
    dim = length(basis)
    J = bilinear_manybody_matrix(current_operator_x(system, qx=qx, qy=qy), basis)
    MB = [manybody_propagator(B, basis, V) for B in Bseq]

    prefix = Vector{Matrix{ComplexF64}}(undef, L)
    prefix[1] = MB[1]
    for l in 2:L
        prefix[l] = MB[l] * prefix[l - 1]
    end
    suffix = Vector{Matrix{ComplexF64}}(undef, L + 1)
    suffix[L + 1] = Matrix{ComplexF64}(I, dim, dim)
    for l in L:-1:1
        suffix[l] = suffix[l + 1] * MB[l]
    end

    Ufull = prefix[L]
    Z = tr(Ufull)
    total = zero(ComplexF64)
    if endpoint == :right
        for l in 1:L
            total += tr(suffix[l + 1] * J * prefix[l] * J)
        end
    elseif endpoint == :left
        total += tr(Ufull * J * J)
        for l in 1:(L - 1)
            total += tr(suffix[l + 1] * J * prefix[l] * J)
        end
    else
        error("endpoint must be :right or :left")
    end
    return real(Δτ * total / Z / V)
end

function grand_canonical_exact_response(system, Bseq::Vector{<:AbstractMatrix}, z::ComplexF64; qx::Float64, qy::Float64)
    V = system.V
    L = system.L
    Δτ = system.β / system.L
    basis = all_basis(V)
    dim = length(basis)
    J = bilinear_manybody_matrix(current_operator_x(system, qx=qx, qy=qy), basis)
    MB = [manybody_propagator(B, basis, V) for B in Bseq]

    prefix = Vector{Matrix{ComplexF64}}(undef, L)
    prefix[1] = MB[1]
    for l in 2:L
        prefix[l] = MB[l] * prefix[l - 1]
    end
    suffix = Vector{Matrix{ComplexF64}}(undef, L + 1)
    suffix[L + 1] = Matrix{ComplexF64}(I, dim, dim)
    for l in L:-1:1
        suffix[l] = suffix[l + 1] * MB[l]
    end

    Zop = Diagonal(ComplexF64[z ^ count_ones(state) for state in basis])
    Ufull = prefix[L]
    Z = tr(Zop * Ufull)
    total = zero(ComplexF64)
    for l in 1:L
        total += tr(Zop * suffix[l + 1] * J * prefix[l] * J)
    end
    return total * Δτ / Z / V
end

function grand_canonical_exact_slices(system, Bseq::Vector{<:AbstractMatrix}, z::ComplexF64; qx::Float64, qy::Float64)
    V = system.V
    L = system.L
    basis = all_basis(V)
    dim = length(basis)
    J = bilinear_manybody_matrix(current_operator_x(system, qx=qx, qy=qy), basis)
    MB = [manybody_propagator(B, basis, V) for B in Bseq]

    prefix = Vector{Matrix{ComplexF64}}(undef, L)
    prefix[1] = MB[1]
    for l in 2:L
        prefix[l] = MB[l] * prefix[l - 1]
    end
    suffix = Vector{Matrix{ComplexF64}}(undef, L + 1)
    suffix[L + 1] = Matrix{ComplexF64}(I, dim, dim)
    for l in L:-1:1
        suffix[l] = suffix[l + 1] * MB[l]
    end

    Zop = Diagonal(ComplexF64[z ^ count_ones(state) for state in basis])
    Ufull = prefix[L]
    Z = tr(Zop * Ufull)
    vals = ComplexF64[]
    for l in 1:L
        push!(vals, tr(Zop * suffix[l + 1] * J * prefix[l] * J) / Z / V)
    end
    return vals
end

function grand_canonical_est_slices(system, z::ComplexF64, prefix::Vector{<:LDR}, suffix::Vector{<:LDR}; qx::Float64, qy::Float64)
    V = system.V
    L = system.L

    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    tmp = similar(Icomplex)
    full_scaled = ldr(Icomplex)
    suffix_scaled = ldr(Icomplex)
    Gττ = zeros(ComplexF64, V, V)
    G00 = zeros(ComplexF64, V, V)
    Gτ0 = zeros(ComplexF64, V, V)
    G0τ = zeros(ComplexF64, V, V)
    vals = ComplexF64[]

    full = prefix[end]
    scale_factorization!(full_scaled, full, z, ws, tmp)
    inv_IpA!(Gττ, full_scaled, ws)
    copyto!(G00, Gττ)
    copyto!(Gτ0, Icomplex)
    @. Gτ0 = Gτ0 - Gττ
    @. G0τ = -Gττ
    push!(vals, current_corr_pair_q(system, Gττ, G00, Gτ0₁=Gτ0, G0τ₁=G0τ, qx=qx, qy=qy, same_spin=true))

    for l in 1:(L - 1)
        U = prefix[l + 1]
        Vfac = suffix[l + 1]
        scale_factorization!(suffix_scaled, Vfac, z, ws, tmp)
        inv_IpUV!(Gττ, U, suffix_scaled, ws)
        inv_IpUV!(G00, suffix_scaled, U, ws)
        inv_invUpV!(Gτ0, U, suffix_scaled, ws)
        inv_invUpV!(G0τ, suffix_scaled, U, ws)
        @. G0τ = -G0τ
        push!(vals, current_corr_pair_q(system, Gττ, G00, Gτ0₁=Gτ0, G0τ₁=G0τ, qx=qx, qy=qy, same_spin=true))
    end

    return vals
end

function grand_canonical_est_response(system, z::ComplexF64, prefix::Vector{<:LDR}, suffix::Vector{<:LDR}; qx::Float64, qy::Float64)
    return sum(grand_canonical_est_slices(system, z, prefix, suffix, qx=qx, qy=qy)) * (system.β / system.L)
end

function grand_canonical_est_response_nominsign(system, z::ComplexF64, prefix::Vector{<:LDR}, suffix::Vector{<:LDR}; qx::Float64, qy::Float64)
    V = system.V
    L = system.L
    Δτ = system.β / system.L

    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    tmp = similar(Icomplex)
    full_scaled = ldr(Icomplex)
    suffix_scaled = ldr(Icomplex)
    Gττ = zeros(ComplexF64, V, V)
    G00 = zeros(ComplexF64, V, V)
    Gτ0 = zeros(ComplexF64, V, V)
    G0τ = zeros(ComplexF64, V, V)
    total = zero(ComplexF64)

    full = prefix[end]
    scale_factorization!(full_scaled, full, z, ws, tmp)
    inv_IpA!(Gττ, full_scaled, ws)
    copyto!(G00, Gττ)
    copyto!(Gτ0, Icomplex)
    @. Gτ0 = Gτ0 - Gττ
    @. G0τ = Gττ
    total += current_corr_pair_q(system, Gττ, G00, Gτ0₁=Gτ0, G0τ₁=G0τ, qx=qx, qy=qy, same_spin=true)

    for l in 1:(L - 1)
        U = prefix[l + 1]
        Vfac = suffix[l + 1]
        scale_factorization!(suffix_scaled, Vfac, z, ws, tmp)
        inv_IpUV!(Gττ, U, suffix_scaled, ws)
        inv_IpUV!(G00, suffix_scaled, U, ws)
        inv_invUpV!(Gτ0, U, suffix_scaled, ws)
        inv_invUpV!(G0τ, suffix_scaled, U, ws)
        total += current_corr_pair_q(system, Gττ, G00, Gτ0₁=Gτ0, G0τ₁=G0τ, qx=qx, qy=qy, same_spin=true)
    end

    return total * Δτ
end

function main()
    system = build_system()
    qmc = build_qmc(system)
    Random.seed!(1234)

    walker = Walker(system, qmc)
    ρup = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    qmin = 2π / system.Ns[1]

    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)
    sweep!(system, qmc, walker, loop_number=qmc.measure_interval)
    update!(system, walker, ρup, 1)

    Bup, _ = build_B_slices(system, walker)
    prefix_up, suffix_up = build_prefix_suffix(Bup)
    full_dense_from_walker = zeros(Float64, system.V, system.V)
    ws_check = ldr_workspace(Matrix{Float64}(I, system.V, system.V))
    copyto!(full_dense_from_walker, walker.F[1], ws_check)
    full_dense_prefix = zeros(ComplexF64, system.V, system.V)
    copyto!(full_dense_prefix, prefix_up[end], ldr_workspace(Matrix{ComplexF64}(I, system.V, system.V)))
    full_dense_forward = Matrix{ComplexF64}(I, system.V, system.V)
    for B in Bup
        full_dense_forward = full_dense_forward * B
    end

    λ_exact_L = exact_same_spin_response(system, Bup, qx=qmin, qy=0.0, endpoint=:right)
    λ_exact_T = exact_same_spin_response(system, Bup, qx=0.0, qy=qmin, endpoint=:right)
    λ_exact_L_left = exact_same_spin_response(system, Bup, qx=qmin, qy=0.0, endpoint=:left)
    λ_exact_T_left = exact_same_spin_response(system, Bup, qx=0.0, qy=qmin, endpoint=:left)
    λ_est_L = real(canonical_same_spin_current_response(system, ρup, prefix_up, suffix_up, qx=qmin, qy=0.0))
    λ_est_T = real(canonical_same_spin_current_response(system, ρup, prefix_up, suffix_up, qx=0.0, qy=qmin))
    z = ComplexF64(ρup.expiφμ[1])
    λ_gc_exact_L = grand_canonical_exact_response(system, Bup, z, qx=qmin, qy=0.0)
    λ_gc_est_L = grand_canonical_est_response(system, z, prefix_up, suffix_up, qx=qmin, qy=0.0)
    λ_gc_est_L_nominsign = grand_canonical_est_response_nominsign(system, z, prefix_up, suffix_up, qx=qmin, qy=0.0)
    gc_exact_slices_L = grand_canonical_exact_slices(system, Bup, z, qx=qmin, qy=0.0)
    gc_est_slices_L = grand_canonical_est_slices(system, z, prefix_up, suffix_up, qx=qmin, qy=0.0)
    V = system.V
    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    tmp = similar(Icomplex)
    full_scaled = ldr(Icomplex)
    Gfull = zeros(ComplexF64, V, V)
    scale_factorization!(full_scaled, prefix_up[end], z, ws, tmp)
    inv_IpA!(Gfull, full_scaled, ws)
    ρfull = Icomplex .- Gfull
    ρm1 = @view ρup.ρₘ[:, :, 1]

    println("exact_lambdaL = ", λ_exact_L)
    println("exact_lambdaT = ", λ_exact_T)
    println("exact_left_lambdaL = ", λ_exact_L_left)
    println("exact_left_lambdaT = ", λ_exact_T_left)
    println("est_lambdaL = ", λ_est_L)
    println("est_lambdaT = ", λ_est_T)
    println("gc_z = ", z)
    println("gc_exact_lambdaL = ", λ_gc_exact_L)
    println("gc_est_lambdaL = ", λ_gc_est_L)
    println("gc_est_lambdaL_nominsign = ", λ_gc_est_L_nominsign)
    println("rho_full_diff = ", maximum(abs.(ρfull .- ρm1)))
    println("walker_vs_prefix_diff = ", maximum(abs.(full_dense_from_walker .- full_dense_prefix)))
    println("walker_vs_forward_diff = ", maximum(abs.(full_dense_from_walker .- full_dense_forward)))
    println("gc_first5_exact = ", gc_exact_slices_L[1:5])
    println("gc_first5_est = ", gc_est_slices_L[1:5])
    shifted_exact = vcat(gc_exact_slices_L[end:end], gc_exact_slices_L[1:end-1])
    println("gc_first5_exact_shifted = ", shifted_exact[1:5])
    println("gc_maxdiff_shifted = ", maximum(abs.(shifted_exact .- gc_est_slices_L)))

    # exact unequal-time one-body Green-function check for the first nontrivial slice
    full_basis = all_basis(V)
    Cops = [annihilation_matrix(i, full_basis) for i in 1:V]
    Cdops = [adjoint(C) for C in Cops]
    MBfull = [manybody_propagator(B, full_basis, V) for B in Bup]
    prefix_full = Vector{Matrix{ComplexF64}}(undef, system.L)
    prefix_full[1] = MBfull[1]
    for l in 2:system.L
        prefix_full[l] = MBfull[l] * prefix_full[l - 1]
    end
    suffix_full = Vector{Matrix{ComplexF64}}(undef, system.L + 1)
    suffix_full[system.L + 1] = Matrix{ComplexF64}(I, length(full_basis), length(full_basis))
    for l in system.L:-1:1
        suffix_full[l] = suffix_full[l + 1] * MBfull[l]
    end
    Zop = Diagonal(ComplexF64[z ^ count_ones(state) for state in full_basis])
    Zfull = tr(Zop * prefix_full[end])
    MBfullT = [manybody_propagator_transpose(B, full_basis, V) for B in Bup]
    prefix_fullT = Vector{Matrix{ComplexF64}}(undef, system.L)
    prefix_fullT[1] = MBfullT[1]
    for l in 2:system.L
        prefix_fullT[l] = MBfullT[l] * prefix_fullT[l - 1]
    end
    ZfullT = tr(Zop * prefix_fullT[end])
    Gfull_exact_T = zeros(ComplexF64, V, V)
    for a in 1:V, b in 1:V
        Gfull_exact_T[a, b] = tr(Zop * prefix_fullT[end] * Cops[a] * Cdops[b]) / ZfullT
    end
    Gfull_exact = zeros(ComplexF64, V, V)
    for a in 1:V, b in 1:V
        Gfull_exact[a, b] = tr(Zop * prefix_full[end] * Cops[a] * Cdops[b]) / Zfull
    end
    ltest = 1
    Utest = prefix_full[ltest]
    Vtest = suffix_full[ltest + 1]
    Jfull = bilinear_manybody_matrix(current_operator_x(system, qx=qmin, qy=0.0), full_basis)
    JfullT = bilinear_manybody_matrix(transpose(current_operator_x(system, qx=qmin, qy=0.0)), full_basis)
    Jmat = current_operator_x(system, qx=qmin, qy=0.0)
    G00_exact = zeros(ComplexF64, V, V)
    Gττ_exact = zeros(ComplexF64, V, V)
    Gτ0_exact = zeros(ComplexF64, V, V)
    G0τ_exact = zeros(ComplexF64, V, V)
    M_VU = zeros(ComplexF64, V, V)
    M_VCU = zeros(ComplexF64, V, V)
    M_CVU = zeros(ComplexF64, V, V)
    M_VUCdC = zeros(ComplexF64, V, V)
    for a in 1:V, b in 1:V
        G00_exact[a, b] = tr(Zop * Vtest * Utest * Cops[a] * Cdops[b]) / Zfull
        Gττ_exact[a, b] = tr(Zop * Vtest * Cops[a] * Cdops[b] * Utest) / Zfull
        Gτ0_exact[a, b] = tr(Zop * Vtest * Cops[a] * Utest * Cdops[b]) / Zfull
        G0τ_exact[a, b] = tr(Zop * Cops[a] * Vtest * Cdops[b] * Utest) / Zfull
        M_VU[a, b] = tr(Zop * Vtest * Utest * Cops[a] * Cdops[b]) / Zfull
        M_VCU[a, b] = tr(Zop * Vtest * Cops[a] * Utest * Cdops[b]) / Zfull
        M_CVU[a, b] = tr(Zop * Cops[a] * Vtest * Utest * Cdops[b]) / Zfull
        M_VUCdC[a, b] = tr(Zop * Vtest * Utest * Cdops[b] * Cops[a]) / Zfull
    end
    suffix_scaled = ldr(Icomplex)
    Gττ_test = zeros(ComplexF64, V, V)
    G00_test = zeros(ComplexF64, V, V)
    Gτ0_test = zeros(ComplexF64, V, V)
    G0τ_test = zeros(ComplexF64, V, V)
    scale_factorization!(suffix_scaled, suffix_up[ltest + 1], z, ws, tmp)
    inv_IpUV!(Gττ_test, prefix_up[ltest + 1], suffix_scaled, ws)
    inv_IpUV!(G00_test, suffix_scaled, prefix_up[ltest + 1], ws)
    inv_invUpV!(Gτ0_test, prefix_up[ltest + 1], suffix_scaled, ws)
    inv_invUpV!(G0τ_test, suffix_scaled, prefix_up[ltest + 1], ws)
    @. G0τ_test = -G0τ_test
    println("G00_diff_l1 = ", maximum(abs.(G00_exact .- G00_test)))
    println("G00_crossdiff_l1 = ", maximum(abs.(G00_exact .- Gττ_test)))
    println("G00_transpose_diff_l1 = ", maximum(abs.(G00_exact .- transpose(G00_test))))
    println("G00_vs_full_diff_l1 = ", maximum(abs.(G00_test .- Gfull)))
    println("Gττ_diff_l1 = ", maximum(abs.(Gττ_exact .- Gττ_test)))
    println("Gττ_crossdiff_l1 = ", maximum(abs.(Gττ_exact .- G00_test)))
    println("Gττ_transpose_diff_l1 = ", maximum(abs.(Gττ_exact .- transpose(Gττ_test))))
    println("Gττ_vs_full_diff_l1 = ", maximum(abs.(Gττ_test .- Gfull)))
    println("Gτ0_diff_l1 = ", maximum(abs.(Gτ0_exact .- Gτ0_test)))
    println("G0τ_diff_l1 = ", maximum(abs.(G0τ_exact .- G0τ_test)))
    println("Gfull_exact_diff = ", maximum(abs.(Gfull_exact .- Gfull)))
    println("Gfull_exact_T_diff = ", maximum(abs.(Gfull_exact_T .- Gfull)))
    println("match_G00_VU = ", maximum(abs.(M_VU .- G00_test)))
    println("match_Gτ0_VCU = ", maximum(abs.(M_VCU .- Gτ0_test)))
    println("match_G0τ_CVU = ", maximum(abs.(M_CVU .+ G0τ_test)))
    println("match_G00_CdC = ", maximum(abs.(M_VUCdC .- (Icomplex .- G00_test))))
    println("current_from_exact_G_l1 = ", current_corr_pair_q(system, Gττ_exact, G00_exact, Gτ0₁=Gτ0_exact, G0τ₁=G0τ_exact, qx=qmin, qy=0.0, same_spin=true))
    println("current_from_test_G_l1 = ", current_corr_pair_q(system, Gττ_test, G00_test, Gτ0₁=Gτ0_test, G0τ₁=G0τ_test, qx=qmin, qy=0.0, same_spin=true))
    println("bilinear_general_from_test_G_l1 = ", bilinear_corr_general(Jmat, current_operator_x(system, qx=-qmin, qy=0.0), Gττ_test, G00_test, Gτ0_test, G0τ_test))
    Udense = zeros(ComplexF64, V, V)
    Vdense = zeros(ComplexF64, V, V)
    copyto!(Udense, prefix_up[ltest + 1], ldr_workspace(Icomplex))
    copyto!(Vdense, suffix_scaled, ldr_workspace(Icomplex))
    println("trace_formula_l1 = ", bilinear_corr_trace_formula(Udense, Vdense, Jmat, current_operator_x(system, qx=-qmin, qy=0.0)))
    println("trace_formula_fd_l1 = ", finite_difference_trace_corr(Udense, Vdense, Jmat, current_operator_x(system, qx=-qmin, qy=0.0), full_basis, z))
    println("det_fd_l1 = ", finite_difference_det_corr(Udense, Vdense, Jmat, current_operator_x(system, qx=-qmin, qy=0.0), z))
    println("candidate_VJUJ = ", tr(Zop * Vtest * Jfull * Utest * Jfull) / Zfull / V)
    println("candidate_VJUJ_T = ", tr(Zop * Vtest * JfullT * Utest * JfullT) / Zfull / V)
    println("candidate_VUJJ = ", tr(Zop * Vtest * Utest * Jfull * Jfull) / Zfull / V)
    println("candidate_JUVJ = ", tr(Zop * Jfull * Utest * Vtest * Jfull) / Zfull / V)
end

main()
