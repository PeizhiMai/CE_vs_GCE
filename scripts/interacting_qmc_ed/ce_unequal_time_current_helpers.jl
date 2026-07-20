using LinearAlgebra
using CanEnsAFQMC

@inline function ce_site_index_2d(x::Int, y::Int, lx::Int)
    return x + 1 + y * lx
end

@inline function ce_periodic_site(x::Int, y::Int, lx::Int, ly::Int)
    return ce_site_index_2d(mod(x, lx), mod(y, ly), lx)
end

function build_B_slices(system, walker)
    V, L = system.V, system.L
    Bup = [zeros(ComplexF64, V, V) for _ in 1:L]
    Bdn = [zeros(ComplexF64, V, V) for _ in 1:L]
    tmp = zeros(ComplexF64, V, V)
    for l in 1:L
        σ = @view walker.auxfield[:, l]
        CanEnsAFQMC.imagtime_propagator!(Bup[l], Bdn[l], σ, system, tmpmat=tmp)
    end
    return Bup, Bdn
end

function build_prefix_suffix(Bseq::Vector{<:AbstractMatrix})
    V = size(Bseq[1], 1)
    L = length(Bseq)
    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    prefix = ldrs(Icomplex, L + 1)
    suffix = ldrs(Icomplex, L + 1)

    copyto!(prefix[1], I, ws)
    for l in 1:L
        copyto!(prefix[l + 1], prefix[l])
        lmul!(Bseq[l], prefix[l + 1], ws)
    end

    copyto!(suffix[L + 1], I, ws)
    for l in L:-1:1
        copyto!(suffix[l], suffix[l + 1])
        rmul!(suffix[l], Bseq[l], ws)
    end

    return prefix, suffix
end

function ce_hs_diagonal_slices(system, walker, spin::Int)
    V, L = system.V, system.L
    dseq = [zeros(ComplexF64, V) for _ in 1:L]
    aux = system.auxfield
    @inbounds for l in 1:L
        σ = @view walker.auxfield[:, l]
        d = dseq[l]
        for i in 1:V
            if system.useChargeHST
                idx = isone(σ[i]) ? 1 : 2
            elseif spin == 1
                idx = isone(σ[i]) ? 1 : 2
            else
                idx = isone(σ[i]) ? 2 : 1
            end
            d[i] = ComplexF64(aux[idx])
        end
    end
    return dseq
end

function ce_circulant_kinetic_1d(n::Int, α::Float64)
    K = zeros(ComplexF64, n, n)
    @inbounds for x in 0:(n - 1), xp in 0:(n - 1)
        s = zero(ComplexF64)
        for m in 0:(n - 1)
            k = 2π * m / n
            s += exp(α * cos(k)) * cis(k * (x - xp))
        end
        K[x + 1, xp + 1] = s / n
    end
    return K
end

function ce_structured_kinetic_factors(system; inverse::Bool=false)
    lx, ly, lz = system.Ns
    lz == 1 || error("structured kinetic factors require a 2D Hubbard lattice")
    system.useFirstOrderTrotter && error("structured current accumulator currently requires symmetric/second-order Trotter")
    Δτ = system.β / system.L
    dτ = system.useFirstOrderTrotter ? Δτ : Δτ / 2
    src = ce_site_index_2d(0, 0, lx)
    dstx = ce_periodic_site(1, 0, lx, ly)
    dsty = ce_periodic_site(0, 1, lx, ly)
    tx = -system.T[dstx, src]
    ty = -system.T[dsty, src]
    abs(imag(tx)) < 1e-12 || error("structured current accumulator expects real nearest-neighbor hopping; got tx=$(tx)")
    abs(imag(ty)) < 1e-12 || error("structured current accumulator expects real nearest-neighbor hopping; got ty=$(ty)")
    αx = 2 * real(tx) * dτ
    αy = 2 * real(ty) * dτ
    if inverse
        αx = -αx
        αy = -αy
    end
    return ce_circulant_kinetic_1d(lx, αx), ce_circulant_kinetic_1d(ly, αy)
end

function apply_kinetic_left_separable!(
    out::AbstractMatrix{ComplexF64},
    M::AbstractMatrix{ComplexF64},
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64},
    tmp::AbstractMatrix{ComplexF64},
    lx::Int,
    ly::Int,
)
    V = lx * ly
    size(M, 1) == V || error("left kinetic input row dimension mismatch")
    ncols = size(M, 2)
    @inbounds for c in 1:ncols
        # tmp = Kx * M along x at fixed y.
        for y in 0:(ly - 1), x in 0:(lx - 1)
            acc = zero(ComplexF64)
            row = x + 1 + y * lx
            for xp in 0:(lx - 1)
                acc += Kx[x + 1, xp + 1] * M[xp + 1 + y * lx, c]
            end
            tmp[row, c] = acc
        end
        # out = Ky * tmp along y at fixed x.
        for y in 0:(ly - 1), x in 0:(lx - 1)
            acc = zero(ComplexF64)
            row = x + 1 + y * lx
            for yp in 0:(ly - 1)
                acc += Ky[y + 1, yp + 1] * tmp[x + 1 + yp * lx, c]
            end
            out[row, c] = acc
        end
    end
    return out
end

function apply_kinetic_right_separable!(
    out::AbstractMatrix{ComplexF64},
    M::AbstractMatrix{ComplexF64},
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64},
    tmp::AbstractMatrix{ComplexF64},
    lx::Int,
    ly::Int,
)
    V = lx * ly
    size(M, 2) == V || error("right kinetic input column dimension mismatch")
    nrows = size(M, 1)
    @inbounds for r in 1:nrows
        # tmp = M * Kx along x at fixed y.
        for y in 0:(ly - 1), x in 0:(lx - 1)
            acc = zero(ComplexF64)
            col = x + 1 + y * lx
            for xp in 0:(lx - 1)
                acc += M[r, xp + 1 + y * lx] * Kx[xp + 1, x + 1]
            end
            tmp[r, col] = acc
        end
        # out = tmp * Ky along y at fixed x.
        for y in 0:(ly - 1), x in 0:(lx - 1)
            acc = zero(ComplexF64)
            col = x + 1 + y * lx
            for yp in 0:(ly - 1)
                acc += tmp[r, x + 1 + yp * lx] * Ky[yp + 1, y + 1]
            end
            out[r, col] = acc
        end
    end
    return out
end

function apply_B_left_structured!(
    out::AbstractMatrix{ComplexF64},
    M::AbstractMatrix{ComplexF64},
    d::AbstractVector{ComplexF64},
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64},
    work1::AbstractMatrix{ComplexF64},
    work2::AbstractMatrix{ComplexF64},
    lx::Int,
    ly::Int,
)
    apply_kinetic_left_separable!(work1, M, Kx, Ky, work2, lx, ly)
    @inbounds for c in axes(work1, 2), i in eachindex(d)
        work1[i, c] *= d[i]
    end
    apply_kinetic_left_separable!(out, work1, Kx, Ky, work2, lx, ly)
    return out
end

function apply_B_right_structured!(
    out::AbstractMatrix{ComplexF64},
    M::AbstractMatrix{ComplexF64},
    d::AbstractVector{ComplexF64},
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64},
    work1::AbstractMatrix{ComplexF64},
    work2::AbstractMatrix{ComplexF64},
    lx::Int,
    ly::Int,
)
    apply_kinetic_right_separable!(work1, M, Kx, Ky, work2, lx, ly)
    @inbounds for j in eachindex(d), r in axes(work1, 1)
        work1[r, j] *= d[j]
    end
    apply_kinetic_right_separable!(out, work1, Kx, Ky, work2, lx, ly)
    return out
end

function structured_similarity_forward!(
    out::AbstractMatrix{ComplexF64},
    M::AbstractMatrix{ComplexF64},
    d::AbstractVector{ComplexF64},
    dinv::AbstractVector{ComplexF64},
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64},
    Kxinv::AbstractMatrix{ComplexF64},
    Kyinv::AbstractMatrix{ComplexF64},
    work1::AbstractMatrix{ComplexF64},
    work2::AbstractMatrix{ComplexF64},
    work3::AbstractMatrix{ComplexF64},
    lx::Int,
    ly::Int,
)
    # out = (Bk D Bk) * M * (Bk⁻¹ D⁻¹ Bk⁻¹)
    apply_B_left_structured!(work1, M, d, Kx, Ky, work2, work3, lx, ly)
    apply_B_right_structured!(out, work1, dinv, Kxinv, Kyinv, work2, work3, lx, ly)
    return out
end

function structured_similarity_backward!(
    out::AbstractMatrix{ComplexF64},
    M::AbstractMatrix{ComplexF64},
    d::AbstractVector{ComplexF64},
    dinv::AbstractVector{ComplexF64},
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64},
    Kxinv::AbstractMatrix{ComplexF64},
    Kyinv::AbstractMatrix{ComplexF64},
    work1::AbstractMatrix{ComplexF64},
    work2::AbstractMatrix{ComplexF64},
    work3::AbstractMatrix{ComplexF64},
    lx::Int,
    ly::Int,
)
    # out = (Bk⁻¹ D⁻¹ Bk⁻¹) * M * (Bk D Bk)
    apply_B_left_structured!(work1, M, dinv, Kxinv, Kyinv, work2, work3, lx, ly)
    apply_B_right_structured!(out, work1, d, Kx, Ky, work2, work3, lx, ly)
    return out
end


"""
    ce_current_operator_x(system; qx=0, qy=0)

Local current-operator constructor used by the benchmark helpers.  It mirrors
the newer `CanEnsAFQMC.current_operator_x` API, but is defined here so the
benchmark also runs on CADES environments whose installed CanEnsAFQMC does not
yet export that helper.
"""
function ce_current_operator_x(system; qx::Float64=0.0, qy::Float64=0.0)
    lx, ly, lz = system.Ns
    lz == 1 || error("ce_current_operator_x assumes a 2D lattice with Ns=(Lx,Ly,1)")
    J = zeros(ComplexF64, system.V, system.V)
    @inbounds for y in 0:(ly - 1), x in 0:(lx - 1)
        src = ce_site_index_2d(x, y, lx)
        dst = ce_site_index_2d(mod(x + 1, lx), y, lx)
        phase = cis(qx * x + qy * y)
        t_forward = -system.T[dst, src]
        t_backward = -system.T[src, dst]
        J[dst, src] += 1im * t_forward * phase
        J[src, dst] += -1im * t_backward * phase
    end
    return J
end

function ce_current_operator_x_entries(system; qx::Float64=0.0, qy::Float64=0.0)
    lx, ly, lz = system.Ns
    lz == 1 || error("ce_current_operator_x_entries assumes a 2D lattice with Ns=(Lx,Ly,1)")
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    sizehint!(rows, 2 * system.V)
    sizehint!(cols, 2 * system.V)
    sizehint!(vals, 2 * system.V)
    @inbounds for y in 0:(ly - 1), x in 0:(lx - 1)
        src = ce_site_index_2d(x, y, lx)
        dst = ce_site_index_2d(mod(x + 1, lx), y, lx)
        phase = cis(qx * x + qy * y)
        t_forward = -system.T[dst, src]
        t_backward = -system.T[src, dst]
        push!(rows, dst)
        push!(cols, src)
        push!(vals, 1im * t_forward * phase)
        push!(rows, src)
        push!(cols, dst)
        push!(vals, -1im * t_backward * phase)
    end
    return (rows=rows, cols=cols, vals=vals)
end

function ce_kinetic_operator_x(system)
    lx, ly, lz = system.Ns
    lz == 1 || error("ce_kinetic_operator_x assumes a 2D lattice with Ns=(Lx,Ly,1)")
    K = zeros(ComplexF64, system.V, system.V)
    @inbounds for y in 0:(ly - 1), x in 0:(lx - 1)
        src = ce_site_index_2d(x, y, lx)
        dst = ce_site_index_2d(mod(x + 1, lx), y, lx)
        K[dst, src] += system.T[dst, src]
        K[src, dst] += system.T[src, dst]
    end
    return K
end

function ce_measure_bilinear_density(ρ::DensityMatrix, A::AbstractMatrix)
    return sum(A .* transpose(ρ.ρ₁))
end

function ce_measure_KxPerSite(system, ρup::DensityMatrix, ρdn::DensityMatrix)
    Kx = ce_kinetic_operator_x(system)
    return (ce_measure_bilinear_density(ρup, Kx) + ce_measure_bilinear_density(ρdn, Kx)) / system.V
end

function ce_measure_Energy(system, ρup::DensityMatrix, ρdn::DensityMatrix)
    T = system.T
    ρu = ρup.ρ₁
    ρd = ρdn.ρ₁
    kinetic = zero(ComplexF64)
    @inbounds for idx in eachindex(T)
        if T[idx] != 0
            kinetic += T[idx] * (ρu[idx] + ρd[idx])
        end
    end
    potential = zero(ComplexF64)
    @inbounds for i in 1:system.V
        potential += system.U * ρu[i, i] * ρd[i, i]
    end
    return ComplexF64[kinetic, potential, kinetic + potential]
end

function ce_recompute_density_matrix_canonical!(system, ρ::DensityMatrix, spin::Int)
    # CanEnsAFQMC's conjugate-spin density-matrix shortcut is valid for the
    # balanced spin-HS case, but the 3x3 validation suite also uses (Nup,Ndn)
    # = (5,4).  Recompute the canonical Fourier weights directly from the
    # eigenspectrum and the requested fixed spin particle number so both
    # balanced and imbalanced sectors use the same fixed-N projection.
    N = system.N[spin]
    t = ρ.t[]
    λ = ComplexF64.(collect(@view ρ.λ[t]))
    μ = CanEnsAFQMC.fermilevel(λ, N)
    λscaled = ComplexF64.(λ ./ μ)
    logZ = canonical_log_coefficient_scaled(λscaled, N)
    @inbounds for m in 1:ρ.Nft
        ρ.expiφμ[m] = ρ.expiφ[m] / μ
        logdet = zero(ComplexF64)
        for λα in λscaled
            logdet += log(1 + ρ.expiφ[m] * λα)
        end
        ρ.Z̃ₘ[m] = exp(logdet - N * ρ.iφ[m] - logZ)
    end
    CanEnsAFQMC.compute_RDM(ρ, ρ.ws)
    ρ₁ = ρ.ρ₁
    Gₘ = ρ.Gₘ
    @inbounds for i in eachindex(IndexCartesian(), ρ₁)
        tmp = sum(Gₘ[:, i[1], i[2]] .* ρ.Z̃ₘ) / ρ.Nft
        ρ₁[i] = eltype(ρ₁) <: Real ? real(tmp) : tmp
    end
    return ρ
end

function ce_update_density_matrices!(system, walker, ρup::DensityMatrix, ρdn::DensityMatrix)
    update!(system, walker, ρup, 1)
    ce_recompute_density_matrix_canonical!(system, ρup, 1)
    if system.U > 0 && !system.useChargeHST
        # Repulsive spin-HS fields couple with opposite signs to the two spin
        # sectors. Even for Nup=Ndn, their one-body propagators and canonical
        # density matrices are configuration-by-configuration distinct.
        update!(system, walker, ρdn, 2)
    elseif system.N[1] == system.N[2]
        update!(system, walker, ρdn, ρup)
    else
        update!(system, walker, ρdn, 2)
    end
    ce_recompute_density_matrix_canonical!(system, ρdn, 2)
    return nothing
end

function scale_factorization!(
    Fscaled::LDR{ComplexF64},
    Fsrc::LDR{ComplexF64},
    z::ComplexF64,
    ws::LDRWorkspace{ComplexF64},
    tmp::AbstractMatrix{ComplexF64},
)
    copyto!(tmp, Fsrc, ws)
    @. tmp = z * tmp
    copyto!(Fscaled, tmp, ws)
    return nothing
end

function current_expect_from_G(A::AbstractMatrix{ComplexF64}, G::AbstractMatrix{ComplexF64})
    V = size(G, 1)
    ρ = Matrix{ComplexF64}(I, V, V) .- G
    return sum(A .* transpose(ρ))
end

function current_expect_from_entries(J, G::AbstractMatrix{ComplexF64})
    rows, cols, vals = J.rows, J.cols, J.vals
    out = zero(ComplexF64)
    @inbounds for p in eachindex(vals)
        i = rows[p]
        j = cols[p]
        out += vals[p] * ((i == j ? one(ComplexF64) : zero(ComplexF64)) - G[j, i])
    end
    return out
end

function bilinear_weight_from_entries(J, W::AbstractMatrix{ComplexF64})
    rows, cols, vals = J.rows, J.cols, J.vals
    out = zero(ComplexF64)
    @inbounds for p in eachindex(vals)
        out += vals[p] * W[rows[p], cols[p]]
    end
    return out
end

function trace_bilinear_from_entries(J, M::AbstractMatrix{ComplexF64})
    rows, cols, vals = J.rows, J.cols, J.vals
    out = zero(ComplexF64)
    @inbounds for p in eachindex(vals)
        out += vals[p] * M[cols[p], rows[p]]
    end
    return out
end

function mul_current_entries!(Y::AbstractMatrix{ComplexF64}, J, X::AbstractMatrix{ComplexF64})
    fill!(Y, 0)
    rows, cols, vals = J.rows, J.cols, J.vals
    ncols = size(X, 2)
    @inbounds for p in eachindex(vals)
        i = rows[p]
        j = cols[p]
        v = vals[p]
        for c in 1:ncols
            Y[i, c] += v * X[j, c]
        end
    end
    return Y
end

function current_corr_from_matrices(
    system,
    Jq::AbstractMatrix{ComplexF64},
    Jm::AbstractMatrix{ComplexF64},
    Gττ₁::AbstractMatrix{ComplexF64},
    G00₂::AbstractMatrix{ComplexF64};
    Gτ0₁::Union{Nothing,AbstractMatrix{ComplexF64}}=nothing,
    G0τ₁::Union{Nothing,AbstractMatrix{ComplexF64}}=nothing,
    same_spin::Bool,
)
    disc = current_expect_from_G(Jq, Gττ₁) * current_expect_from_G(Jm, G00₂)
    if same_spin
        @assert Gτ0₁ !== nothing && G0τ₁ !== nothing
        conn = tr(Jq * Gτ0₁ * Jm * G0τ₁)
        return (disc - conn) / system.V
    end
    return disc / system.V
end

function current_corr_from_entries(
    system,
    Jq,
    Jm,
    Gττ₁::AbstractMatrix{ComplexF64},
    G00₂::AbstractMatrix{ComplexF64};
    Gτ0₁::Union{Nothing,AbstractMatrix{ComplexF64}}=nothing,
    G0τ₁::Union{Nothing,AbstractMatrix{ComplexF64}}=nothing,
    same_spin::Bool,
)
    disc = current_expect_from_entries(Jq, Gττ₁) * current_expect_from_entries(Jm, G00₂)
    if same_spin
        @assert Gτ0₁ !== nothing && G0τ₁ !== nothing
        rows_q, cols_q, vals_q = Jq.rows, Jq.cols, Jq.vals
        rows_m, cols_m, vals_m = Jm.rows, Jm.cols, Jm.vals
        conn = zero(ComplexF64)
        @inbounds for p in eachindex(vals_q)
            i = rows_q[p]
            j = cols_q[p]
            vq = vals_q[p]
            for r in eachindex(vals_m)
                k = rows_m[r]
                l = cols_m[r]
                conn += vq * Gτ0₁[j, k] * vals_m[r] * G0τ₁[l, i]
            end
        end
        return (disc - conn) / system.V
    end
    return disc / system.V
end

function canonical_occ_paircorr_direct(λ_in::AbstractVector{<:Number}, N::Int)
    Ns = length(λ_in)
    n = zeros(ComplexF64, Ns)
    ninj = zeros(ComplexF64, Ns, Ns)
    if N <= 0
        return n, ninj
    elseif N >= Ns
        fill!(n, 1)
        fill!(ninj, 1)
        return n, ninj
    end

    # Rescale by an approximate Fermi level.  All fixed-N ratios below are
    # invariant under λ -> λ / μ, but the recursion is much better conditioned.
    μ = CanEnsAFQMC.fermilevel(λ_in, N)
    λ = ComplexF64.(λ_in ./ μ)

    coeff = zeros(ComplexF64, N + 1)
    coeff[1] = 1
    @inbounds for λα in λ
        for k in N:-1:1
            coeff[k + 1] += λα * coeff[k]
        end
    end
    Z = coeff[N + 1]
    abs(Z) > 0 || error("canonical partition coefficient is zero for N=$N")

    poly_ex_i = zeros(ComplexF64, N + 1)
    pair_tol = 1e-10
    @inbounds for i in 1:Ns
        fill!(poly_ex_i, 0)
        poly_ex_i[1] = 1
        for a in 1:Ns
            a == i && continue
            λa = λ[a]
            for k in N:-1:1
                poly_ex_i[k + 1] += λa * poly_ex_i[k]
            end
        end

        n[i] = λ[i] * poly_ex_i[N] / Z
        ninj[i, i] = n[i]
    end
    N == 1 && return n, ninj

    # Pair occupations.  The simple synthetic-division formula for
    # e_{N-2}^{(ij)} is badly conditioned for the exactly degenerate free
    # 3x3 benchmark.  Use the stable two-level identity when λ_i != λ_j,
    #
    #   n_ij = (λ_j n_i - λ_i n_j) / (λ_j - λ_i),
    #
    # and fall back to a direct coefficient recursion only for nearly
    # degenerate pairs.
    coeff_ex_ij = zeros(ComplexF64, max(N - 1, 1))
    @inbounds for i in 1:Ns
        for j in 1:Ns
            i == j && continue
            denom = λ[j] - λ[i]
            scale = max(abs(λ[i]), abs(λ[j]), 1.0)
            if abs(denom) > pair_tol * scale
                ninj[i, j] = (λ[j] * n[i] - λ[i] * n[j]) / denom
            else
                fill!(coeff_ex_ij, 0)
                coeff_ex_ij[1] = 1
                for a in 1:Ns
                    (a == i || a == j) && continue
                    λa = λ[a]
                    for k in (N - 2):-1:1
                        coeff_ex_ij[k + 1] += λa * coeff_ex_ij[k]
                    end
                end
                ninj[i, j] = λ[i] * λ[j] * coeff_ex_ij[N - 1] / Z
            end
        end
    end

    return n, ninj
end

function canonical_bilinear_product_eig(
    Ã::AbstractMatrix{ComplexF64},
    B̃::AbstractMatrix{ComplexF64},
    n::AbstractVector{ComplexF64},
    ninj::AbstractMatrix{ComplexF64},
)
    Ns = length(n)
    out = zero(ComplexF64)
    @inbounds for i in 1:Ns, k in 1:Ns
        out += Ã[i, i] * B̃[k, k] * ninj[i, k]
    end
    @inbounds for i in 1:Ns, j in 1:Ns
        i == j && continue
        out += Ã[i, j] * B̃[j, i] * (n[i] - ninj[i, j])
    end
    return out
end

function canonical_bilinear_product_weight!(
    W::AbstractMatrix{ComplexF64},
    B̃::AbstractMatrix{ComplexF64},
    n::AbstractVector{ComplexF64},
    ninj::AbstractMatrix{ComplexF64},
)
    Ns = length(n)
    fill!(W, 0)
    @inbounds for i in 1:Ns
        diag_weight = zero(ComplexF64)
        for k in 1:Ns
            diag_weight += B̃[k, k] * ninj[i, k]
        end
        W[i, i] = diag_weight
        for j in 1:Ns
            i == j && continue
            W[i, j] = B̃[j, i] * (n[i] - ninj[i, j])
        end
    end
    return W
end

function canonical_bilinear_product_second_weight!(
    W::AbstractMatrix{ComplexF64},
    Ã::AbstractMatrix{ComplexF64},
    n::AbstractVector{ComplexF64},
    ninj::AbstractMatrix{ComplexF64},
)
    Ns = length(n)
    fill!(W, 0)
    @inbounds for k in 1:Ns
        diag_weight = zero(ComplexF64)
        for i in 1:Ns
            diag_weight += Ã[i, i] * ninj[i, k]
        end
        W[k, k] = diag_weight
    end
    @inbounds for a in 1:Ns, b in 1:Ns
        a == b && continue
        W[a, b] = Ã[b, a] * (n[b] - ninj[b, a])
    end
    return W
end

function canonical_bilinear_product_weight(
    B̃::AbstractMatrix{ComplexF64},
    n::AbstractVector{ComplexF64},
    ninj::AbstractMatrix{ComplexF64},
)
    return canonical_bilinear_product_weight!(
        zeros(ComplexF64, length(n), length(n)),
        B̃, n, ninj,
    )
end

function canonical_bilinear_product_from_weight(
    Ã::AbstractMatrix{ComplexF64},
    W::AbstractMatrix{ComplexF64},
)
    out = zero(ComplexF64)
    @inbounds for idx in eachindex(Ã, W)
        out += Ã[idx] * W[idx]
    end
    return out
end

function canonical_bilinear_expect_eig(
    Ã::AbstractMatrix{ComplexF64},
    n::AbstractVector{ComplexF64},
)
    out = zero(ComplexF64)
    @inbounds for i in eachindex(n)
        out += Ã[i, i] * n[i]
    end
    return out
end

function finite_complex_vector(v::AbstractVector{<:Complex})
    @inbounds for z in v
        (isfinite(real(z)) && isfinite(imag(z))) || return false
    end
    return true
end

function canonical_recursion_check_slices(L::Int)
    L <= 1 && return Int[]
    return unique(sort(filter(l -> 1 <= l <= L - 1, [1, max(1, L ÷ 2), L - 1])))
end

function stable_similarity_residual!(
    residual::AbstractMatrix{ComplexF64},
    Ufac::LDR,
    UinvJQ::AbstractMatrix{ComplexF64},
    JQ::AbstractMatrix{ComplexF64},
    ws::LDRWorkspace,
)
    copyto!(residual, UinvJQ)
    lmul!(Ufac, residual, ws)
    residual .-= JQ
    denom = max(norm(JQ), eps(Float64))
    return norm(residual) / denom
end

function dense_from_factorization(F::LDR{ComplexF64}, ws::LDRWorkspace{ComplexF64})
    V = length(F.d)
    M = zeros(ComplexF64, V, V)
    copyto!(M, F, ws)
    return M
end

"""
    spin_hs_sector_normalization_ratio(system, Δnup, Δndn, nslices)

Return the physical-Hamiltonian normalization factor needed when an unequal-time
estimator propagates through `nslices` time slices in a particle sector shifted by
`(Δnup, Δndn)` relative to the fixed sector sampled by `CanEnsAFQMC`.

For the spin-channel Hirsch decomposition used here (`useChargeHST=false`), the
unnormalized local HS sum satisfies, in a fixed `(Nup,Ndn)` sector,

    HS_sum = const * exp(Δτ U (Nup + Ndn)/2) * exp(-Δτ U D)

per time slice, where `D = Σ_i n_{i↑} n_{i↓}`.  Equal-time fixed-sector
observables are unaffected because this sector-dependent scalar cancels between
numerator and denominator.  Single-particle unequal-time estimators do not stay
in the same sector between the creation and annihilation operators, so the scalar
does not cancel.  This helper converts the raw HS estimator back to the
unshifted Hubbard Hamiltonian used by the ED scripts.

Examples for adding/removing one up electron over imaginary time τ:

    add:    exp(-U * τ / 2)
    remove: exp(+U * τ / 2)

The current-current response is number conserving between operators, so this
factor is 1 for that measurement.
"""
function spin_hs_sector_normalization_ratio(system, Δnup::Int, Δndn::Int, nslices::Int)
    nslices == 0 && return 1.0
    if system.useChargeHST
        if abs(system.U) > 1e-14
            error("physical unequal-time normalization is only implemented for spin HS (useChargeHST=false)")
        end
        return 1.0
    end
    Δτ = system.β / system.L
    return exp(-0.5 * system.U * Δτ * nslices * (Δnup + Δndn))
end

"""
    elementary_symmetric_coefficient(λ, k)

Return the coefficient of `x^k` in `prod(a, 1 + x*λ[a])`.
"""
function elementary_symmetric_coefficient(λ::AbstractVector{<:Number}, k::Int)
    Ns = length(λ)
    (k < 0 || k > Ns) && return zero(ComplexF64)
    coeff = zeros(ComplexF64, k + 1)
    coeff[1] = 1
    @inbounds for λα in λ
        for n in min(k, Ns):-1:1
            coeff[n + 1] += λα * coeff[n]
        end
    end
    return coeff[k + 1]
end

"""
    elementary_symmetric_excluding(λ, k)

Return the vector whose `α`th element is the coefficient of `x^k` in
`prod(γ != α, 1 + x*λ[γ])`.
"""
function elementary_symmetric_excluding(λ::AbstractVector{<:Number}, k::Int)
    Ns = length(λ)
    out = zeros(ComplexF64, Ns)
    (k < 0 || k > Ns - 1) && return out
    @inbounds for α in 1:Ns
        coeff = zeros(ComplexF64, k + 1)
        coeff[1] = 1
        for γ in 1:Ns
            γ == α && continue
            λγ = λ[γ]
            for n in k:-1:1
                coeff[n + 1] += λγ * coeff[n]
            end
        end
        out[α] = coeff[k + 1]
    end
    return out
end

function complex_bigfloat(z::Number)
    return Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))
end

function elementary_symmetric_coefficient_big(λ::AbstractVector{Complex{BigFloat}}, k::Int)
    Ns = length(λ)
    (k < 0 || k > Ns) && return Complex{BigFloat}(0)
    coeff = zeros(Complex{BigFloat}, k + 1)
    coeff[1] = 1
    @inbounds for λα in λ
        for n in k:-1:1
            coeff[n + 1] += λα * coeff[n]
        end
    end
    return coeff[k + 1]
end

function canonical_log_coefficient_scaled(λscaled::AbstractVector{ComplexF64}, N::Int)
    # Use the same stabilized canonical recursion as the Metropolis weights for
    # the measurement normalization.  A direct elementary-symmetric coefficient
    # can overflow, underflow, or suffer cancellation while still returning a
    # nonzero ComplexF64; those errors shift every Fourier-projected equal-time
    # observable coherently for the affected Markov chain.
    abslogZ, sgnlogZ = CanEnsAFQMC.compute_pf_recursion(λscaled, N, isReal=false)
    logZ = ComplexF64(abslogZ) + log(ComplexF64(sgnlogZ))
    if isfinite(real(logZ)) && isfinite(imag(logZ))
        return logZ
    end

    # Rare fallback for extremely ill-conditioned spectra.  The λ -> λ/μ
    # scaling has already been applied by the caller, so this is the log of the
    # scaled coefficient e_N(λscaled).
    return setprecision(BigFloat, 1024) do
        λbig = complex_bigfloat.(λscaled)
        Zbig = elementary_symmetric_coefficient_big(λbig, N)
        abs(Zbig) > 0 || error("canonical partition coefficient is zero for N=$N")
        ComplexF64(log(Zbig))
    end
end

function elementary_symmetric_excluding_big(λ::AbstractVector{Complex{BigFloat}}, k::Int)
    Ns = length(λ)
    out = zeros(Complex{BigFloat}, Ns)
    (k < 0 || k > Ns - 1) && return out
    @inbounds for α in 1:Ns
        coeff = zeros(Complex{BigFloat}, k + 1)
        coeff[1] = 1
        for γ in 1:Ns
            γ == α && continue
            λγ = λ[γ]
            for n in k:-1:1
                coeff[n + 1] += λγ * coeff[n]
            end
        end
        out[α] = coeff[k + 1]
    end
    return out
end

struct CanonicalUnequalTimeGreenCache
    N::Int
    λ::Vector{ComplexF64}
    P::Matrix{ComplexF64}
    P⁻¹::Matrix{ComplexF64}
    add_diag::Vector{ComplexF64}
    rem_diag::Vector{ComplexF64}
    add_coeff::Matrix{ComplexF64}
    rem_coeff::Matrix{ComplexF64}
end

"""
    CanonicalUnequalTimeGreenCache(system, prefix, spin)

Build the canonical-recursion data for unequal-time one-particle Green functions
for one HS configuration and one spin sector.  This mirrors the equal-time APF
procedure in the paper: diagonalize the full one-body propagator `F`, compute
canonical elementary-symmetric coefficients, and reuse them for all τ slices.
"""
function CanonicalUnequalTimeGreenCache(system, prefix::Vector{<:LDR}, spin::Int)
    V = system.V
    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    λ, P, P⁻¹ = eigen(prefix[end], ws)
    λ = ComplexF64.(λ)
    P = ComplexF64.(P)
    P⁻¹ = ComplexF64.(P⁻¹)
    N = system.N[spin]

    # Direct canonical coefficient ratios in the eigenbasis.  We use
    # high-precision arithmetic for the elementary-symmetric ratios because
    # small hole factors E_N^(α)/E_N are later multiplied by large unequal-time
    # propagators near τ≈β.
    add_diag = setprecision(BigFloat, 1024) do
        λbig = complex_bigfloat.(λ)
        ZN = elementary_symmetric_coefficient_big(λbig, N)
        abs(ZN) > 0 || error("canonical partition coefficient is zero for N=$N")
        Eadd = elementary_symmetric_excluding_big(λbig, N)
        ComplexF64.(Eadd ./ ZN)
    end
    # Use the paper's stable occupation recursion for the removal branch when
    # possible; it gives a more accurate τ=0 equal-time limit than direct
    # coefficient division for the very ill-conditioned free 3x3 test.
    n = ComplexF64.(CanEnsAFQMC.compute_occ_recursion(λ, N, isReal=false))
    rem_diag = n ./ λ
    add_coeff = (P * Diagonal(add_diag)) * P⁻¹
    rem_coeff = (P * Diagonal(rem_diag)) * P⁻¹
    return CanonicalUnequalTimeGreenCache(N, λ, P, P⁻¹, add_diag, rem_diag, add_coeff, rem_coeff)
end

"""
    canonical_unequal_time_greens(system, cache, prefix, suffix, slice; ...)

Return `(addition, removal)` matrices for `τ = slice * Δτ`.

The addition branch is

    Tr_N[ Γ(Y) c_i Γ(X) c_j† ] / Z_N

and the removal branch is

    Tr_N[ Γ(Y) c_j† Γ(X) c_i ] / Z_N,

where `X = B(slice*Δτ,0)` and `Y = B(β,slice*Δτ)`.  No grand-canonical
observable is formed; elementary-symmetric coefficients perform the canonical
coefficient extraction directly.
"""
function canonical_unequal_time_greens(
    system,
    cache::CanonicalUnequalTimeGreenCache,
    prefix::Vector{<:LDR},
    suffix::Vector{<:LDR},
    slice::Int;
    physical_normalization::Bool=true,
    spin::Int=1,
)
    V = system.V
    0 <= slice <= system.L || error("slice must satisfy 0 <= slice <= L")
    Xfac = prefix[slice + 1]
    Yfac = suffix[slice + 1]

    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    X = zeros(ComplexF64, V, V)
    Y = zeros(ComplexF64, V, V)
    copyto!(X, Xfac, ws)
    copyto!(Y, Yfac, ws)

    addition = X * cache.add_coeff
    removal = cache.rem_coeff * Y

    if physical_normalization
        Δn = spin == 1 ? (1, 0) : (0, 1)
        addition .*= spin_hs_sector_normalization_ratio(system, Δn[1], Δn[2], slice)
        removal .*= spin_hs_sector_normalization_ratio(system, -Δn[1], -Δn[2], slice)
    end

    return addition, removal
end

function canonical_unequal_time_local_greens(
    system,
    cache::CanonicalUnequalTimeGreenCache,
    prefix::Vector{<:LDR},
    suffix::Vector{<:LDR},
    slice::Int;
    physical_normalization::Bool=true,
    spin::Int=1,
)
    addition, removal = canonical_unequal_time_greens(
        system, cache, prefix, suffix, slice,
        physical_normalization=physical_normalization, spin=spin,
    )
    return tr(addition) / system.V, tr(removal) / system.V
end

"""
    green_realspace_average(system, G)

Return the translationally averaged real-space Green function

    G(r,τ) = (1/V) * sum_j G[j+r, j; τ]

for a 2D lattice.  The returned matrix is indexed as
`Gr[dx+1, dy+1]` with `dx=0:Lx-1`, `dy=0:Ly-1`.
"""
function green_realspace_average(system, G::AbstractMatrix{<:Number})
    lx, ly, lz = system.Ns
    lz == 1 || error("green_realspace_average assumes a 2D lattice with Ns=(Lx,Ly,1)")
    V = lx * ly
    Gr = zeros(ComplexF64, lx, ly)
    @inbounds for dy in 0:(ly - 1), dx in 0:(lx - 1)
        acc = zero(ComplexF64)
        for y in 0:(ly - 1), x in 0:(lx - 1)
            j = ce_periodic_site(x, y, lx, ly)
            i = ce_periodic_site(x + dx, y + dy, lx, ly)
            acc += G[i, j]
        end
        Gr[dx + 1, dy + 1] = acc / V
    end
    return Gr
end

"""
    green_momentum_from_realspace(Gr)

Fourier transform the translationally averaged real-space Green function to

    G(k,τ) = sum_r exp(-i k⋅r) G(r,τ),

with `kx = 2π*nx/Lx` and `ky = 2π*ny/Ly`.  The returned matrix is indexed as
`Gk[nx+1, ny+1]`.
"""
function green_momentum_from_realspace(Gr::AbstractMatrix{<:Number})
    lx, ly = size(Gr)
    Gk = zeros(ComplexF64, lx, ly)
    @inbounds for ny in 0:(ly - 1), nx in 0:(lx - 1)
        kx = 2π * nx / lx
        ky = 2π * ny / ly
        acc = zero(ComplexF64)
        for dy in 0:(ly - 1), dx in 0:(lx - 1)
            acc += cis(-(kx * dx + ky * dy)) * Gr[dx + 1, dy + 1]
        end
        Gk[nx + 1, ny + 1] = acc
    end
    return Gk
end

function green_momentum_average(system, G::AbstractMatrix{<:Number})
    return green_momentum_from_realspace(green_realspace_average(system, G))
end

function current_corr_pair_q(
    system,
    Gττ₁::AbstractMatrix{ComplexF64},
    G00₂::AbstractMatrix{ComplexF64};
    Gτ0₁::Union{Nothing,AbstractMatrix{ComplexF64}}=nothing,
    G0τ₁::Union{Nothing,AbstractMatrix{ComplexF64}}=nothing,
    qx::Float64,
    qy::Float64,
    same_spin::Bool,
)
    # Matrix-form Wick contraction for
    #
    #   V⁻¹ < J_x(q,τ) J_x(-q,0) > .
    #
    # The older implementation expanded this expression bond-by-bond in real
    # space.  That hand expansion is easy to get wrong for finite momentum
    # current operators and, in particular, failed single-configuration checks
    # against the general trace formula for q=(0,qmin).  Keep the public helper
    # name, but evaluate the same object directly from the bilinear matrices.
    Jq = ce_current_operator_x_entries(system, qx=qx, qy=qy)
    Jm = ce_current_operator_x_entries(system, qx=-qx, qy=-qy)
    return current_corr_from_entries(
        system, Jq, Jm, Gττ₁, G00₂;
        Gτ0₁=Gτ0₁, G0τ₁=G0τ₁, same_spin=same_spin,
    )
end

function canonical_same_spin_current_response(
    system,
    ρ::DensityMatrix,
    prefix::Vector{<:LDR},
    suffix::Vector{<:LDR};
    qx::Float64,
    qy::Float64,
    return_expectations::Bool=false,
)
    V = system.V
    L = system.L
    Nft = ρ.Nft
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
    Jq = ce_current_operator_x_entries(system, qx=qx, qy=qy)
    Jm = ce_current_operator_x_entries(system, qx=-qx, qy=-qy)
    jτ_q = zeros(ComplexF64, L)
    j0_m = zeros(ComplexF64, L)

    full = prefix[end]
    @inbounds for m in 1:Nft
        z = ComplexF64(ρ.expiφμ[m])
        w = ComplexF64(ρ.Z̃ₘ[m]) / Nft
        response_m = zero(ComplexF64)

        scale_factorization!(full_scaled, full, z, ws, tmp)
        inv_IpA!(Gττ, full_scaled, ws)
        copyto!(G00, Gττ)
        copyto!(Gτ0, Icomplex)
        @. Gτ0 = Gτ0 - Gττ
        @. G0τ = -Gττ
        response_m += current_corr_from_entries(system, Jq, Jm, Gττ, G00, Gτ0₁=Gτ0, G0τ₁=G0τ, same_spin=true)
        jτ_q[1] += w * current_expect_from_entries(Jq, Gττ)
        j0_m[1] += w * current_expect_from_entries(Jm, G00)

        for l in 1:(L - 1)
            U = prefix[l + 1]
            Vfac = suffix[l + 1]
            scale_factorization!(suffix_scaled, Vfac, z, ws, tmp)
            inv_IpUV!(Gττ, U, suffix_scaled, ws)
            inv_IpUV!(G00, suffix_scaled, U, ws)
            inv_invUpV!(Gτ0, U, suffix_scaled, ws)
            inv_invUpV!(G0τ, suffix_scaled, U, ws)
            @. G0τ = -G0τ
            response_m += current_corr_from_entries(system, Jq, Jm, Gττ, G00, Gτ0₁=Gτ0, G0τ₁=G0τ, same_spin=true)
            jτ_q[l + 1] += w * current_expect_from_entries(Jq, Gττ)
            j0_m[l + 1] += w * current_expect_from_entries(Jm, G00)
        end

        total += Δτ * w * response_m
    end

    if return_expectations
        return (response=total, jτ_q=jτ_q, j0_m=j0_m)
    end
    return total
end

function canonical_same_spin_current_response_fast(
    system,
    ρ::DensityMatrix,
    prefix::Vector{<:LDR};
    qx::Float64,
    qy::Float64,
    spin::Int=1,
    return_expectations::Bool=false,
)
    V = system.V
    L = system.L
    Δτ = system.β / system.L

    t = ρ.t[]
    λ = ComplexF64.(collect(@view ρ.λ[t]))
    P = ComplexF64.(@view ρ.P[:, t])
    P⁻¹ = ComplexF64.(@view ρ.P⁻¹[t, :])
    n, ninj = canonical_occ_paircorr_direct(λ, system.N[spin])

    Jq = ce_current_operator_x_entries(system, qx=qx, qy=qy)
    Jm = ce_current_operator_x_entries(system, qx=-qx, qy=-qy)
    JP = zeros(ComplexF64, V, V)
    mul_current_entries!(JP, Jm, P)
    B̃ = P⁻¹ * JP
    j0 = canonical_bilinear_expect_eig(B̃, n)

    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    U = zeros(ComplexF64, V, V)
    Q = zeros(ComplexF64, V, V)
    JQ = zeros(ComplexF64, V, V)
    Ã = zeros(ComplexF64, V, V)

    total = zero(ComplexF64)
    jτ_q = zeros(ComplexF64, L)
    j0_m = fill(j0, L)

    @inbounds for l in 0:(L - 1)
        if l == 0
            mul_current_entries!(JQ, Jq, P)
            mul!(Ã, P⁻¹, JQ)
        else
            copyto!(U, prefix[l + 1], ws)
            mul!(Q, U, P)
            mul_current_entries!(JQ, Jq, Q)
            Ã .= Q \ JQ
        end
        total += canonical_bilinear_product_eig(Ã, B̃, n, ninj)
        jτ_q[l + 1] = canonical_bilinear_expect_eig(Ã, n)
    end

    response = Δτ * total / system.V
    if return_expectations
        return (response=response, jτ_q=jτ_q, j0_m=j0_m)
    end
    return response
end

function canonical_same_spin_current_responses_fast(
    system,
    ρ::DensityMatrix,
    prefix::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    spin::Int=1,
)
    # Historical API retained for diagnostics.  Production callers should use
    # canonical_same_spin_current_responses_canonical_recursion with B slices so
    # the site-basis weight can be propagated without the ill-conditioned
    # U(τ)^{-1} J U(τ) similarity transform.
    return canonical_same_spin_current_responses_fast_similarity(
        system, ρ, prefix, momenta; spin=spin,
    )
end

function canonical_same_spin_current_responses_fast_similarity(
    system,
    ρ::DensityMatrix,
    prefix::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    spin::Int=1,
    residual_tol::Float64=1e-8,
    check_residuals::Bool=true,
)
    V = system.V
    L = system.L
    Δτ = system.β / system.L
    nq = length(momenta)

    t = ρ.t[]
    λ = ComplexF64.(collect(@view ρ.λ[t]))
    P = ComplexF64.(@view ρ.P[:, t])
    P⁻¹ = ComplexF64.(@view ρ.P⁻¹[t, :])
    n, ninj = canonical_occ_paircorr_direct(λ, system.N[spin])

    Jq = [ce_current_operator_x_entries(system, qx=qx, qy=qy) for (qx, qy) in momenta]
    Jm = [ce_current_operator_x_entries(system, qx=-qx, qy=-qy) for (qx, qy) in momenta]
    B̃ = [zeros(ComplexF64, V, V) for _ in 1:nq]
    W = [zeros(ComplexF64, V, V) for _ in 1:nq]
    j0 = zeros(ComplexF64, nq)
    JP = zeros(ComplexF64, V, V)

    @inbounds for iq in 1:nq
        mul_current_entries!(JP, Jm[iq], P)
        mul!(B̃[iq], P⁻¹, JP)
        canonical_bilinear_product_weight!(W[iq], B̃[iq], n, ninj)
        j0[iq] = canonical_bilinear_expect_eig(B̃[iq], n)
    end

    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    Q = zeros(ComplexF64, V, V)
    JQ = zeros(ComplexF64, V, V)
    UinvJQ = zeros(ComplexF64, V, V)
    Ã = zeros(ComplexF64, V, V)
    residual = zeros(ComplexF64, V, V)
    check_slices = check_residuals ? canonical_recursion_check_slices(L) : Int[]
    max_residual = 0.0

    responses = zeros(ComplexF64, nq)
    jτ_q = zeros(ComplexF64, L, nq)
    j0_m = zeros(ComplexF64, L, nq)
    @inbounds for iq in 1:nq
        j0_m[:, iq] .= j0[iq]
    end

    @inbounds for l in 0:(L - 1)
        if l == 0
            for iq in 1:nq
                mul_current_entries!(JQ, Jq[iq], P)
                mul!(Ã, P⁻¹, JQ)
                responses[iq] += canonical_bilinear_product_from_weight(Ã, W[iq])
                jτ_q[l + 1, iq] = canonical_bilinear_expect_eig(Ã, n)
            end
        else
            copyto!(Q, P)
            lmul!(prefix[l + 1], Q, ws)
            do_residual_check = check_residuals && (l in check_slices)
            for iq in 1:nq
                mul_current_entries!(JQ, Jq[iq], Q)
                ldiv!(UinvJQ, prefix[l + 1], JQ, ws)
                if do_residual_check
                    max_residual = max(
                        max_residual,
                        stable_similarity_residual!(residual, prefix[l + 1], UinvJQ, JQ, ws),
                    )
                end
                mul!(Ã, P⁻¹, UinvJQ)
                responses[iq] += canonical_bilinear_product_from_weight(Ã, W[iq])
                jτ_q[l + 1, iq] = canonical_bilinear_expect_eig(Ã, n)
            end
        end
    end

    responses .*= Δτ / system.V
    finite_complex_vector(responses) || error("canonical-recursion current estimator produced non-finite responses: $(responses)")
    if check_residuals && max_residual > residual_tol
        error("canonical-recursion stable-solve residual too large: max_residual=$(max_residual) residual_tol=$(residual_tol)")
    end
    return (responses=responses, jτ_q=jτ_q, j0_m=j0_m)
end

function canonical_same_spin_current_responses_canonical_recursion(
    system,
    ρ::DensityMatrix,
    Bseq::Vector{<:AbstractMatrix},
    prefix::Vector{<:LDR},
    suffix::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    spin::Int=1,
    drift_tol::Float64=1e-2,
)
    V = system.V
    L = system.L
    Δτ = system.β / system.L
    nq = length(momenta)
    length(Bseq) == L || error("Bseq length $(length(Bseq)) does not match system.L=$L")

    t = ρ.t[]
    λ = ComplexF64.(collect(@view ρ.λ[t]))
    P = ComplexF64.(@view ρ.P[:, t])
    P⁻¹ = ComplexF64.(@view ρ.P⁻¹[t, :])
    n, ninj = canonical_occ_paircorr_direct(λ, system.N[spin])

    Jq = [ce_current_operator_x_entries(system, qx=qx, qy=qy) for (qx, qy) in momenta]
    Jm = [ce_current_operator_x_entries(system, qx=-qx, qy=-qy) for (qx, qy) in momenta]

    B̃ = zeros(ComplexF64, V, V)
    W̃ = zeros(ComplexF64, V, V)
    JP = zeros(ComplexF64, V, V)
    Mtmp = zeros(ComplexF64, V, V)
    Mnext = zeros(ComplexF64, V, V)

    # Forward branch, l near 0:
    #
    #   Tr_N[Γ(Y_l) J_q Γ(U_l) J_-q]
    #     = Tr_N[Γ(F) (U_l^{-1} J_q U_l) J_-q].
    #
    # Instead of forming U_l^{-1}J_qU_l, contract the canonical
    # pair-occupation weight in site basis:
    #
    #   product(U_l^{-1}J_qU_l, J_-q)
    #     = tr[J_q (U_l P W_-q^T P^{-1} U_l^{-1})].
    #
    # The matrix in parentheses is propagated by one well-conditioned B slice
    # at a time, eliminating the Nft roots-of-unity loop and the long U_l solve.
    Mresp_forward = [zeros(ComplexF64, V, V) for _ in 1:nq]
    j0 = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        mul_current_entries!(JP, Jm[iq], P)
        mul!(B̃, P⁻¹, JP)
        canonical_bilinear_product_weight!(W̃, B̃, n, ninj)
        mul!(Mtmp, P, transpose(W̃))
        mul!(Mresp_forward[iq], Mtmp, P⁻¹)
        j0[iq] = canonical_bilinear_expect_eig(B̃, n)
    end
    Mone_forward = (P * Diagonal(n)) * P⁻¹

    # Backward/cyclic branch, l near β:
    #
    # Cyclicity gives
    #   Tr Γ(Y_l)J_qΓ(U_l)J_-q
    #     = Tr Γ(U_lY_l) (Y_l^{-1}J_-qY_l) J_q.
    # Choose eigenvectors of U_lY_l as Y_l^{-1}P. Then the first bilinear is
    # P^{-1}J_-qP and only the second bilinear carries the short Y_l
    # similarity.  We build the linear weight for the second bilinear and
    # propagate Y_l^{-1}(... )Y_l backward one B slice at a time.
    Mresp_backward = [zeros(ComplexF64, V, V) for _ in 1:nq]
    @inbounds for iq in 1:nq
        mul_current_entries!(JP, Jm[iq], P)
        mul!(B̃, P⁻¹, JP)
        canonical_bilinear_product_second_weight!(W̃, B̃, n, ninj)
        mul!(Mtmp, P, transpose(W̃))
        mul!(Mresp_backward[iq], Mtmp, P⁻¹)
    end
    Mone_backward = copy(Mone_forward)

    Binv = [inv(Matrix{ComplexF64}(B)) for B in Bseq]
    Bdense = [Matrix{ComplexF64}(B) for B in Bseq]

    responses = zeros(ComplexF64, nq)
    sum_jτ_q = zeros(ComplexF64, nq)

    forward_last = (L - 1) ÷ 2
    @inbounds for l in 0:forward_last
        for iq in 1:nq
            responses[iq] += trace_bilinear_from_entries(Jq[iq], Mresp_forward[iq])
            sum_jτ_q[iq] += trace_bilinear_from_entries(Jq[iq], Mone_forward)
        end
        if l < forward_last
            B = Bdense[l + 1]
            B⁻¹ = Binv[l + 1]
            for iq in 1:nq
                mul!(Mtmp, B, Mresp_forward[iq])
                mul!(Mnext, Mtmp, B⁻¹)
                copyto!(Mresp_forward[iq], Mnext)
            end
            mul!(Mtmp, B, Mone_forward)
            mul!(Mnext, Mtmp, B⁻¹)
            copyto!(Mone_forward, Mnext)
        end
    end

    @inbounds for l in (L - 1):-1:(forward_last + 1)
        B = Bdense[l + 1]
        B⁻¹ = Binv[l + 1]
        for iq in 1:nq
            mul!(Mtmp, B⁻¹, Mresp_backward[iq])
            mul!(Mnext, Mtmp, B)
            copyto!(Mresp_backward[iq], Mnext)
            responses[iq] += trace_bilinear_from_entries(Jq[iq], Mresp_backward[iq])
        end
        mul!(Mtmp, B⁻¹, Mone_backward)
        mul!(Mnext, Mtmp, B)
        copyto!(Mone_backward, Mnext)
        for iq in 1:nq
            sum_jτ_q[iq] += trace_bilinear_from_entries(Jq[iq], Mone_backward)
        end
    end

    # Guardrail: one extra forward step and one extra backward step estimate the
    # same middle slice.  A large mismatch flags numerical drift in the no-Fourier
    # propagation before the sample can contaminate production averages.
    if L > 2 && drift_tol > 0
        B = Bdense[forward_last + 1]
        B⁻¹ = Binv[forward_last + 1]
        Mone_mid_forward = copy(Mone_forward)
        mul!(Mtmp, B, Mone_mid_forward)
        mul!(Mnext, Mtmp, B⁻¹)
        # Rebuild the same middle slice from the β end.
        Mone_mid_backward = copy(Mone_backward)
        # Mone_backward is already at l=forward_last+1 after the backward loop.
        denom = max(norm(Mnext), norm(Mone_mid_backward), eps(Float64))
        drift = norm(Mnext - Mone_mid_backward) / denom
        if !(isfinite(drift)) || drift > drift_tol
            error("canonical-recursion current estimator drift too large: drift=$(drift) drift_tol=$(drift_tol)")
        end
    end

    responses .*= Δτ / system.V
    finite_complex_vector(responses) || error("canonical-recursion current estimator produced non-finite responses: $(responses)")
    return (responses=responses, sum_jτ_q=sum_jτ_q, j0_m=j0)
end

function canonical_same_spin_current_responses_structured_accumulator(
    system,
    ρ::DensityMatrix,
    dseq::Vector{<:AbstractVector{ComplexF64}},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    spin::Int=1,
    drift_tol::Float64=1e-2,
)
    # SmoQy-style fixed-N accumulator:
    #   * no roots-of-unity Fourier projection loop;
    #   * no dense B-slice construction/inversion;
    #   * current contractions are evaluated only on the local x-bond entries;
    #   * one-slice propagation uses the local diagonal HS field plus separable
    #     periodic kinetic factors Bk = By ⊗ Bx, i.e. B = Bk D Bk.
    #
    # This still evolves the exact canonical pair-occupation weights in the
    # fixed-N sector.  It is the production spin-HS path; dense
    # canonical-recursion remains the validation/reference estimator.
    system.useChargeHST && error("structured canonical current accumulator is currently validated only for spin HS")
    V = system.V
    L = system.L
    Δτ = system.β / system.L
    nq = length(momenta)
    length(dseq) == L || error("dseq length $(length(dseq)) does not match system.L=$L")
    lx, ly, lz = system.Ns
    lz == 1 || error("structured canonical current accumulator requires 2D lattice")
    V == lx * ly || error("structured canonical current accumulator site count mismatch")

    t = ρ.t[]
    λ = ComplexF64.(collect(@view ρ.λ[t]))
    P = ComplexF64.(@view ρ.P[:, t])
    P⁻¹ = ComplexF64.(@view ρ.P⁻¹[t, :])
    n, ninj = canonical_occ_paircorr_direct(λ, system.N[spin])

    Jq = [ce_current_operator_x_entries(system, qx=qx, qy=qy) for (qx, qy) in momenta]
    Jm = [ce_current_operator_x_entries(system, qx=-qx, qy=-qy) for (qx, qy) in momenta]

    B̃ = zeros(ComplexF64, V, V)
    W̃ = zeros(ComplexF64, V, V)
    JP = zeros(ComplexF64, V, V)
    Mtmp = zeros(ComplexF64, V, V)
    Mnext = zeros(ComplexF64, V, V)

    Mresp_forward = [zeros(ComplexF64, V, V) for _ in 1:nq]
    j0 = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        mul_current_entries!(JP, Jm[iq], P)
        mul!(B̃, P⁻¹, JP)
        canonical_bilinear_product_weight!(W̃, B̃, n, ninj)
        mul!(Mtmp, P, transpose(W̃))
        mul!(Mresp_forward[iq], Mtmp, P⁻¹)
        j0[iq] = canonical_bilinear_expect_eig(B̃, n)
    end
    Mone_forward = (P * Diagonal(n)) * P⁻¹

    Mresp_backward = [zeros(ComplexF64, V, V) for _ in 1:nq]
    @inbounds for iq in 1:nq
        mul_current_entries!(JP, Jm[iq], P)
        mul!(B̃, P⁻¹, JP)
        canonical_bilinear_product_second_weight!(W̃, B̃, n, ninj)
        mul!(Mtmp, P, transpose(W̃))
        mul!(Mresp_backward[iq], Mtmp, P⁻¹)
    end
    Mone_backward = copy(Mone_forward)

    dinvseq = [ComplexF64.(one(ComplexF64) ./ d) for d in dseq]
    Kx, Ky = ce_structured_kinetic_factors(system)
    Kxinv, Kyinv = ce_structured_kinetic_factors(system; inverse=true)
    work1 = zeros(ComplexF64, V, V)
    work2 = zeros(ComplexF64, V, V)
    work3 = zeros(ComplexF64, V, V)

    responses = zeros(ComplexF64, nq)
    sum_jτ_q = zeros(ComplexF64, nq)

    forward_last = (L - 1) ÷ 2
    @inbounds for l in 0:forward_last
        for iq in 1:nq
            responses[iq] += trace_bilinear_from_entries(Jq[iq], Mresp_forward[iq])
            sum_jτ_q[iq] += trace_bilinear_from_entries(Jq[iq], Mone_forward)
        end
        if l < forward_last
            d = dseq[l + 1]
            dinv = dinvseq[l + 1]
            for iq in 1:nq
                structured_similarity_forward!(
                    Mnext, Mresp_forward[iq], d, dinv,
                    Kx, Ky, Kxinv, Kyinv, work1, work2, work3, lx, ly,
                )
                copyto!(Mresp_forward[iq], Mnext)
            end
            structured_similarity_forward!(
                Mnext, Mone_forward, d, dinv,
                Kx, Ky, Kxinv, Kyinv, work1, work2, work3, lx, ly,
            )
            copyto!(Mone_forward, Mnext)
        end
    end

    @inbounds for l in (L - 1):-1:(forward_last + 1)
        d = dseq[l + 1]
        dinv = dinvseq[l + 1]
        for iq in 1:nq
            structured_similarity_backward!(
                Mnext, Mresp_backward[iq], d, dinv,
                Kx, Ky, Kxinv, Kyinv, work1, work2, work3, lx, ly,
            )
            copyto!(Mresp_backward[iq], Mnext)
            responses[iq] += trace_bilinear_from_entries(Jq[iq], Mresp_backward[iq])
        end
        structured_similarity_backward!(
            Mnext, Mone_backward, d, dinv,
            Kx, Ky, Kxinv, Kyinv, work1, work2, work3, lx, ly,
        )
        copyto!(Mone_backward, Mnext)
        for iq in 1:nq
            sum_jτ_q[iq] += trace_bilinear_from_entries(Jq[iq], Mone_backward)
        end
    end

    if L > 2 && drift_tol > 0
        d = dseq[forward_last + 1]
        dinv = dinvseq[forward_last + 1]
        structured_similarity_forward!(
            Mnext, Mone_forward, d, dinv,
            Kx, Ky, Kxinv, Kyinv, work1, work2, work3, lx, ly,
        )
        denom = max(norm(Mnext), norm(Mone_backward), eps(Float64))
        drift = norm(Mnext - Mone_backward) / denom
        if !(isfinite(drift)) || drift > drift_tol
            error("structured canonical current accumulator drift too large: drift=$(drift) drift_tol=$(drift_tol)")
        end
    end

    responses .*= Δτ / system.V
    finite_complex_vector(responses) || error("structured canonical current accumulator produced non-finite responses: $(responses)")
    return (responses=responses, sum_jτ_q=sum_jτ_q, j0_m=j0)
end

function canonical_same_spin_current_responses_projected(
    system,
    ρ::DensityMatrix,
    prefix::Vector{<:LDR},
    suffix::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}},
)
    V = system.V
    L = system.L
    Nft = ρ.Nft
    Δτ = system.β / system.L
    nq = length(momenta)

    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    tmp = similar(Icomplex)
    full_scaled = ldr(Icomplex)
    suffix_scaled = ldr(Icomplex)
    Gττ = zeros(ComplexF64, V, V)
    G00 = zeros(ComplexF64, V, V)
    Gτ0 = zeros(ComplexF64, V, V)
    G0τ = zeros(ComplexF64, V, V)

    Jq = [ce_current_operator_x_entries(system, qx=qx, qy=qy) for (qx, qy) in momenta]
    Jm = [ce_current_operator_x_entries(system, qx=-qx, qy=-qy) for (qx, qy) in momenta]
    responses = zeros(ComplexF64, nq)
    jτ_q = zeros(ComplexF64, L, nq)
    j0_m = zeros(ComplexF64, L, nq)
    response_m = zeros(ComplexF64, nq)

    full = prefix[end]
    @inbounds for m in 1:Nft
        z = ComplexF64(ρ.expiφμ[m])
        w = ComplexF64(ρ.Z̃ₘ[m]) / Nft
        fill!(response_m, 0)

        scale_factorization!(full_scaled, full, z, ws, tmp)
        inv_IpA!(Gττ, full_scaled, ws)
        copyto!(G00, Gττ)
        copyto!(Gτ0, Icomplex)
        @. Gτ0 = Gτ0 - Gττ
        @. G0τ = -Gττ
        for iq in 1:nq
            response_m[iq] += current_corr_from_entries(
                system, Jq[iq], Jm[iq], Gττ, G00;
                Gτ0₁=Gτ0, G0τ₁=G0τ, same_spin=true,
            )
            jτ_q[1, iq] += w * current_expect_from_entries(Jq[iq], Gττ)
            j0_m[1, iq] += w * current_expect_from_entries(Jm[iq], G00)
        end

        for l in 1:(L - 1)
            U = prefix[l + 1]
            Vfac = suffix[l + 1]
            scale_factorization!(suffix_scaled, Vfac, z, ws, tmp)
            inv_IpUV!(Gττ, U, suffix_scaled, ws)
            inv_IpUV!(G00, suffix_scaled, U, ws)
            inv_invUpV!(Gτ0, U, suffix_scaled, ws)
            inv_invUpV!(G0τ, suffix_scaled, U, ws)
            @. G0τ = -G0τ
            for iq in 1:nq
                response_m[iq] += current_corr_from_entries(
                    system, Jq[iq], Jm[iq], Gττ, G00;
                    Gτ0₁=Gτ0, G0τ₁=G0τ, same_spin=true,
                )
                jτ_q[l + 1, iq] += w * current_expect_from_entries(Jq[iq], Gττ)
                j0_m[l + 1, iq] += w * current_expect_from_entries(Jm[iq], G00)
            end
        end

        @. responses += Δτ * w * response_m
    end

    return (responses=responses, jτ_q=jτ_q, j0_m=j0_m)
end

function refresh_projected_displaced_greens!(
    Gττ::AbstractMatrix{ComplexF64},
    G00::AbstractMatrix{ComplexF64},
    Gτ0::AbstractMatrix{ComplexF64},
    G0τ::AbstractMatrix{ComplexF64},
    suffix_scaled::LDR{ComplexF64},
    U::LDR{ComplexF64},
    Vfac::LDR{ComplexF64},
    z::ComplexF64,
    ws::LDRWorkspace{ComplexF64},
    tmp::AbstractMatrix{ComplexF64},
)
    scale_factorization!(suffix_scaled, Vfac, z, ws, tmp)
    inv_IpUV!(Gττ, U, suffix_scaled, ws)
    inv_IpUV!(G00, suffix_scaled, U, ws)
    inv_invUpV!(Gτ0, U, suffix_scaled, ws)
    inv_invUpV!(G0τ, suffix_scaled, U, ws)
    @. G0τ = -G0τ
    return nothing
end

function measure_projected_slice!(
    responses::AbstractVector{ComplexF64},
    jτ_q::AbstractMatrix{ComplexF64},
    j0_m::AbstractMatrix{ComplexF64},
    slice_index::Int,
    system,
    Jq,
    Jm,
    Gττ::AbstractMatrix{ComplexF64},
    G00::AbstractMatrix{ComplexF64},
    Gτ0::AbstractMatrix{ComplexF64},
    G0τ::AbstractMatrix{ComplexF64},
    w::ComplexF64,
)
    @inbounds for iq in eachindex(Jq)
        responses[iq] += current_corr_from_entries(
            system, Jq[iq], Jm[iq], Gττ, G00;
            Gτ0₁=Gτ0, G0τ₁=G0τ, same_spin=true,
        )
        jτ_q[slice_index, iq] += w * current_expect_from_entries(Jq[iq], Gττ)
        j0_m[slice_index, iq] += w * current_expect_from_entries(Jm[iq], G00)
    end
    return nothing
end

function propagate_displaced_greens_step!(
    Gττ::AbstractMatrix{ComplexF64},
    Gτ0::AbstractMatrix{ComplexF64},
    G0τ::AbstractMatrix{ComplexF64},
    G00::AbstractMatrix{ComplexF64},
    B::AbstractMatrix,
    B⁻¹::AbstractMatrix{ComplexF64},
    Icomplex::AbstractMatrix{ComplexF64},
    tmp::AbstractMatrix{ComplexF64},
    slice_l::Int,
)
    # Advance from slice_l-1 to slice_l, where slice_l is one-based in Bseq.
    mul!(tmp, B, Gττ)
    mul!(Gττ, tmp, B⁻¹)

    if slice_l == 1
        # The τ=0 equal-time discontinuity uses Gτ0(0)=I-G and G0τ(0)=-G,
        # while the l>0 formula starts from
        # Gτ0(1)=B_1 G and G0τ(1)=-(I-G)B_1^{-1}.
        mul!(Gτ0, B, G00)
        @. tmp = G00 - Icomplex
        mul!(G0τ, tmp, B⁻¹)
    else
        mul!(tmp, B, Gτ0)
        copyto!(Gτ0, tmp)
        mul!(tmp, G0τ, B⁻¹)
        copyto!(G0τ, tmp)
    end
    return nothing
end

function propagate_displaced_greens_step_blocked!(
    Gττ::AbstractMatrix{ComplexF64},
    Gτ0::AbstractMatrix{ComplexF64},
    G0τ::AbstractMatrix{ComplexF64},
    G00::AbstractMatrix{ComplexF64},
    B::AbstractMatrix,
    B⁻¹::AbstractMatrix{ComplexF64},
    Icomplex::AbstractMatrix{ComplexF64},
    left_in::AbstractMatrix{ComplexF64},
    left_out::AbstractMatrix{ComplexF64},
    right_in::AbstractMatrix{ComplexF64},
    right_out::AbstractMatrix{ComplexF64},
    slice_l::Int,
)
    # Same algebra as propagate_displaced_greens_step!, but batch the two left
    # multiplications and two right multiplications into wide/tall BLAS calls:
    #
    #   [B*Gττ  B*Gτ0] and [B*Gττ; G0τ] * B^{-1}.
    #
    # For the small/medium dense matrices used by the CE current estimator this
    # cuts the number of GEMM calls in the hot Nft × L propagation loop roughly
    # in half, while preserving the exact propagated estimator and refresh logic.
    V = size(Gττ, 1)
    @views begin
        copyto!(left_in[:, 1:V], Gττ)
        if slice_l == 1
            copyto!(left_in[:, (V + 1):(2V)], G00)
        else
            copyto!(left_in[:, (V + 1):(2V)], Gτ0)
        end
        mul!(left_out, B, left_in)

        copyto!(right_in[1:V, :], left_out[:, 1:V])
        if slice_l == 1
            copyto!(right_in[(V + 1):(2V), :], G00)
            right_in[(V + 1):(2V), :] .-= Icomplex
        else
            copyto!(right_in[(V + 1):(2V), :], G0τ)
        end
        mul!(right_out, right_in, B⁻¹)

        copyto!(Gττ, right_out[1:V, :])
        copyto!(Gτ0, left_out[:, (V + 1):(2V)])
        copyto!(G0τ, right_out[(V + 1):(2V), :])
    end
    return nothing
end

function max_displaced_green_error(
    Aττ::AbstractMatrix{ComplexF64},
    Aτ0::AbstractMatrix{ComplexF64},
    A0τ::AbstractMatrix{ComplexF64},
    Bττ::AbstractMatrix{ComplexF64},
    Bτ0::AbstractMatrix{ComplexF64},
    B0τ::AbstractMatrix{ComplexF64},
)
    err = 0.0
    @inbounds for idx in eachindex(Aττ, Bττ)
        err = max(err, abs(Aττ[idx] - Bττ[idx]))
    end
    @inbounds for idx in eachindex(Aτ0, Bτ0)
        err = max(err, abs(Aτ0[idx] - Bτ0[idx]))
    end
    @inbounds for idx in eachindex(A0τ, B0τ)
        err = max(err, abs(A0τ[idx] - B0τ[idx]))
    end
    return err
end

function canonical_same_spin_current_responses_propagated(
    system,
    ρ::DensityMatrix,
    Bseq::Vector{<:AbstractMatrix},
    prefix::Vector{<:LDR},
    suffix::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    refresh_interval::Int=10,
)
    V = system.V
    L = system.L
    Nft = ρ.Nft
    Δτ = system.β / system.L
    nq = length(momenta)
    refresh_interval < 0 && error("refresh_interval must be nonnegative")

    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    tmp = similar(Icomplex)
    full_scaled = ldr(Icomplex)
    suffix_scaled = ldr(Icomplex)
    Gττ = zeros(ComplexF64, V, V)
    G00 = zeros(ComplexF64, V, V)
    Gτ0 = zeros(ComplexF64, V, V)
    G0τ = zeros(ComplexF64, V, V)
    prop_left_in = zeros(ComplexF64, V, 2V)
    prop_left_out = zeros(ComplexF64, V, 2V)
    prop_right_in = zeros(ComplexF64, 2V, V)
    prop_right_out = zeros(ComplexF64, 2V, V)

    Binv = [inv(Matrix{ComplexF64}(B)) for B in Bseq]
    Jq = [ce_current_operator_x_entries(system, qx=qx, qy=qy) for (qx, qy) in momenta]
    Jm = [ce_current_operator_x_entries(system, qx=-qx, qy=-qy) for (qx, qy) in momenta]
    responses = zeros(ComplexF64, nq)
    jτ_q = zeros(ComplexF64, L, nq)
    j0_m = zeros(ComplexF64, L, nq)
    response_m = zeros(ComplexF64, nq)

    full = prefix[end]
    @inbounds for m in 1:Nft
        z = ComplexF64(ρ.expiφμ[m])
        w = ComplexF64(ρ.Z̃ₘ[m]) / Nft
        fill!(response_m, 0)

        scale_factorization!(full_scaled, full, z, ws, tmp)
        inv_IpA!(Gττ, full_scaled, ws)
        copyto!(G00, Gττ)
        copyto!(Gτ0, Icomplex)
        @. Gτ0 = Gτ0 - Gττ
        @. G0τ = -Gττ
        measure_projected_slice!(
            response_m, jτ_q, j0_m, 1,
            system, Jq, Jm, Gττ, G00, Gτ0, G0τ, w,
        )

        for l in 1:(L - 1)
            if refresh_interval > 0 && (l % refresh_interval == 0)
                refresh_projected_displaced_greens!(
                    Gττ, G00, Gτ0, G0τ,
                    suffix_scaled, prefix[l + 1], suffix[l + 1], z, ws, tmp,
                )
            else
                B = Bseq[l]
                B⁻¹ = Binv[l]
                propagate_displaced_greens_step_blocked!(
                    Gττ, Gτ0, G0τ, G00,
                    B, B⁻¹, Icomplex,
                    prop_left_in, prop_left_out, prop_right_in, prop_right_out,
                    l,
                )
                # G00 = (I + zF)^-1 is independent of the time slice.
            end

            measure_projected_slice!(
                response_m, jτ_q, j0_m, l + 1,
                system, Jq, Jm, Gττ, G00, Gτ0, G0τ, w,
            )
        end

        @. responses += Δτ * w * response_m
    end

    return (responses=responses, jτ_q=jτ_q, j0_m=j0_m)
end

function canonical_same_spin_current_responses_propagated_adaptive(
    system,
    ρ::DensityMatrix,
    Bseq::Vector{<:AbstractMatrix},
    prefix::Vector{<:LDR},
    suffix::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    refresh_interval::Int=10,
    refresh_tol::Float64=1e-6,
    refresh_min::Int=1,
    refresh_max::Int=max(refresh_interval, refresh_min),
    refresh_growth_patience::Int=3,
)
    V = system.V
    L = system.L
    Nft = ρ.Nft
    Δτ = system.β / system.L
    nq = length(momenta)
    refresh_interval < 1 && error("adaptive refresh_interval must be positive")
    refresh_min < 1 && error("adaptive refresh_min must be positive")
    refresh_max < refresh_min && error("adaptive refresh_max must be >= refresh_min")
    refresh_tol <= 0 && error("adaptive refresh_tol must be positive")
    refresh_growth_patience < 1 && error("adaptive refresh_growth_patience must be positive")

    Icomplex = Matrix{ComplexF64}(I, V, V)
    ws = ldr_workspace(Icomplex)
    tmp = similar(Icomplex)
    full_scaled = ldr(Icomplex)
    suffix_scaled = ldr(Icomplex)

    Gττ = zeros(ComplexF64, V, V)
    G00 = zeros(ComplexF64, V, V)
    Gτ0 = zeros(ComplexF64, V, V)
    G0τ = zeros(ComplexF64, V, V)
    prop_left_in = zeros(ComplexF64, V, 2V)
    prop_left_out = zeros(ComplexF64, V, 2V)
    prop_right_in = zeros(ComplexF64, 2V, V)
    prop_right_out = zeros(ComplexF64, 2V, V)

    cand_Gττ = zeros(ComplexF64, V, V)
    cand_Gτ0 = zeros(ComplexF64, V, V)
    cand_G0τ = zeros(ComplexF64, V, V)
    stable_Gττ = zeros(ComplexF64, V, V)
    stable_G00 = zeros(ComplexF64, V, V)
    stable_Gτ0 = zeros(ComplexF64, V, V)
    stable_G0τ = zeros(ComplexF64, V, V)

    Binv = [inv(Matrix{ComplexF64}(B)) for B in Bseq]
    Jq = [ce_current_operator_x_entries(system, qx=qx, qy=qy) for (qx, qy) in momenta]
    Jm = [ce_current_operator_x_entries(system, qx=-qx, qy=-qy) for (qx, qy) in momenta]
    responses = zeros(ComplexF64, nq)
    jτ_q = zeros(ComplexF64, L, nq)
    j0_m = zeros(ComplexF64, L, nq)
    response_m = zeros(ComplexF64, nq)

    initial_interval = clamp(refresh_interval, refresh_min, refresh_max)
    full = prefix[end]
    @inbounds for m in 1:Nft
        z = ComplexF64(ρ.expiφμ[m])
        w = ComplexF64(ρ.Z̃ₘ[m]) / Nft
        fill!(response_m, 0)

        scale_factorization!(full_scaled, full, z, ws, tmp)
        inv_IpA!(Gττ, full_scaled, ws)
        copyto!(G00, Gττ)
        copyto!(Gτ0, Icomplex)
        @. Gτ0 = Gτ0 - Gττ
        @. G0τ = -Gττ
        measure_projected_slice!(
            response_m, jτ_q, j0_m, 1,
            system, Jq, Jm, Gττ, G00, Gτ0, G0τ, w,
        )

        current_l = 0
        current_interval = initial_interval
        safe_streak = 0
        while current_l < L - 1
            remaining = L - 1 - current_l
            h = min(current_interval, remaining)
            accepted = false
            endpoint = current_l + h
            err = Inf

            while !accepted
                endpoint = current_l + h
                copyto!(cand_Gττ, Gττ)
                copyto!(cand_Gτ0, Gτ0)
                copyto!(cand_G0τ, G0τ)
                for slice_l in (current_l + 1):endpoint
                    propagate_displaced_greens_step_blocked!(
                        cand_Gττ, cand_Gτ0, cand_G0τ, G00,
                        Bseq[slice_l], Binv[slice_l], Icomplex,
                        prop_left_in, prop_left_out, prop_right_in, prop_right_out,
                        slice_l,
                    )
                end

                refresh_projected_displaced_greens!(
                    stable_Gττ, stable_G00, stable_Gτ0, stable_G0τ,
                    suffix_scaled, prefix[endpoint + 1], suffix[endpoint + 1], z, ws, tmp,
                )
                err = max_displaced_green_error(
                    cand_Gττ, cand_Gτ0, cand_G0τ,
                    stable_Gττ, stable_Gτ0, stable_G0τ,
                )
                if err <= refresh_tol || h <= refresh_min
                    accepted = true
                else
                    current_interval = max(refresh_min, max(1, h ÷ 2))
                    h = min(current_interval, remaining)
                    safe_streak = 0
                end
            end

            # Accept this segment.  Re-play only the cheap propagated steps for
            # interior slices, then measure the segment endpoint with the stable
            # LDR refresh already computed above.  If h==1 no propagated slice is
            # ever measured after a failed stability test.
            copyto!(cand_Gττ, Gττ)
            copyto!(cand_Gτ0, Gτ0)
            copyto!(cand_G0τ, G0τ)
            if h > 1
                for slice_l in (current_l + 1):(endpoint - 1)
                    propagate_displaced_greens_step_blocked!(
                        cand_Gττ, cand_Gτ0, cand_G0τ, G00,
                        Bseq[slice_l], Binv[slice_l], Icomplex,
                        prop_left_in, prop_left_out, prop_right_in, prop_right_out,
                        slice_l,
                    )
                    measure_projected_slice!(
                        response_m, jτ_q, j0_m, slice_l + 1,
                        system, Jq, Jm, cand_Gττ, G00, cand_Gτ0, cand_G0τ, w,
                    )
                end
            end
            measure_projected_slice!(
                response_m, jτ_q, j0_m, endpoint + 1,
                system, Jq, Jm, stable_Gττ, stable_G00, stable_Gτ0, stable_G0τ, w,
            )

            copyto!(Gττ, stable_Gττ)
            copyto!(G00, stable_G00)
            copyto!(Gτ0, stable_Gτ0)
            copyto!(G0τ, stable_G0τ)
            current_l = endpoint

            if err < refresh_tol / 10 && current_interval < refresh_max
                safe_streak += 1
                if safe_streak >= refresh_growth_patience
                    current_interval = min(refresh_max, current_interval + 1)
                    safe_streak = 0
                end
            else
                safe_streak = 0
            end
        end

        @. responses += Δτ * w * response_m
    end

    return (responses=responses, jτ_q=jτ_q, j0_m=j0_m)
end

function measure_current_responses_unequaltime(
    system,
    ρup::DensityMatrix,
    ρdn::DensityMatrix,
    prefix_up::Vector{<:LDR},
    suffix_up::Vector{<:LDR},
    prefix_dn::Vector{<:LDR},
    suffix_dn::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}},
)
    # Use the numerically stable canonical Fourier-projection estimator for
    # production measurements.  The eigenbasis fast path above is useful as a
    # diagnostic at small β, but it is not stable enough for the β=10 3x3 ED
    # benchmark or low-temperature production runs.
    up = canonical_same_spin_current_responses_projected(system, ρup, prefix_up, suffix_up, momenta)
    dn = canonical_same_spin_current_responses_projected(system, ρdn, prefix_dn, suffix_dn, momenta)
    Δτ = system.β / system.L
    nq = length(momenta)
    out = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        cross = Δτ * sum(
            up.jτ_q[:, iq] .* dn.j0_m[:, iq] .+
            dn.jτ_q[:, iq] .* up.j0_m[:, iq],
        ) / system.V
        out[iq] = up.responses[iq] + dn.responses[iq] + cross
    end
    return out
end

function measure_current_responses_unequaltime_canonical_recursion(
    system,
    ρup::DensityMatrix,
    ρdn::DensityMatrix,
    Bup::Vector{<:AbstractMatrix},
    Bdn::Vector{<:AbstractMatrix},
    prefix_up::Vector{<:LDR},
    suffix_up::Vector{<:LDR},
    prefix_dn::Vector{<:LDR},
    suffix_dn::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    allow_imbalanced::Bool=false,
    drift_tol::Float64=1e-2,
)
    if !allow_imbalanced && system.N[1] != system.N[2]
        error(
            "canonical-recursion BKT current estimator is validated only for balanced sectors " *
            "(Nup=Ndn) at present; got N=$(system.N). " *
            "Use --allow-imbalanced-canonical-recursion=true only for experimental debugging.",
        )
    end
    # Exact fixed-N current-current response without the roots-of-unity
    # canonical Fourier loop.  B slices are used only for short, one-slice
    # similarity propagation of the canonical recursion weights.
    up = canonical_same_spin_current_responses_canonical_recursion(
        system, ρup, Bup, prefix_up, suffix_up, momenta;
        spin=1,
        drift_tol=drift_tol,
    )
    dn = canonical_same_spin_current_responses_canonical_recursion(
        system, ρdn, Bdn, prefix_dn, suffix_dn, momenta;
        spin=2,
        drift_tol=drift_tol,
    )
    Δτ = system.β / system.L
    nq = length(momenta)
    out = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        cross = Δτ * (
            up.sum_jτ_q[iq] * dn.j0_m[iq] +
            dn.sum_jτ_q[iq] * up.j0_m[iq]
        ) / system.V
        out[iq] = up.responses[iq] + dn.responses[iq] + cross
    end
    finite_complex_vector(out) || error("canonical-recursion current estimator produced non-finite total responses: $(out)")
    return out
end

function measure_current_responses_unequaltime_canonical_stable(
    system,
    ρup::DensityMatrix,
    ρdn::DensityMatrix,
    prefix_up::Vector{<:LDR},
    prefix_dn::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    allow_imbalanced::Bool=false,
)
    if !allow_imbalanced && system.N[1] != system.N[2]
        error(
            "canonical-stable BKT current estimator is validated only for balanced sectors " *
            "(Nup=Ndn) at present; got N=$(system.N). " *
            "Use --allow-imbalanced-canonical-recursion=true only for experimental debugging.",
        )
    end
    # Stable no-Fourier fixed-N current estimator.  This keeps the same
    # canonical occupation/pair-occupation contractions as canonical-recursion,
    # but evaluates each displaced current operator with LDR-backed ldiv!
    # solves instead of propagating site-basis weights by long products of B and
    # B^{-1}.  It is more expensive than the structured SmoQy-style accumulator,
    # but remains exact in the fixed-N sector and is robust for the colder U=-3
    # grids where the B-slice accumulator drift guard can trip.
    up = canonical_same_spin_current_responses_fast_similarity(
        system, ρup, prefix_up, momenta; spin=1,
    )
    dn = canonical_same_spin_current_responses_fast_similarity(
        system, ρdn, prefix_dn, momenta; spin=2,
    )
    Δτ = system.β / system.L
    nq = length(momenta)
    out = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        cross = Δτ * sum(
            up.jτ_q[:, iq] .* dn.j0_m[:, iq] .+
            dn.jτ_q[:, iq] .* up.j0_m[:, iq],
        ) / system.V
        out[iq] = up.responses[iq] + dn.responses[iq] + cross
    end
    finite_complex_vector(out) || error("canonical-stable current estimator produced non-finite total responses: $(out)")
    return out
end

function measure_current_responses_unequaltime_structured_accumulator(
    system,
    ρup::DensityMatrix,
    ρdn::DensityMatrix,
    walker,
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    allow_imbalanced::Bool=false,
    drift_tol::Float64=1e-2,
)
    if !allow_imbalanced && system.N[1] != system.N[2]
        error(
            "structured canonical BKT current accumulator is validated only for balanced sectors " *
            "(Nup=Ndn) at present; got N=$(system.N). " *
            "Use --allow-imbalanced-canonical-recursion=true only for experimental debugging.",
        )
    end
    if system.useChargeHST
        error(
            "structured canonical BKT current accumulator is currently validated only for spin HS. " *
            "Use --bkt-current-estimator=propagated for charge-HS diagnostics.",
        )
    end

    dseq_up = ce_hs_diagonal_slices(system, walker, 1)
    up = canonical_same_spin_current_responses_structured_accumulator(
        system, ρup, dseq_up, momenta;
        spin=1,
        drift_tol=drift_tol,
    )
    if system.N[1] == system.N[2]
        # In spin-HS CanEnsAFQMC copies the balanced up density matrix into the
        # down sector after every accepted update, but the down one-slice
        # potential is the conjugate/sign-flipped HS field, so we must still
        # propagate the down accumulator with its own d_l sequence.
    end
    dseq_dn = ce_hs_diagonal_slices(system, walker, 2)
    dn = canonical_same_spin_current_responses_structured_accumulator(
        system, ρdn, dseq_dn, momenta;
        spin=2,
        drift_tol=drift_tol,
    )
    Δτ = system.β / system.L
    nq = length(momenta)
    out = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        cross = Δτ * (
            up.sum_jτ_q[iq] * dn.j0_m[iq] +
            dn.sum_jτ_q[iq] * up.j0_m[iq]
        ) / system.V
        out[iq] = up.responses[iq] + dn.responses[iq] + cross
    end
    finite_complex_vector(out) || error("structured canonical current accumulator produced non-finite total responses: $(out)")
    return out
end

function combine_same_spin_current_response(system, up)
    # Balanced charge-HS attractive-Hubbard configurations have identical up
    # and down one-body propagators in the fixed-Nup=Ndn sector.  In that case
    # the total current response can be assembled from one same-spin
    # accumulator rather than measuring the second spin redundantly:
    #
    #   Λ = Λ↑↑ + Λ↓↓ + Λ↑↓ + Λ↓↑
    #     = 2 Λ↑↑ + 2 Δτ/V (Στ <J↑(τ)> <J↑(0)>).
    #
    # This is exact only when the two spin sectors are represented by the same
    # one-body matrices and the same fixed particle number.  The caller is
    # responsible for enforcing that precondition.
    Δτ = system.β / system.L
    nq = length(up.responses)
    out = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        cross = 2 * Δτ * up.sum_jτ_q[iq] * up.j0_m[iq] / system.V
        out[iq] = 2 * up.responses[iq] + cross
    end
    return out
end

function combine_same_spin_current_response_from_timeseries(system, up)
    Δτ = system.β / system.L
    nq = length(up.responses)
    out = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        cross = 2 * Δτ * sum(up.jτ_q[:, iq] .* up.j0_m[:, iq]) / system.V
        out[iq] = 2 * up.responses[iq] + cross
    end
    return out
end

function measure_current_responses_unequaltime_canonical_accumulator(
    system,
    ρup::DensityMatrix,
    ρdn::DensityMatrix,
    Bup::Vector{<:AbstractMatrix},
    Bdn::Vector{<:AbstractMatrix},
    prefix_up::Vector{<:LDR},
    suffix_up::Vector{<:LDR},
    prefix_dn::Vector{<:LDR},
    suffix_dn::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    allow_imbalanced::Bool=false,
    drift_tol::Float64=1e-2,
)
    if !allow_imbalanced && system.N[1] != system.N[2]
        error(
            "canonical-accumulator BKT current estimator is validated only for balanced sectors " *
            "(Nup=Ndn) at present; got N=$(system.N). " *
            "Use --allow-imbalanced-canonical-recursion=true only for experimental debugging.",
        )
    end

    # This is the SmoQy-style current accumulator path for fixed-N CE.  It keeps
    # the canonical-recursion contractions but avoids the roots-of-unity Fourier
    # projection loop.  For spin-HS attractive-U configurations the one-slice
    # propagators are well-conditioned enough to accumulate the current weights
    # by B-slice propagation.  For charge-HS attractive-U configurations the
    # real HS potentials can be strongly ill-conditioned; there we rebase every
    # τ slice with the stable LDR similarity formula (the refresh interval is
    # effectively one) until a multi-slice stable-refresh accumulator is
    # validated.  This preserves correctness while retaining the no-Fourier-loop
    # contraction.
    if system.useChargeHST
        error(
            "canonical-accumulator is not yet validated for attractive-Hubbard charge HS. " *
            "The real charge-HS B slices are too ill-conditioned for the current no-Fourier " *
            "accumulator; use --bkt-current-estimator=propagated for charge-HS validation, " *
            "or use spin HS with canonical-accumulator/canonical-recursion for production.",
        )
    end

    # Spin-HS/default path: propagate the accumulated current-response weights
    # directly in site basis.
    up = canonical_same_spin_current_responses_canonical_recursion(
        system, ρup, Bup, prefix_up, suffix_up, momenta;
        spin=1,
        drift_tol=drift_tol,
    )
    if system.useChargeHST && system.N[1] == system.N[2]
        out = combine_same_spin_current_response(system, up)
        finite_complex_vector(out) || error("canonical-accumulator current estimator produced non-finite total responses: $(out)")
        return out
    end

    dn = canonical_same_spin_current_responses_canonical_recursion(
        system, ρdn, Bdn, prefix_dn, suffix_dn, momenta;
        spin=2,
        drift_tol=drift_tol,
    )
    Δτ = system.β / system.L
    nq = length(momenta)
    out = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        cross = Δτ * (
            up.sum_jτ_q[iq] * dn.j0_m[iq] +
            dn.sum_jτ_q[iq] * up.j0_m[iq]
        ) / system.V
        out[iq] = up.responses[iq] + dn.responses[iq] + cross
    end
    finite_complex_vector(out) || error("canonical-accumulator current estimator produced non-finite total responses: $(out)")
    return out
end

function measure_current_responses_unequaltime_propagated(
    system,
    ρup::DensityMatrix,
    ρdn::DensityMatrix,
    Bup::Vector{<:AbstractMatrix},
    Bdn::Vector{<:AbstractMatrix},
    prefix_up::Vector{<:LDR},
    suffix_up::Vector{<:LDR},
    prefix_dn::Vector{<:LDR},
    suffix_dn::Vector{<:LDR},
    momenta::AbstractVector{<:Tuple{Float64,Float64}};
    refresh_interval::Int=10,
    adaptive_refresh::Bool=false,
    refresh_tol::Float64=1e-6,
    refresh_min::Int=1,
    refresh_max::Int=max(refresh_interval, refresh_min),
    refresh_growth_patience::Int=3,
)
    if adaptive_refresh
        up = canonical_same_spin_current_responses_propagated_adaptive(
            system, ρup, Bup, prefix_up, suffix_up, momenta;
            refresh_interval=refresh_interval,
            refresh_tol=refresh_tol,
            refresh_min=refresh_min,
            refresh_max=refresh_max,
            refresh_growth_patience=refresh_growth_patience,
        )
        dn = canonical_same_spin_current_responses_propagated_adaptive(
            system, ρdn, Bdn, prefix_dn, suffix_dn, momenta;
            refresh_interval=refresh_interval,
            refresh_tol=refresh_tol,
            refresh_min=refresh_min,
            refresh_max=refresh_max,
            refresh_growth_patience=refresh_growth_patience,
        )
    else
        up = canonical_same_spin_current_responses_propagated(
            system, ρup, Bup, prefix_up, suffix_up, momenta,
            refresh_interval=refresh_interval,
        )
        dn = canonical_same_spin_current_responses_propagated(
            system, ρdn, Bdn, prefix_dn, suffix_dn, momenta,
            refresh_interval=refresh_interval,
        )
    end
    Δτ = system.β / system.L
    nq = length(momenta)
    out = zeros(ComplexF64, nq)
    @inbounds for iq in 1:nq
        cross = Δτ * sum(
            up.jτ_q[:, iq] .* dn.j0_m[:, iq] .+
            dn.jτ_q[:, iq] .* up.j0_m[:, iq],
        ) / system.V
        out[iq] = up.responses[iq] + dn.responses[iq] + cross
    end
    return out
end

function measure_current_response_unequaltime(
    system,
    ρup::DensityMatrix,
    ρdn::DensityMatrix,
    prefix_up::Vector{<:LDR},
    suffix_up::Vector{<:LDR},
    prefix_dn::Vector{<:LDR},
    suffix_dn::Vector{<:LDR};
    qx::Float64,
    qy::Float64,
)
    return measure_current_responses_unequaltime(
        system, ρup, ρdn, prefix_up, suffix_up, prefix_dn, suffix_dn,
        [(qx, qy)],
    )[1]
end
