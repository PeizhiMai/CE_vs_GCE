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
    q = zeros(ComplexF64, N + 1)
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

        for j in 1:Ns
            i == j && continue
            # Divide poly_ex_i(x) by (1 + λ[j] x) to get the polynomial
            # excluding both i and j.  q[k+1] is the x^k coefficient.
            q[1] = poly_ex_i[1]
            for k in 1:N
                q[k + 1] = poly_ex_i[k + 1] - λ[j] * q[k]
            end
            ninj[i, j] = λ[i] * λ[j] * q[N - 1] / Z
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
    up = canonical_same_spin_current_response_fast(system, ρup, prefix_up, qx=qx, qy=qy, spin=1, return_expectations=true)
    dn = canonical_same_spin_current_response_fast(system, ρdn, prefix_dn, qx=qx, qy=qy, spin=2, return_expectations=true)
    Δτ = system.β / system.L
    cross = Δτ * sum(up.jτ_q .* dn.j0_m .+ dn.jτ_q .* up.j0_m) / system.V
    return up.response + dn.response + cross
end
