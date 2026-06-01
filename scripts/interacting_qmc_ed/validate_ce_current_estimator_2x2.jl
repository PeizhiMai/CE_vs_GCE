using LinearAlgebra
using Printf
using CanEnsAFQMC

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
            row = index[new_state]
            M[row, col] += A[i, j] * sign
        end
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

function manybody_propagator_sector(B::AbstractMatrix{ComplexF64}, occ_lists::Vector{Vector{Int}})
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

function prefix_suffix(Ms)
    dim = size(Ms[1], 1)
    L = length(Ms)
    pre = Vector{Matrix{ComplexF64}}(undef, L + 1)
    suf = Vector{Matrix{ComplexF64}}(undef, L + 1)
    pre[1] = Matrix{ComplexF64}(I, dim, dim)
    for l in 1:L
        pre[l + 1] = Ms[l] * pre[l]
    end
    suf[L + 1] = Matrix{ComplexF64}(I, dim, dim)
    for l in L:-1:1
        suf[l] = suf[l + 1] * Ms[l]
    end
    return pre, suf
end

function estimator_fixedorigin(system, Ms, Jq, Jm)
    pre, suf = prefix_suffix(Ms)
    Z = tr(pre[end])
    acc = zero(ComplexF64)
    for l in 0:(system.L - 1)
        U = pre[l + 1]
        V = suf[l + 1]
        acc += tr(V * Jq * U * Jm) / Z
    end
    return real((system.β / system.L) * acc / system.V)
end

function mixed_derivatives_product(Ms, A, B)
    dim = size(Ms[1], 1)
    P00 = Matrix{ComplexF64}(I, dim, dim)
    P10 = zeros(ComplexF64, dim, dim)
    P01 = zeros(ComplexF64, dim, dim)
    P11 = zeros(ComplexF64, dim, dim)
    AB = A * B
    for M in Ms
        Q00 = M * P00
        Q10 = M * P10
        Q01 = M * P01
        Q11 = M * P11
        P00 = Q00
        P10 = A * Q00 + Q10
        P01 = B * Q00 + Q01
        P11 = AB * Q00 + A * Q01 + B * Q10 + Q11
        scale = maximum(abs, P00)
        scale = max(scale, maximum(abs, P10))
        scale = max(scale, maximum(abs, P01))
        scale = max(scale, maximum(abs, P11))
        if scale > 0
            P00 ./= scale; P10 ./= scale; P01 ./= scale; P11 ./= scale
        end
    end
    return tr(P00), tr(P10), tr(P01), tr(P11)
end

function estimator_doublesum(system, Ms, Jq, Jm)
    Z, ZA, ZB, ZAB = mixed_derivatives_product(Ms, Jq, Jm)
    return Z, ZA, ZB, ZAB
end

function site_index_2d(x::Int, y::Int, lx::Int)
    x + 1 + y * lx
end

function full_docc_diag(up_basis, dn_basis, nsites)
    vals = Float64[]
    for dn in dn_basis, up in up_basis
        docc = 0
        for i in 1:nsites
            docc += occ(up, i) & occ(dn, i)
        end
        push!(vals, docc)
    end
    vals
end

function main()
    lx = 2; ly = 2; nup = 2; ndn = 2; u = -5.0; beta = 1.0; dtau = 0.25
    L = round(Int, beta / dtau)
    system = GenericHubbard((lx, ly, 1), (nup, ndn), hopping_matrix_Hubbard_2d(lx, ly, 1.0), u, 0.0, beta, L,
        sys_type=Float64, useChargeHST=true, useFirstOrderTrotter=false)

    up_basis = gen_basis(system.V, nup)
    dn_basis = gen_basis(system.V, ndn)
    occ_up = [occupied_sites(s, system.V) for s in up_basis]
    occ_dn = [occupied_sites(s, system.V) for s in dn_basis]
    dup = length(up_basis); ddn = length(dn_basis)

    qmin = 2π / lx
    JupL = bilinear_sector_matrix(current_operator_x(system, qx=qmin, qy=0.0), up_basis)
    JupT = bilinear_sector_matrix(current_operator_x(system, qx=0.0, qy=qmin), up_basis)
    JdnL = JupL; JdnT = JupT
    Iu = Matrix{ComplexF64}(I, dup, dup)
    Id = Matrix{ComplexF64}(I, ddn, ddn)
    JfullL = kron(Id, JupL) + kron(JdnL, Iu)
    JfullLm = kron(Id, adjoint(JupL)) + kron(adjoint(JdnL), Iu)
    JfullT = kron(Id, JupT) + kron(JdnT, Iu)
    JfullTm = kron(Id, adjoint(JupT)) + kron(adjoint(JdnT), Iu)

    tmp = zeros(ComplexF64, system.V, system.V)

    Wslice_HS = zeros(ComplexF64, dup * ddn, dup * ddn)
    for code1 in 0:(2^system.V - 1)
        σ = Vector{Int}(undef, system.V)
        for i in 1:system.V
            bit = (code1 >> (i - 1)) & 1
            σ[i] = bit == 1 ? 1 : -1
        end
        B = zeros(ComplexF64, system.V, system.V)
        CanEnsAFQMC.imagtime_propagator!(B, σ, system, tmpmat=tmp)
        M = manybody_propagator_sector(B, occ_up)
        Wslice_HS .+= kron(M, M)
    end

    Wslice = Wslice_HS
    Wtot = Wslice^L
    Zfull = tr(Wtot)
    λL_exact = real((beta / L) * sum(tr(Wslice^(L-l) * JfullL * Wslice^l * JfullLm) / Zfull for l in 0:(L-1)) / system.V)
    λT_exact = real((beta / L) * sum(tr(Wslice^(L-l) * JfullT * Wslice^l * JfullTm) / Zfull for l in 0:(L-1)) / system.V)

    totalZ = 0.0 + 0im
    λL_dbl = 0.0 + 0im
    λT_dbl = 0.0 + 0im
    nsigma = 2^(system.V * system.L)
    for code in 0:(nsigma - 1)
        Bup = Vector{Matrix{ComplexF64}}(undef, system.L)
        for l in 1:system.L
            σ = Vector{Int}(undef, system.V)
            for i in 1:system.V
                bit = (code >> ((l - 1) * system.V + (i - 1))) & 1
                σ[i] = bit == 1 ? 1 : -1
            end
            B = zeros(ComplexF64, system.V, system.V)
            CanEnsAFQMC.imagtime_propagator!(B, σ, system, tmpmat=tmp)
            Bup[l] = B
        end
        Ms = [manybody_propagator_sector(B, occ_up) for B in Bup]
        P = Ms[1]
        for l in 2:system.L
            P = Ms[l] * P
        end
        Ztrue = tr(P)
        Zs, ZA_L, ZB_L, ZAB_L = mixed_derivatives_product(Ms, JupL, adjoint(JupL))
        _, ZA_T, ZB_T, ZAB_T = mixed_derivatives_product(Ms, JupT, adjoint(JupT))
        λconf_L = real((beta / system.L) * (ZAB_L * Zs + ZA_L * ZB_L + ZB_L * ZA_L + Zs * ZAB_L) / (Zs * Zs) / system.L / system.V)
        λconf_T = real((beta / system.L) * (ZAB_T * Zs + ZA_T * ZB_T + ZB_T * ZA_T + Zs * ZAB_T) / (Zs * Zs) / system.L / system.V)
        Zconf = Ztrue * Ztrue
        totalZ += Zconf
        λL_dbl += Zconf * λconf_L
        λT_dbl += Zconf * λconf_T
    end

    @printf("2x2 beta=%.3f dtau=%.3f L=%d\n", beta, dtau, L)
    @printf("Z full-space = %.12f%+.12fi\n", real(Zfull), imag(Zfull))
    @printf("Z HS sum     = %.12f%+.12fi\n", real(totalZ), imag(totalZ))
    @printf("Exact Trotter full-space: lambdaL=%.12f lambdaT=%.12f\n", λL_exact, λT_exact)
    @printf("HS average double-sum:    lambdaL=%.12f lambdaT=%.12f\n", real(λL_dbl/totalZ), real(λT_dbl/totalZ))
end

main()
