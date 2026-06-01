using LinearAlgebra
using Random
using Statistics
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
    out = Vector{Int}(undef, count_ones(state))
    k = 1
    for i in 1:nsites
        if occ(state, i) == 1
            out[k] = i
            k += 1
        end
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
function build_prefix_suffix(Bmats::Vector{Matrix{ComplexF64}})
    dim = size(Bmats[1], 1)
    L = length(Bmats)
    prefix = Vector{Matrix{ComplexF64}}(undef, L + 1)
    suffix = Vector{Matrix{ComplexF64}}(undef, L + 1)
    prefix[1] = Matrix{ComplexF64}(I, dim, dim)
    for l in 1:L
        prefix[l + 1] = Bmats[l] * prefix[l]
    end
    suffix[L + 1] = Matrix{ComplexF64}(I, dim, dim)
    for l in L:-1:1
        suffix[l] = suffix[l + 1] * Bmats[l]
    end
    return prefix, suffix
end

function sample_current_response_doublesum_from_slices(system, Bup, Bdn, occ_lists, Jq, Jm)
    mup = [manybody_propagator_sector(B, occ_lists) for B in Bup]
    mdn = [manybody_propagator_sector(B, occ_lists) for B in Bdn]
    function mixed_derivs(Ms, A, B)
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
                P00 ./= scale
                P10 ./= scale
                P01 ./= scale
                P11 ./= scale
            end
        end
        return tr(P00), tr(P10), tr(P01), tr(P11)
    end
    Zu0, ZuA, ZuB, ZuAB = mixed_derivs(mup, Jq, Jm)
    Zd0, ZdA, ZdB, ZdAB = mixed_derivs(mdn, Jq, Jm)
    total = ZuAB * Zd0 + ZuA * ZdB + ZuB * ZdA + Zu0 * ZdAB
    return real((system.β / system.L) * total / (Zu0 * Zd0) / system.L / system.V)
end

function sample_current_response_exactsector_from_slices(system, Bup, Bdn, occ_lists, Jq, Jm)
    mup = [manybody_propagator_sector(B, occ_lists) for B in Bup]
    mdn = [manybody_propagator_sector(B, occ_lists) for B in Bdn]
    pup, sup = build_prefix_suffix(mup)
    pdn, sdn = build_prefix_suffix(mdn)
    Zup = tr(pup[end])
    Zdn = tr(pdn[end])
    Δτ = system.β / system.L
    λ = zero(ComplexF64)
    for l in 0:(system.L - 1)
        Uup = pup[l + 1]
        Vup = sup[l + 1]
        Udn = pdn[l + 1]
        Vdn = sdn[l + 1]
        same_up = tr(Vup * Jq * Uup * Jm) / Zup
        same_dn = tr(Vdn * Jq * Udn * Jm) / Zdn
        jτ_up = tr(Vup * Jq * Uup) / Zup
        j0_up = tr(Vup * Uup * Jm) / Zup
        jτ_dn = tr(Vdn * Jq * Udn) / Zdn
        j0_dn = tr(Vdn * Udn * Jm) / Zdn
        λ += same_up + same_dn + jτ_up * j0_dn + jτ_dn * j0_up
    end
    return real(Δτ * λ / system.V)
end

# build a constant-slice system/walker surrogate
lx=3; ly=3; nup=4; ndn=4; u=-5.0; beta=10.0; dtau=0.025; L=round(Int,beta/dtau)
T = hopping_matrix_Hubbard_2d(lx,ly,1.0)
system = GenericHubbard((lx,ly,1),(nup,ndn),T,u,0.0,beta,L,sys_type=Float64,useChargeHST=true,useFirstOrderTrotter=false)
σ = ones(Int, system.V)
Bup = [zeros(ComplexF64, system.V, system.V) for _ in 1:L]
Bdn = [zeros(ComplexF64, system.V, system.V) for _ in 1:L]
tmp = zeros(ComplexF64, system.V, system.V)
for l in 1:L
    CanEnsAFQMC.imagtime_propagator!(Bup[l], Bdn[l], σ, system, tmpmat=tmp)
end
basis = gen_basis(system.V, nup)
occ_lists = [occupied_sites(state, system.V) for state in basis]
qmin = 2π / system.Ns[1]
JqL = bilinear_sector_matrix(current_operator_x(system, qx=qmin, qy=0.0), basis)
JmL = bilinear_sector_matrix(current_operator_x(system, qx=-qmin, qy=0.0), basis)
JqT = bilinear_sector_matrix(current_operator_x(system, qx=0.0, qy=qmin), basis)
JmT = bilinear_sector_matrix(current_operator_x(system, qx=0.0, qy=-qmin), basis)

λL_exactsector = sample_current_response_exactsector_from_slices(system, Bup, Bdn, occ_lists, JqL, JmL)
λT_exactsector = sample_current_response_exactsector_from_slices(system, Bup, Bdn, occ_lists, JqT, JmT)

# compare to density-matrix spectral formula on same constant configuration
qmc = QMC(system, nwarmups=0, nsamples=1, measure_interval=1, stab_interval=10, useClusterUpdate=false, num_FourierPoints=system.V+1, forceSymmetry=true, isLowrank=true, lrThld=1e-10, saveRatio=false)
walker = Walker(system, qmc, auxfield=ones(Int, system.V, system.L))
ρup = DensityMatrix(system, Nft=qmc.num_FourierPoints)
ρdn = DensityMatrix(system, Nft=qmc.num_FourierPoints)
update!(system, walker, ρup, 1)
update!(system, walker, ρdn, ρup)
λL_spectral = real(measure_BilinearStaticResponse(system, ρup, 1, current_operator_x(system, qx=qmin, qy=0.0), B=current_operator_x(system, qx=-qmin, qy=0.0)) + measure_BilinearStaticResponse(system, ρdn, 2, current_operator_x(system, qx=qmin, qy=0.0), B=current_operator_x(system, qx=-qmin, qy=0.0)) ) / system.V
λT_spectral = real(measure_BilinearStaticResponse(system, ρup, 1, current_operator_x(system, qx=0.0, qy=qmin), B=current_operator_x(system, qx=0.0, qy=-qmin)) + measure_BilinearStaticResponse(system, ρdn, 2, current_operator_x(system, qx=0.0, qy=qmin), B=current_operator_x(system, qx=0.0, qy=-qmin)) ) / system.V

λL_doublesum = sample_current_response_doublesum_from_slices(system, Bup, Bdn, occ_lists, JqL, JmL)
λT_doublesum = sample_current_response_doublesum_from_slices(system, Bup, Bdn, occ_lists, JqT, JmT)
println("constant config exactsector λL=", λL_exactsector, " λT=", λT_exactsector)
println("constant config doublesum  λL=", λL_doublesum, " λT=", λT_doublesum)
println("constant config spectral   λL=", λL_spectral, " λT=", λT_spectral)
