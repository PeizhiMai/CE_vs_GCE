using LinearAlgebra
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
    basis
end

@inline occ(state::Int, site::Int) = (state >> (site - 1)) & 1
occupied_sites(state::Int, nsites::Int) = [i for i in 1:nsites if occ(state, i) == 1]

function manybody_propagator_sector(B, occ_lists)
    dim = length(occ_lists)
    M = zeros(ComplexF64, dim, dim)
    for row in 1:dim, col in 1:dim
        M[row, col] = det(@view B[occ_lists[row], occ_lists[col]])
    end
    M
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

function slice_sum_maxdiff(; useChargeHST::Bool, sys_type::DataType)
    lx = 2
    ly = 2
    nup = 2
    ndn = 2
    u = -5.0
    beta = 0.25
    L = 1

    system = GenericHubbard(
        (lx, ly, 1),
        (nup, ndn),
        hopping_matrix_Hubbard_2d(lx, ly, 1.0),
        u,
        0.0,
        beta,
        L,
        sys_type=sys_type,
        useChargeHST=useChargeHST,
        useFirstOrderTrotter=false,
    )

    basis = gen_basis(system.V, nup)
    occs = [occupied_sites(s, system.V) for s in basis]
    dup = length(basis)
    Khalf = manybody_propagator_sector(complex(system.Bk), occs)
    Wslice_exact = kron(Khalf, Khalf) * Diagonal(exp.(-beta * u .* full_docc_diag(basis, basis, system.V))) * kron(Khalf, Khalf)

    tmp = zeros(ComplexF64, system.V, system.V)
    Wslice_hs = zeros(ComplexF64, dup * dup, dup * dup)
    for code in 0:(2^system.V - 1)
        σ = [((code >> (i - 1)) & 1) == 1 ? 1 : -1 for i in 1:system.V]
        if useChargeHST
            B = zeros(ComplexF64, system.V, system.V)
            CanEnsAFQMC.imagtime_propagator!(B, σ, system, tmpmat=tmp)
            M = manybody_propagator_sector(B, occs)
            Wslice_hs .+= kron(M, M)
        else
            Bup = zeros(ComplexF64, system.V, system.V)
            Bdn = zeros(ComplexF64, system.V, system.V)
            CanEnsAFQMC.imagtime_propagator!(Bup, Bdn, σ, system, tmpmat=tmp)
            Mup = manybody_propagator_sector(Bup, occs)
            Mdn = manybody_propagator_sector(Bdn, occs)
            Wslice_hs .+= kron(Mdn, Mup)
        end
    end

    α = Wslice_hs[1, 1] / Wslice_exact[1, 1]
    return maximum(abs, Wslice_hs .- α .* Wslice_exact)
end

println("useChargeHST=true, sys_type=Float64, maxdiff=", slice_sum_maxdiff(useChargeHST=true, sys_type=Float64))
println("useChargeHST=false, sys_type=ComplexF64, maxdiff=", slice_sum_maxdiff(useChargeHST=false, sys_type=ComplexF64))
