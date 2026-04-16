#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays
using Printf
using TOML
using CanEnsAFQMC: hopping_matrix_Hubbard_2d

function parse_args(args::Vector{String})
    params = Dict{String, Any}(
        "lx" => 4,
        "ly" => 2,
        "nup" => 4,
        "ndn" => 4,
        "u" => 4.0,
        "t" => 1.0,
        "beta_list" => collect(1.0:1.0:20.0),
        "outdir" => joinpath("results", "interacting_qmc_ed", "ed_hubbard_4x2_half_filling_beta_scan"),
    )

    for arg in args
        if startswith(arg, "--lx=")
            params["lx"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--ly=")
            params["ly"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--nup=")
            params["nup"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--ndn=")
            params["ndn"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--u=")
            params["u"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--t=")
            params["t"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--beta-list=")
            raw = split(split(arg, "=", limit=2)[2], ",")
            params["beta_list"] = [parse(Float64, strip(v)) for v in raw if !isempty(strip(v))]
        elseif startswith(arg, "--outdir=")
            params["outdir"] = split(arg, "=", limit=2)[2]
        elseif arg in ("-h", "--help")
            println(
                """
                Exact diagonalization energy scan for the spinful Hubbard model in a fixed (Nup, Ndn) sector.

                Options:
                  --lx=<int>             Lattice size in x (default: 4)
                  --ly=<int>             Lattice size in y (default: 2)
                  --nup=<int>            Number of up-spin fermions (default: 4)
                  --ndn=<int>            Number of down-spin fermions (default: 4)
                  --u=<float>            On-site interaction strength (default: 4.0)
                  --t=<float>            Hopping amplitude (default: 1.0)
                  --beta-list=<list>     Comma-separated beta list (default: 1,2,...,20)
                  --outdir=<path>        Output directory
                """
            )
            exit(0)
        else
            error("Unrecognized argument: $arg")
        end
    end

    return params
end

function generate_basis(nsites::Int, nparticles::Int)
    basis = UInt64[]
    limit = UInt64(1) << nsites
    for state in UInt64(0):UInt64(limit - 1)
        count_ones(state) == nparticles || continue
        push!(basis, state)
    end
    return basis
end

@inline isoccupied(state::UInt64, site::Int) = ((state >> (site - 1)) & UInt64(1)) == UInt64(1)

function apply_cdag_c(state::UInt64, i::Int, j::Int)
    isoccupied(state, j) || return nothing
    isoccupied(state, i) && return nothing

    mask_j = UInt64(1) << (j - 1)
    state_after_annihilation = state & ~mask_j
    sign1 = isodd(count_ones(state & (mask_j - 1))) ? -1.0 : 1.0

    mask_i = UInt64(1) << (i - 1)
    sign2 = isodd(count_ones(state_after_annihilation & (mask_i - 1))) ? -1.0 : 1.0
    new_state = state_after_annihilation | mask_i

    return sign1 * sign2, new_state
end

function one_spin_hamiltonian(tmat::AbstractMatrix{<:Real}, basis::Vector{UInt64})
    index_of = Dict(state => idx for (idx, state) in enumerate(basis))
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (col, state) in enumerate(basis)
        for i in axes(tmat, 1), j in axes(tmat, 2)
            amp = tmat[i, j]
            amp == 0 && continue
            result = apply_cdag_c(state, i, j)
            result === nothing && continue
            sign, new_state = result
            row = index_of[new_state]
            push!(rows, row)
            push!(cols, col)
            push!(vals, amp * sign)
        end
    end

    return sparse(rows, cols, vals, length(basis), length(basis))
end

function interaction_diagonal(up_basis::Vector{UInt64}, dn_basis::Vector{UInt64}, u::Float64)
    diag = Vector{Float64}(undef, length(up_basis) * length(dn_basis))
    idx = 1
    for up in up_basis
        for dn in dn_basis
            diag[idx] = u * count_ones(up & dn)
            idx += 1
        end
    end
    return diag
end

function thermal_energy(evals::Vector{Float64}, beta::Float64)
    emin = minimum(evals)
    shifted = evals .- emin
    weights = exp.(-beta .* shifted)
    z = sum(weights)
    mean_shifted = sum(shifted .* weights) / z
    return emin + mean_shifted
end

function write_metadata(path::String, params::Dict{String, Any}, nbasis_up::Int, nbasis_dn::Int, dim::Int, evals::Vector{Float64})
    metadata = Dict(
        "model" => "spinful Hubbard",
        "boundary" => "periodic",
        "ensemble" => "fixed Nup and Ndn exact diagonalization",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"], "nsites" => params["lx"] * params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"], "ntotal" => params["nup"] + params["ndn"]),
        "hamiltonian" => Dict("t" => params["t"], "u" => params["u"]),
        "hilbert_space" => Dict("nbasis_up" => nbasis_up, "nbasis_dn" => nbasis_dn, "dimension" => dim),
        "spectrum" => Dict("ground_state_energy" => minimum(evals), "max_energy" => maximum(evals)),
        "beta_list" => params["beta_list"],
    )
    open(path, "w") do io
        TOML.print(io, metadata)
    end
end

function main()
    params = parse_args(ARGS)
    nsites = params["lx"] * params["ly"]
    tmat = hopping_matrix_Hubbard_2d(params["lx"], params["ly"], params["t"])

    up_basis = generate_basis(nsites, params["nup"])
    dn_basis = generate_basis(nsites, params["ndn"])
    hup = one_spin_hamiltonian(tmat, up_basis)
    hdn = one_spin_hamiltonian(tmat, dn_basis)

    dim_up = length(up_basis)
    dim_dn = length(dn_basis)
    full_dim = dim_up * dim_dn
    hfull = kron(sparse(I, dim_dn, dim_dn), hup) + kron(hdn, sparse(I, dim_up, dim_up))
    hfull = hfull + spdiagm(0 => interaction_diagonal(up_basis, dn_basis, params["u"]))

    evals = eigvals(Symmetric(Matrix(hfull)))
    sort!(evals)

    outdir = normpath(joinpath(@__DIR__, "..", params["outdir"]))
    mkpath(outdir)

    write_metadata(joinpath(outdir, "metadata.toml"), params, dim_up, dim_dn, full_dim, evals)

    open(joinpath(outdir, "energy_vs_beta.tsv"), "w") do io
        println(io, "beta\ttotal_energy\ttotal_energy_per_site\ttotal_energy_per_particle")
        for beta in params["beta_list"]
            etot = thermal_energy(evals, beta)
            @printf(
                io,
                "%.12f\t%.12f\t%.12f\t%.12f\n",
                beta,
                etot,
                etot / nsites,
                etot / (params["nup"] + params["ndn"]),
            )
        end
    end

    open(joinpath(outdir, "spectrum.tsv"), "w") do io
        println(io, "level_index\teigenvalue")
        for (idx, value) in enumerate(evals)
            @printf(io, "%d\t%.12f\n", idx, value)
        end
    end

    println("Computed exact Hubbard ED energy scan")
    println("  lattice = $(params["lx"])x$(params["ly"])")
    println("  nup = $(params["nup"]), ndn = $(params["ndn"])")
    println("  U = $(params["u"]), t = $(params["t"])")
    println("  Hilbert dimension = $full_dim")
    println("  output directory = $outdir")
end

main()
