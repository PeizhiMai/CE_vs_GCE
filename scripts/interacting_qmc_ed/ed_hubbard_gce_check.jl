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
        "u" => -5.0,
        "t" => 1.0,
        "tprime" => 0.0,
        "mu" => -1.0,
        "beta" => 10.0,
        "ph_sym_form" => true,
        "hopping_convention" => "canens",
        "outdir" => joinpath("results", "interacting_qmc_ed", "ed_gce_hubbard_4x2_Um5_mu_m1_beta10"),
    )
    for arg in args
        if startswith(arg, "--lx=")
            params["lx"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--ly=")
            params["ly"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--u=")
            params["u"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--t=")
            params["t"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tprime=") || startswith(arg, "--tp=")
            params["tprime"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--mu=")
            params["mu"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--beta=")
            params["beta"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--ph-sym-form=")
            params["ph_sym_form"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--hopping-convention=")
            params["hopping_convention"] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--outdir=")
            params["outdir"] = split(arg, "=", limit=2)[2]
        elseif arg in ("-h", "--help")
            println("""
            Grand-canonical exact diagonalization check for small spinful Hubbard clusters.

            Hamiltonian with --ph-sym-form=true:
              K + U Σ_i (n_up_i - 1/2)(n_dn_i - 1/2) - μ Σ_iσ n_iσ

            Options:
              --lx=<int>              default 4
              --ly=<int>              default 2
              --u=<float>             default -5.0
              --t=<float>             default 1.0
              --tprime=<float>        parsed/recorded; only tprime=0 is currently implemented
              --mu=<float>            default -1.0
              --beta=<float>          default 10.0
              --ph-sym-form=<bool>    default true
              --hopping-convention=<canens|smoqy> default canens
              --outdir=<path>         output directory
            """)
            exit(0)
        else
            error("Unrecognized argument: $arg")
        end
    end
    abs(params["tprime"]) < 1e-14 || error("This simple check currently supports tprime=0 only.")
    params["hopping_convention"] in ("smoqy", "canens") || error("--hopping-convention must be smoqy or canens")
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
    rows = Int[]; cols = Int[]; vals = Float64[]
    for (col, state) in enumerate(basis)
        for i in axes(tmat, 1), j in axes(tmat, 2)
            amp = tmat[i, j]
            amp == 0 && continue
            result = apply_cdag_c(state, i, j)
            result === nothing && continue
            sign, new_state = result
            push!(rows, index_of[new_state]); push!(cols, col); push!(vals, amp * sign)
        end
    end
    return sparse(rows, cols, vals, length(basis), length(basis))
end

function one_spin_bilinear_operator(nsites::Int, basis::Vector{UInt64}, terms::Vector{Tuple{Int,Int,ComplexF64}})
    index_of = Dict(state => idx for (idx, state) in enumerate(basis))
    rows = Int[]; cols = Int[]; vals = ComplexF64[]
    for (col, state) in enumerate(basis)
        for (i, j, amp) in terms
            result = apply_cdag_c(state, i, j)
            result === nothing && continue
            sign, new_state = result
            push!(rows, index_of[new_state]); push!(cols, col); push!(vals, amp * sign)
        end
    end
    return sparse(rows, cols, vals, length(basis), length(basis))
end

@inline site_index(x::Int, y::Int, lx::Int) = x + 1 + y * lx

function x_bond_terms(lx::Int, ly::Int; t::Float64, qx::Float64=0.0, qy::Float64=0.0, current::Bool=true)
    terms = Tuple{Int,Int,ComplexF64}[]
    for y in 0:(ly-1), x in 0:(lx-1)
        src = site_index(x, y, lx)
        dst = site_index(mod(x + 1, lx), y, lx)
        phase = cis(qx * x + qy * y)
        if current
            # j_x(r) = i t exp(i q r) [c†_{r+x} c_r - c†_r c_{r+x}]
            push!(terms, (dst, src, 1im * t * phase))
            push!(terms, (src, dst, -1im * t * phase))
        else
            # K_x = -t Σ_r [c†_{r+x} c_r + c†_r c_{r+x}]
            push!(terms, (dst, src, ComplexF64(-t)))
            push!(terms, (src, dst, ComplexF64(-t)))
        end
    end
    return terms
end

function static_integrated_response(evals::Vector{Float64}, Avec::AbstractMatrix{<:Complex}, β::Float64, e0::Float64)
    total = 0.0
    n = length(evals)
    @inbounds for m in 1:n, nidx in 1:n
        em = evals[m]
        en = evals[nidx]
        denom = em - en
        kernel = if abs(denom) < 1e-10
            β * exp(-β * (em - e0))
        else
            (exp(-β * (en - e0)) - exp(-β * (em - e0))) / denom
        end
        total += abs2(Avec[m, nidx]) * kernel
    end
    return total
end

function thermal_expectation(evals::Vector{Float64}, Oeig_diag::Vector{Float64}, β::Float64, e0::Float64)
    total = 0.0
    @inbounds for i in eachindex(evals)
        total += exp(-β * (evals[i] - e0)) * Oeig_diag[i]
    end
    return total
end

function double_occupancy_diagonal(up_basis::Vector{UInt64}, dn_basis::Vector{UInt64})
    dim_up = length(up_basis); dim_dn = length(dn_basis)
    diag = Vector{Float64}(undef, dim_up * dim_dn)
    idx = 1
    # Ordering matches kron(I_dn, hup) + kron(hdn, I_up): up index is fastest.
    for dn in dn_basis, up in up_basis
        diag[idx] = count_ones(up & dn)
        idx += 1
    end
    return diag
end

function logsumexp(logw::Vector{Float64})
    m = maximum(logw)
    return m + log(sum(exp.(logw .- m)))
end

function hopping_matrix_smoqy_square(lx::Int, ly::Int, t::Float64)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for y in 0:(ly-1), x in 0:(lx-1)
        i = site_index(x, y, lx)
        jx = site_index(mod(x + 1, lx), y, lx)
        jy = site_index(x, mod(y + 1, ly), lx)
        # SmoQyDQMC TightBindingModel uses directed +x/+y hopping definitions
        # and then includes the Hermitian conjugate for each definition.
        push!(rows, jx); push!(cols, i); push!(vals, -t)
        push!(rows, i); push!(cols, jx); push!(vals, -t)
        push!(rows, jy); push!(cols, i); push!(vals, -t)
        push!(rows, i); push!(cols, jy); push!(vals, -t)
    end
    return Matrix(sparse(rows, cols, vals, lx * ly, lx * ly))
end

function hopping_matrix_for_params(params, lx::Int, ly::Int)
    if params["hopping_convention"] == "smoqy"
        return hopping_matrix_smoqy_square(lx, ly, params["t"])
    else
        return hopping_matrix_Hubbard_2d(lx, ly, params["t"])
    end
end

function y_bond_terms(lx::Int, ly::Int; t::Float64, qx::Float64=0.0, qy::Float64=0.0, current::Bool=true)
    terms = Tuple{Int,Int,ComplexF64}[]
    for y in 0:(ly-1), x in 0:(lx-1)
        src = site_index(x, y, lx)
        dst = site_index(x, mod(y + 1, ly), lx)
        phase = cis(qx * x + qy * y)
        if current
            push!(terms, (dst, src, 1im * t * phase))
            push!(terms, (src, dst, -1im * t * phase))
        else
            push!(terms, (dst, src, ComplexF64(-t)))
            push!(terms, (src, dst, ComplexF64(-t)))
        end
    end
    return terms
end

function main()
    params = parse_args(ARGS)
    lx = params["lx"]; ly = params["ly"]; nsites = lx * ly
    U = params["u"]; μ = params["mu"]; β = params["beta"]
    tmat = hopping_matrix_for_params(params, lx, ly)

    energies = Float64[]
    ntotals = Float64[]
    double_occs = Float64[]
    sector_rows = String[]
    sector_data = Vector{NamedTuple}()

    for nup in 0:nsites, ndn in 0:nsites
        up_basis = generate_basis(nsites, nup)
        dn_basis = generate_basis(nsites, ndn)
        dim_up = length(up_basis); dim_dn = length(dn_basis); dim = dim_up * dim_dn

        hup = one_spin_hamiltonian(tmat, up_basis)
        hdn = one_spin_hamiltonian(tmat, dn_basis)
        kfull = kron(sparse(I, dim_dn, dim_dn), hup) + kron(hdn, sparse(I, dim_up, dim_up))
        docc_diag = double_occupancy_diagonal(up_basis, dn_basis)

        if params["ph_sym_form"]
            sector_constant = -0.5 * U * (nup + ndn) + 0.25 * U * nsites - μ * (nup + ndn)
            diag_potential = U .* docc_diag .+ sector_constant
        else
            sector_constant = -μ * (nup + ndn)
            diag_potential = U .* docc_diag .+ sector_constant
        end

        hfull = kfull + spdiagm(0 => diag_potential)
        F = eigen(Symmetric(Matrix(hfull)))
        evals = F.values
        vecs = F.vectors
        docc_eig = vec(sum(abs2.(vecs) .* reshape(docc_diag, :, 1), dims=1))

        append!(energies, evals)
        append!(ntotals, fill(float(nup + ndn), length(evals)))
        append!(double_occs, docc_eig)

        push!(sector_data, (nup=nup, ndn=ndn, dim=dim, up_basis=up_basis, dn_basis=dn_basis, evals=evals, vecs=vecs))
        push!(sector_rows, @sprintf("%d\t%d\t%d\t%.12f\t%.12f", nup, ndn, dim, minimum(evals), maximum(evals)))
        @printf("sector nup=%d ndn=%d dim=%d Emin=%.12f\n", nup, ndn, dim, minimum(evals))
    end

    logw = -β .* energies
    logZ = logsumexp(logw)
    w = exp.(logw .- logZ)

    E = sum(w .* energies)
    N = sum(w .* ntotals)
    N2 = sum(w .* ntotals .* ntotals)
    D = sum(w .* double_occs)
    density = N / nsites
    double_occ_per_site = D / nsites
    compressibility = β / nsites * (N2 - N^2)

    e0 = minimum(energies)
    Zshift = sum(exp.(-β .* (energies .- e0)))
    qmin = 2π / lx
    lambda_l_num = 0.0
    lambda_t_num = 0.0
    kx_num = 0.0
    ky_num = 0.0

    for sector in sector_data
        dim_up = length(sector.up_basis)
        dim_dn = length(sector.dn_basis)
        Iup = sparse(I, dim_up, dim_up)
        Idn = sparse(I, dim_dn, dim_dn)

        jx_l_up = one_spin_bilinear_operator(nsites, sector.up_basis, x_bond_terms(lx, ly; t=params["t"], qx=qmin, qy=0.0, current=true))
        jx_l_dn = one_spin_bilinear_operator(nsites, sector.dn_basis, x_bond_terms(lx, ly; t=params["t"], qx=qmin, qy=0.0, current=true))
        jx_t_up = one_spin_bilinear_operator(nsites, sector.up_basis, x_bond_terms(lx, ly; t=params["t"], qx=0.0, qy=qmin, current=true))
        jx_t_dn = one_spin_bilinear_operator(nsites, sector.dn_basis, x_bond_terms(lx, ly; t=params["t"], qx=0.0, qy=qmin, current=true))
        kx_up = one_spin_bilinear_operator(nsites, sector.up_basis, x_bond_terms(lx, ly; t=params["t"], current=false))
        kx_dn = one_spin_bilinear_operator(nsites, sector.dn_basis, x_bond_terms(lx, ly; t=params["t"], current=false))
        ky_up = one_spin_bilinear_operator(nsites, sector.up_basis, y_bond_terms(lx, ly; t=params["t"], current=false))
        ky_dn = one_spin_bilinear_operator(nsites, sector.dn_basis, y_bond_terms(lx, ly; t=params["t"], current=false))

        Jl = kron(Idn, jx_l_up) + kron(jx_l_dn, Iup)
        Jt = kron(Idn, jx_t_up) + kron(jx_t_dn, Iup)
        Kx = kron(Idn, kx_up) + kron(kx_dn, Iup)
        Ky = kron(Idn, ky_up) + kron(ky_dn, Iup)

        Vl = sector.vecs' * Matrix(Jl) * sector.vecs
        Vt = sector.vecs' * Matrix(Jt) * sector.vecs
        Kxeig = real.(diag(sector.vecs' * Matrix(Kx) * sector.vecs))
        Kyeig = real.(diag(sector.vecs' * Matrix(Ky) * sector.vecs))

        lambda_l_num += static_integrated_response(sector.evals, Vl, β, e0)
        lambda_t_num += static_integrated_response(sector.evals, Vt, β, e0)
        kx_num += thermal_expectation(sector.evals, Kxeig, β, e0)
        ky_num += thermal_expectation(sector.evals, Kyeig, β, e0)
    end

    lambda_l = lambda_l_num / (Zshift * nsites)
    lambda_t = lambda_t_num / (Zshift * nsites)
    kx_per_site = kx_num / (Zshift * nsites)
    ky_bond_observable_per_site = ky_num / (Zshift * nsites)
    rho_s_current = 0.25 * (lambda_l - lambda_t)
    rho_s_diamagnetic = 0.25 * (-kx_per_site - lambda_t)

    if params["ph_sym_form"]
        interaction_ph = U * (D - 0.5 * N + 0.25 * nsites)
        mu_energy = -μ * N
        kinetic = E - interaction_ph - mu_energy
    else
        interaction_ph = U * D
        mu_energy = -μ * N
        kinetic = E - interaction_ph - mu_energy
    end

    physical_kinetic_per_site = kinetic / nsites
    physical_ky_per_site = physical_kinetic_per_site - kx_per_site

    root = normpath(joinpath(@__DIR__, "..", ".."))
    outdir = normpath(joinpath(root, params["outdir"]))
    mkpath(outdir)

    metadata = Dict(
        "model" => "spinful Hubbard grand-canonical ED",
        "boundary" => "periodic",
        "lattice" => Dict("lx" => lx, "ly" => ly, "nsites" => nsites),
        "hamiltonian" => Dict("t" => params["t"], "tprime" => params["tprime"], "u" => U, "mu" => μ, "ph_sym_form" => params["ph_sym_form"], "hopping_convention" => params["hopping_convention"]),
        "beta" => β,
        "hilbert_space" => Dict("total_dimension_all_sectors" => length(energies), "max_sector_dimension" => maximum(parse.(Int, getindex.(split.(sector_rows, '\t'), 3)))),
    )
    open(joinpath(outdir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end

    open(joinpath(outdir, "summary.tsv"), "w") do io
        println(io, "beta	mu	t	tp	U	ph_sym_form	logZ	total_energy	kinetic_energy	interaction_energy	chemical_potential_energy	N	density	double_occupancy	double_occupancy_per_site	compressibility	Kx_per_site_physical	Ky_per_site_physical	lambda_longitudinal_qmin0	lambda_transverse_0qmin	rho_s_current	Kx_per_site	diamagnetic_minus_Kx_per_site	rho_s_diamagnetic")
        values = Any[β, μ, params["t"], params["tprime"], U, string(params["ph_sym_form"]), logZ, E, kinetic, interaction_ph, mu_energy, N, density, D, double_occ_per_site, compressibility, kx_per_site, physical_ky_per_site, lambda_l, lambda_t, rho_s_current, kx_per_site, -kx_per_site, rho_s_diamagnetic]
        println(io, join([v isa AbstractFloat ? @sprintf("%.12f", v) : string(v) for v in values], '	'))
    end

    open(joinpath(outdir, "sector_spectrum_bounds.tsv"), "w") do io
        println(io, "nup\tndn\tdimension\tmin_eigenvalue\tmax_eigenvalue")
        for row in sector_rows
            println(io, row)
        end
    end

    println("Grand-canonical ED complete")
    println("  output directory = $outdir")
    @printf("  beta = %.6f, mu = %.6f, U = %.6f\n", β, μ, U)
    @printf("  E = %.12f, E/site = %.12f\n", E, E/nsites)
    @printf("  N = %.12f, density = %.12f\n", N, density)
    @printf("  double_occ/site = %.12f, compressibility = %.12f\n", double_occ_per_site, compressibility)
    @printf("  Kx/site = %.12f, Ky/site = %.12f\n", kx_per_site, physical_ky_per_site)
    @printf("  Lambda_L = %.12f, Lambda_T = %.12f, rho_s = %.12f\n", lambda_l, lambda_t, rho_s_current)
    @printf("  -Kx/N = %.12f, rho_s_diamagnetic = %.12f\n", -kx_per_site, rho_s_diamagnetic)
end

main()
