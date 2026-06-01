#!/usr/bin/env julia

using LinearAlgebra
using Random
using Statistics
using TOML
using CanEnsAFQMC

include(joinpath(@__DIR__, "ce_unequal_time_current_helpers.jl"))

function parse_args(args)
    params = Dict(
        "lx" => 3,
        "ly" => 3,
        "nup" => 4,
        "ndn" => 4,
        "u" => -5.0,
        "dtau" => 0.025,
        "beta" => 10.0,
        "nwarmups" => 64,
        "batch_nsamples" => 4,
        "measure_interval" => 4,
        "stab_interval" => 10,
        "cluster_size" => 3,
        "num_fourier_points" => 10,
        "use_lowrank" => false,
        "seed" => 1234,
        "max_batches" => 1,
        "use_charge_hs" => false,
        "sys_type" => "complex",
        "tau_slices" => [1, 10, 40, 200],
        "output_dir" => joinpath("results", "interacting_qmc_ed", "ce_unequaltime_greens_exactsector_3x3"),
    )
    for arg in args
        if startswith(arg, "--batch-nsamples=")
            params["batch_nsamples"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--max-batches=")
            params["max_batches"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--nwarmups=")
            params["nwarmups"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--measure-interval=")
            params["measure_interval"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dtau=")
            params["dtau"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--u=")
            params["u"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--beta=")
            params["beta"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            params["seed"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num-fourier-points=")
            params["num_fourier_points"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use-lowrank=")
            params["use_lowrank"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use-charge-hs=")
            params["use_charge_hs"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sys-type=")
            params["sys_type"] = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tau-slices=")
            params["tau_slices"] = [parse(Int, x) for x in split(split(arg, "=", limit=2)[2], ",") if !isempty(x)]
        elseif startswith(arg, "--output-dir=")
            params["output_dir"] = split(arg, "=", limit=2)[2]
        end
    end
    return params
end

function build_system(params)
    tmat = hopping_matrix_Hubbard_2d(params["lx"], params["ly"], 1.0)
    L = round(Int, params["beta"] / params["dtau"])
    sys_type = params["sys_type"] == "complex" ? ComplexF64 : Float64
    return GenericHubbard(
        (params["lx"], params["ly"], 1),
        (params["nup"], params["ndn"]),
        tmat,
        params["u"],
        0.0,
        params["beta"],
        L,
        sys_type=sys_type,
        useChargeHST=params["use_charge_hs"],
        useFirstOrderTrotter=false,
    )
end

function build_qmc(system, params)
    return QMC(
        system,
        nwarmups=params["nwarmups"],
        nsamples=params["batch_nsamples"],
        measure_interval=params["measure_interval"],
        stab_interval=params["stab_interval"],
        useClusterUpdate=true,
        cluster_size=params["cluster_size"],
        num_FourierPoints=params["num_fourier_points"],
        forceSymmetry=true,
        isLowrank=params["use_lowrank"],
        lrThld=1e-10,
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

function build_prefix_suffix_dense(Ms::Vector{Matrix{ComplexF64}})
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

function annihilation_N_from_Np1(site::Int, basisN::Vector{Int}, basisNp1::Vector{Int})
    idxN = Dict(state => i for (i, state) in enumerate(basisN))
    M = zeros(ComplexF64, length(basisN), length(basisNp1))
    for (col, state) in enumerate(basisNp1)
        occ(state, site) == 0 && continue
        mask = 1 << (site - 1)
        new_state = state & ~mask
        sign = isodd(count_ones(state & (mask - 1))) ? -1 : 1
        row = idxN[new_state]
        M[row, col] = sign
    end
    return M
end

function exact_local_gτ0(MN_prefix, MN_suffix, MNP1_prefix, CopsN, CdopsN, slice::Int, V::Int)
    ZN = tr(MN_prefix[end])
    VN = MN_suffix[slice + 1]
    UNp1 = MNP1_prefix[slice + 1]
    total = zero(ComplexF64)
    for site in 1:V
        total += tr(VN * CopsN[site] * UNp1 * CdopsN[site])
    end
    return total / ZN / V
end

function exact_addition_matrix_gτ0(MN_prefix, MN_suffix, MNP1_prefix, CopsN, CdopsN, slice::Int, V::Int)
    ZN = tr(MN_prefix[end])
    VN = MN_suffix[slice + 1]
    UNp1 = MNP1_prefix[slice + 1]
    G = zeros(ComplexF64, V, V)
    @inbounds for j in 1:V, i in 1:V
        G[i, j] = tr(VN * CopsN[i] * UNp1 * CdopsN[j]) / ZN
    end
    return G
end

function exact_removal_matrix_gτ0(MN_prefix, MN_suffix, MNM1_prefix, CopsNm1, CdopsNm1, slice::Int, V::Int)
    ZN = tr(MN_prefix[end])
    VN = MN_suffix[slice + 1]
    UNm1 = MNM1_prefix[slice + 1]
    G = zeros(ComplexF64, V, V)
    @inbounds for j in 1:V, i in 1:V
        G[i, j] = tr(VN * CdopsNm1[j] * UNm1 * CopsNm1[i]) / ZN
    end
    return G
end

function write_metadata(outdir, params)
    metadata = Dict(
        "target" => "3x3 CE unequal-time single-particle Green benchmark",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"]),
        "model" => Dict(
            "u" => params["u"],
            "dtau" => params["dtau"],
            "beta" => params["beta"],
            "use_charge_hs" => params["use_charge_hs"],
            "sys_type" => params["sys_type"],
            "use_lowrank" => params["use_lowrank"],
        ),
        "tau_slices" => params["tau_slices"],
    )
    open(joinpath(outdir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function write_summary(outdir, params, tau_slices, exact_mean, exact_err, formula_mean, formula_err, diff_mean, diff_err, nsamples, batches)
    open(joinpath(outdir, "greens_tau0_summary.tsv"), "w") do io
        println(io, "slice\ttau\texact_local_Gtau0_real\texact_stderr\tformula_local_Gtau0_real\tformula_stderr\tdelta_formula_minus_exact\tdelta_stderr\tnsamples\tbatches")
        for (k, slice) in enumerate(tau_slices)
            tau = slice * params["dtau"]
            println(io,
                string(
                    slice, '\t', tau, '\t',
                    exact_mean[k], '\t', exact_err[k], '\t',
                    formula_mean[k], '\t', formula_err[k], '\t',
                    diff_mean[k], '\t', diff_err[k], '\t',
                    nsamples, '\t', batches
                )
            )
        end
    end
end

function mean_complex(all_values::Array{ComplexF64,4})
    return dropdims(mean(all_values, dims=1), dims=1)
end

function stderr_reim(all_values::Array{ComplexF64,4})
    nsamples = size(all_values, 1)
    re_err = dropdims(std(real.(all_values), dims=1; corrected=true), dims=1) ./ sqrt(nsamples)
    im_err = dropdims(std(imag.(all_values), dims=1; corrected=true), dims=1) ./ sqrt(nsamples)
    return re_err, im_err
end

function write_realspace_comparison(outdir, filename, params, tau_slices, exact_mean, formula_mean, diff_mean, diff_re_err, diff_im_err, nsamples, batches)
    ntau, lx, ly = size(exact_mean)
    open(joinpath(outdir, filename), "w") do io
        println(io, "slice\ttau\tdx\tdy\texact_re\texact_im\tformula_re\tformula_im\tdelta_re\tdelta_re_stderr\tdelta_im\tdelta_im_stderr\tdelta_abs\tnsamples\tbatches")
        for k in 1:ntau, dy in 0:(ly - 1), dx in 0:(lx - 1)
            exact = exact_mean[k, dx + 1, dy + 1]
            formula = formula_mean[k, dx + 1, dy + 1]
            diff = diff_mean[k, dx + 1, dy + 1]
            println(io, string(
                tau_slices[k], '\t', tau_slices[k] * params["dtau"], '\t',
                dx, '\t', dy, '\t',
                real(exact), '\t', imag(exact), '\t',
                real(formula), '\t', imag(formula), '\t',
                real(diff), '\t', diff_re_err[k, dx + 1, dy + 1], '\t',
                imag(diff), '\t', diff_im_err[k, dx + 1, dy + 1], '\t',
                abs(diff), '\t', nsamples, '\t', batches
            ))
        end
    end
end

function write_momentum_comparison(outdir, filename, params, tau_slices, exact_mean, formula_mean, diff_mean, diff_re_err, diff_im_err, nsamples, batches)
    ntau, lx, ly = size(exact_mean)
    open(joinpath(outdir, filename), "w") do io
        println(io, "slice\ttau\tnx\tny\tkx\tky\texact_re\texact_im\tformula_re\tformula_im\tdelta_re\tdelta_re_stderr\tdelta_im\tdelta_im_stderr\tdelta_abs\tnsamples\tbatches")
        for k in 1:ntau, ny in 0:(ly - 1), nx in 0:(lx - 1)
            kx = 2π * nx / lx
            ky = 2π * ny / ly
            exact = exact_mean[k, nx + 1, ny + 1]
            formula = formula_mean[k, nx + 1, ny + 1]
            diff = diff_mean[k, nx + 1, ny + 1]
            println(io, string(
                tau_slices[k], '\t', tau_slices[k] * params["dtau"], '\t',
                nx, '\t', ny, '\t', kx, '\t', ky, '\t',
                real(exact), '\t', imag(exact), '\t',
                real(formula), '\t', imag(formula), '\t',
                real(diff), '\t', diff_re_err[k, nx + 1, ny + 1], '\t',
                imag(diff), '\t', diff_im_err[k, nx + 1, ny + 1], '\t',
                abs(diff), '\t', nsamples, '\t', batches
            ))
        end
    end
end

function main()
    params = parse_args(ARGS)
    root = normpath(joinpath(@__DIR__, "..", ".."))
    outdir = joinpath(root, params["output_dir"])
    mkpath(outdir)
    write_metadata(outdir, params)

    system = build_system(params)
    qmc = build_qmc(system, params)
    tau_slices = sort(unique(params["tau_slices"]))
    any(s -> s < 1 || s >= system.L, tau_slices) && error("tau slices must satisfy 1 <= slice < L")

    basisN = gen_basis(system.V, system.N[1])
    basisNp1 = gen_basis(system.V, system.N[1] + 1)
    basisNm1 = gen_basis(system.V, system.N[1] - 1)
    occN = [occupied_sites(state, system.V) for state in basisN]
    occNp1 = [occupied_sites(state, system.V) for state in basisNp1]
    occNm1 = [occupied_sites(state, system.V) for state in basisNm1]
    CopsN = [annihilation_N_from_Np1(site, basisN, basisNp1) for site in 1:system.V]
    CdopsN = [adjoint(C) for C in CopsN]
    CopsNm1 = [annihilation_N_from_Np1(site, basisNm1, basisN) for site in 1:system.V]
    CdopsNm1 = [adjoint(C) for C in CopsNm1]

    all_exact = Matrix{Float64}(undef, 0, length(tau_slices))
    all_formula = Matrix{Float64}(undef, 0, length(tau_slices))
    lx, ly, _ = system.Ns
    all_exact_add_r = Array{ComplexF64}(undef, 0, length(tau_slices), lx, ly)
    all_formula_add_r = Array{ComplexF64}(undef, 0, length(tau_slices), lx, ly)
    all_exact_add_k = Array{ComplexF64}(undef, 0, length(tau_slices), lx, ly)
    all_formula_add_k = Array{ComplexF64}(undef, 0, length(tau_slices), lx, ly)
    all_exact_rem_r = Array{ComplexF64}(undef, 0, length(tau_slices), lx, ly)
    all_formula_rem_r = Array{ComplexF64}(undef, 0, length(tau_slices), lx, ly)
    all_exact_rem_k = Array{ComplexF64}(undef, 0, length(tau_slices), lx, ly)
    all_formula_rem_k = Array{ComplexF64}(undef, 0, length(tau_slices), lx, ly)

    for batch in 1:params["max_batches"]
        Random.seed!(params["seed"] + batch - 1)
        walker = Walker(system, qmc)
        sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

        batch_exact = zeros(Float64, qmc.nsamples, length(tau_slices))
        batch_formula = zeros(Float64, qmc.nsamples, length(tau_slices))
        batch_exact_add_r = zeros(ComplexF64, qmc.nsamples, length(tau_slices), lx, ly)
        batch_formula_add_r = zeros(ComplexF64, qmc.nsamples, length(tau_slices), lx, ly)
        batch_exact_add_k = zeros(ComplexF64, qmc.nsamples, length(tau_slices), lx, ly)
        batch_formula_add_k = zeros(ComplexF64, qmc.nsamples, length(tau_slices), lx, ly)
        batch_exact_rem_r = zeros(ComplexF64, qmc.nsamples, length(tau_slices), lx, ly)
        batch_formula_rem_r = zeros(ComplexF64, qmc.nsamples, length(tau_slices), lx, ly)
        batch_exact_rem_k = zeros(ComplexF64, qmc.nsamples, length(tau_slices), lx, ly)
        batch_formula_rem_k = zeros(ComplexF64, qmc.nsamples, length(tau_slices), lx, ly)

        for sample in 1:qmc.nsamples
            sweep!(system, qmc, walker, loop_number=qmc.measure_interval)

            Bup, _ = build_B_slices(system, walker)
            prefix_up, suffix_up = build_prefix_suffix(Bup)
            cache_up = CanonicalUnequalTimeGreenCache(system, prefix_up, 1)

            MsN = [manybody_propagator_sector(B, occN) for B in Bup]
            MsNp1 = [manybody_propagator_sector(B, occNp1) for B in Bup]
            MsNm1 = [manybody_propagator_sector(B, occNm1) for B in Bup]
            prefixN, suffixN = build_prefix_suffix_dense(MsN)
            prefixNp1, _ = build_prefix_suffix_dense(MsNp1)
            prefixNm1, _ = build_prefix_suffix_dense(MsNm1)

            for (k, slice) in enumerate(tau_slices)
                exact_add = exact_addition_matrix_gτ0(prefixN, suffixN, prefixNp1, CopsN, CdopsN, slice, system.V)
                exact_rem = exact_removal_matrix_gτ0(prefixN, suffixN, prefixNm1, CopsNm1, CdopsNm1, slice, system.V)
                formula_add, formula_rem = canonical_unequal_time_greens(
                    system, cache_up, prefix_up, suffix_up, slice,
                    physical_normalization=false, spin=1,
                )
                g_exact = tr(exact_add) / system.V
                g_formula = tr(formula_add) / system.V
                batch_exact[sample, k] = real(g_exact)
                batch_formula[sample, k] = real(g_formula)

                exact_add_r = green_realspace_average(system, exact_add)
                formula_add_r = green_realspace_average(system, formula_add)
                exact_rem_r = green_realspace_average(system, exact_rem)
                formula_rem_r = green_realspace_average(system, formula_rem)
                batch_exact_add_r[sample, k, :, :] .= exact_add_r
                batch_formula_add_r[sample, k, :, :] .= formula_add_r
                batch_exact_add_k[sample, k, :, :] .= green_momentum_from_realspace(exact_add_r)
                batch_formula_add_k[sample, k, :, :] .= green_momentum_from_realspace(formula_add_r)
                batch_exact_rem_r[sample, k, :, :] .= exact_rem_r
                batch_formula_rem_r[sample, k, :, :] .= formula_rem_r
                batch_exact_rem_k[sample, k, :, :] .= green_momentum_from_realspace(exact_rem_r)
                batch_formula_rem_k[sample, k, :, :] .= green_momentum_from_realspace(formula_rem_r)
            end
        end

        all_exact = vcat(all_exact, batch_exact)
        all_formula = vcat(all_formula, batch_formula)
        all_exact_add_r = cat(all_exact_add_r, batch_exact_add_r; dims=1)
        all_formula_add_r = cat(all_formula_add_r, batch_formula_add_r; dims=1)
        all_exact_add_k = cat(all_exact_add_k, batch_exact_add_k; dims=1)
        all_formula_add_k = cat(all_formula_add_k, batch_formula_add_k; dims=1)
        all_exact_rem_r = cat(all_exact_rem_r, batch_exact_rem_r; dims=1)
        all_formula_rem_r = cat(all_formula_rem_r, batch_formula_rem_r; dims=1)
        all_exact_rem_k = cat(all_exact_rem_k, batch_exact_rem_k; dims=1)
        all_formula_rem_k = cat(all_formula_rem_k, batch_formula_rem_k; dims=1)
        exact_mean = vec(mean(all_exact, dims=1))
        formula_mean = vec(mean(all_formula, dims=1))
        exact_err = vec(std(all_exact, dims=1; corrected=true)) ./ sqrt(size(all_exact, 1))
        formula_err = vec(std(all_formula, dims=1; corrected=true)) ./ sqrt(size(all_formula, 1))
        diff = all_formula .- all_exact
        diff_mean = vec(mean(diff, dims=1))
        diff_err = vec(std(diff, dims=1; corrected=true)) ./ sqrt(size(diff, 1))
        write_summary(outdir, params, tau_slices, exact_mean, exact_err, formula_mean, formula_err, diff_mean, diff_err, size(all_exact, 1), batch)

        exact_add_r_mean = mean_complex(all_exact_add_r)
        formula_add_r_mean = mean_complex(all_formula_add_r)
        diff_add_r = all_formula_add_r .- all_exact_add_r
        diff_add_r_mean = mean_complex(diff_add_r)
        diff_add_r_re_err, diff_add_r_im_err = stderr_reim(diff_add_r)

        exact_add_k_mean = mean_complex(all_exact_add_k)
        formula_add_k_mean = mean_complex(all_formula_add_k)
        diff_add_k = all_formula_add_k .- all_exact_add_k
        diff_add_k_mean = mean_complex(diff_add_k)
        diff_add_k_re_err, diff_add_k_im_err = stderr_reim(diff_add_k)

        exact_rem_r_mean = mean_complex(all_exact_rem_r)
        formula_rem_r_mean = mean_complex(all_formula_rem_r)
        diff_rem_r = all_formula_rem_r .- all_exact_rem_r
        diff_rem_r_mean = mean_complex(diff_rem_r)
        diff_rem_r_re_err, diff_rem_r_im_err = stderr_reim(diff_rem_r)

        exact_rem_k_mean = mean_complex(all_exact_rem_k)
        formula_rem_k_mean = mean_complex(all_formula_rem_k)
        diff_rem_k = all_formula_rem_k .- all_exact_rem_k
        diff_rem_k_mean = mean_complex(diff_rem_k)
        diff_rem_k_re_err, diff_rem_k_im_err = stderr_reim(diff_rem_k)

        write_realspace_comparison(outdir, "greens_r_tau_add_exactsector.tsv", params, tau_slices, exact_add_r_mean, formula_add_r_mean, diff_add_r_mean, diff_add_r_re_err, diff_add_r_im_err, size(all_exact, 1), batch)
        write_momentum_comparison(outdir, "greens_k_tau_add_exactsector.tsv", params, tau_slices, exact_add_k_mean, formula_add_k_mean, diff_add_k_mean, diff_add_k_re_err, diff_add_k_im_err, size(all_exact, 1), batch)
        write_realspace_comparison(outdir, "greens_r_tau_remove_exactsector.tsv", params, tau_slices, exact_rem_r_mean, formula_rem_r_mean, diff_rem_r_mean, diff_rem_r_re_err, diff_rem_r_im_err, size(all_exact, 1), batch)
        write_momentum_comparison(outdir, "greens_k_tau_remove_exactsector.tsv", params, tau_slices, exact_rem_k_mean, formula_rem_k_mean, diff_rem_k_mean, diff_rem_k_re_err, diff_rem_k_im_err, size(all_exact, 1), batch)

        max_add_k_delta = maximum(abs.(diff_add_k_mean))
        max_add_r_delta = maximum(abs.(diff_add_r_mean))
        println("batch=", batch, " nsamples=", size(all_exact, 1),
            " tau1_local_delta=", diff_mean[1],
            " max_add_r_delta=", max_add_r_delta,
            " max_add_k_delta=", max_add_k_delta)
    end
end

main()
