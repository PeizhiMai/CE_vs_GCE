#!/usr/bin/env julia

using DelimitedFiles
using Random
using Statistics
using TOML
using CanEnsAFQMC

function parse_args(args)
    params = Dict(
        "lx" => 4,
        "ly" => 2,
        "nup" => 4,
        "ndn" => 4,
        "u" => 4.0,
        "dtau" => 0.1,
        "beta_list" => collect(1.0:1.0:10.0),
        "nwarmups" => 128,
        "batch_nsamples" => 256,
        "measure_interval" => 4,
        "stab_interval" => 10,
        "cluster_size" => 4,
        "num_fourier_points" => 9,
        "lr_threshold" => 1.0e-10,
        "seed" => 1234,
        "stderr_target" => 0.003,
        "max_batches" => 16,
        "output_dir" => joinpath("results", "interacting_qmc_ed", "ce_qmc_vs_ed_4x2_benchmark"),
        "ed_path" => joinpath("results", "interacting_qmc_ed", "ed_hubbard_4x2_half_filling_beta_0p1_grid", "energy_vs_beta.tsv"),
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
        elseif startswith(arg, "--dtau=")
            params["dtau"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--beta-list=")
            raw = split(split(arg, "=", limit=2)[2], ",")
            params["beta_list"] = [parse(Float64, strip(v)) for v in raw if !isempty(strip(v))]
        elseif startswith(arg, "--nwarmups=")
            params["nwarmups"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--batch-nsamples=")
            params["batch_nsamples"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--measure-interval=")
            params["measure_interval"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--stab-interval=")
            params["stab_interval"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--cluster-size=")
            params["cluster_size"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num-fourier-points=")
            params["num_fourier_points"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--lr-threshold=")
            params["lr_threshold"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            params["seed"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--stderr-target=")
            params["stderr_target"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--max-batches=")
            params["max_batches"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output-dir=")
            params["output_dir"] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--ed-path=")
            params["ed_path"] = split(arg, "=", limit=2)[2]
        else
            error("Unrecognized argument: $arg")
        end
    end

    sort!(params["beta_list"])
    return params
end

function build_system(params, beta)
    tmat = hopping_matrix_Hubbard_2d(params["lx"], params["ly"], 1.0)
    time_slices = round(Int, beta / params["dtau"])
    isapprox(beta / params["dtau"], time_slices; atol=1e-10) || error("beta/dtau must be integer")

    return GenericHubbard(
        (params["lx"], params["ly"], 1),
        (params["nup"], params["ndn"]),
        tmat,
        params["u"],
        0.0,
        beta,
        time_slices,
        sys_type=Float64,
        useChargeHST=false,
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
        forceSymmetry=false,
        isLowrank=true,
        lrThld=params["lr_threshold"],
        saveRatio=false,
    )
end

function run_energy_batch(system, qmc)
    walker = Walker(system, qmc)
    rho_up = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    rho_dn = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    energies = zeros(Float64, qmc.nsamples, 3)

    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

    for sample in 1:qmc.nsamples
        sweep!(system, qmc, walker, loop_number=qmc.measure_interval)
        update!(system, walker, rho_up, 1)
        update!(system, walker, rho_dn, 2)
        energies[sample, :] .= real.(measure_Energy(system, rho_up, rho_dn))
    end

    return energies ./ sum(system.N)
end

function run_beta_point(params, beta, point_index)
    all_energies = Matrix{Float64}(undef, 0, 3)
    batch_summaries = Vector{NamedTuple}()

    for batch_idx in 1:params["max_batches"]
        batch_seed = params["seed"] + 1000 * point_index + batch_idx - 1
        Random.seed!(batch_seed)
        system = build_system(params, beta)
        qmc = build_qmc(system, params)
        batch = run_energy_batch(system, qmc)
        all_energies = vcat(all_energies, batch)

        means = vec(mean(all_energies, dims=1))
        stderrs = vec(std(all_energies, dims=1; corrected=true)) ./ sqrt(size(all_energies, 1))

        push!(batch_summaries, (
            batch=batch_idx,
            seed=batch_seed,
            nsamples=size(all_energies, 1),
            kinetic=means[1],
            kinetic_stderr=stderrs[1],
            potential=means[2],
            potential_stderr=stderrs[2],
            total=means[3],
            total_stderr=stderrs[3],
            time_slices=system.L,
        ))

        stderrs[3] <= params["stderr_target"] && break
    end

    final = batch_summaries[end]
    return (
        beta=beta,
        kinetic=final.kinetic,
        kinetic_stderr=final.kinetic_stderr,
        potential=final.potential,
        potential_stderr=final.potential_stderr,
        total=final.total,
        total_stderr=final.total_stderr,
        time_slices=final.time_slices,
        nsamples=final.nsamples,
        batches=length(batch_summaries),
        batch_summaries=batch_summaries,
    )
end

function write_metadata(outdir, params)
    metadata = Dict(
        "target" => "4x2 CE QMC vs ED benchmark",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"], "ntotal" => params["nup"] + params["ndn"]),
        "model" => Dict("u" => params["u"], "dtau" => params["dtau"]),
        "beta_list" => params["beta_list"],
        "run" => Dict(
            "batch_nsamples" => params["batch_nsamples"],
            "stderr_target" => params["stderr_target"],
            "max_batches" => params["max_batches"],
        ),
        "qmc" => Dict(
            "nwarmups" => params["nwarmups"],
            "measure_interval" => params["measure_interval"],
            "stab_interval" => params["stab_interval"],
            "cluster_size" => params["cluster_size"],
            "num_fourier_points" => params["num_fourier_points"],
            "lowrank_threshold" => params["lr_threshold"],
        ),
        "ed_reference" => params["ed_path"],
    )
    open(joinpath(outdir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function initialize_outputs(outdir, params)
    mkpath(outdir)
    write_metadata(outdir, params)
    open(joinpath(outdir, "ce_energy.tsv"), "w") do io
        println(io, "beta\tkinetic_per_particle\tkinetic_stderr\tpotential_per_particle\tpotential_stderr\ttotal_per_particle\ttotal_stderr\ttime_slices\tnsamples\tbatches")
    end
end

function append_result(outdir, result)
    open(joinpath(outdir, "ce_energy.tsv"), "a") do io
        println(io,
            string(
                result.beta, '\t',
                result.kinetic, '\t',
                result.kinetic_stderr, '\t',
                result.potential, '\t',
                result.potential_stderr, '\t',
                result.total, '\t',
                result.total_stderr, '\t',
                result.time_slices, '\t',
                result.nsamples, '\t',
                result.batches
            )
        )
    end
end

function refresh_overlay(outdir, params)
    root = normpath(joinpath(@__DIR__, ".."))
    run(`python3 $(joinpath(root, "scripts", "plot_ed_vs_ce_qmc_4x2.py")) --ed-path $(joinpath(root, params["ed_path"])) --qmc-path $(joinpath(outdir, "ce_energy.tsv")) --outdir $outdir`)
end

function main()
    params = parse_args(ARGS)
    root = normpath(joinpath(@__DIR__, ".."))
    outdir = joinpath(root, params["output_dir"])
    initialize_outputs(outdir, params)

    for (idx, beta) in enumerate(params["beta_list"])
        result = run_beta_point(params, beta, idx)
        append_result(outdir, result)
        refresh_overlay(outdir, params)
        println(
            "beta=", result.beta,
            " total/N=", result.total,
            " stderr=", result.total_stderr,
            " nsamples=", result.nsamples,
            " batches=", result.batches,
        )
    end
end

main()
