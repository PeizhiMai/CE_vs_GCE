#!/usr/bin/env julia

using DelimitedFiles
using Random
using Statistics
using TOML
using CanEnsAFQMC

function parse_args(args)
    params = Dict(
        "lx" => 3,
        "ly" => 3,
        "nup" => 4,
        "ndn" => 4,
        "u" => -5.0,
        "dtau" => 0.05,
        "beta" => 10.0,
        "nwarmups" => 256,
        "batch_nsamples" => 256,
        "measure_interval" => 4,
        "stab_interval" => 10,
        "cluster_size" => 3,
        "num_fourier_points" => 10,
        "lr_threshold" => 1.0e-10,
        "seed" => 1234,
        "stderr_target" => 0.003,
        "max_batches" => 12,
        "use_charge_hs" => false,
        "sys_type" => "complex",
        "output_dir" => joinpath("results", "interacting_qmc_ed", "ce_qmc_vs_ed_3x3_beta10"),
        "ed_dir" => joinpath("results", "interacting_qmc_ed", "ed_ce_lowtemp_3x3_t1_Um5_Nup4_Ndn4_beta10"),
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
        elseif startswith(arg, "--beta=")
            params["beta"] = parse(Float64, split(arg, "=", limit=2)[2])
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
        elseif startswith(arg, "--use-charge-hs=")
            params["use_charge_hs"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sys-type=")
            params["sys_type"] = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output-dir=")
            params["output_dir"] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--ed-dir=")
            params["ed_dir"] = split(arg, "=", limit=2)[2]
        else
            error("Unrecognized argument: $arg")
        end
    end
    return params
end

function build_system(params)
    tmat = hopping_matrix_Hubbard_2d(params["lx"], params["ly"], 1.0)
    time_slices = round(Int, params["beta"] / params["dtau"])
    isapprox(params["beta"] / params["dtau"], time_slices; atol=1e-10) || error("beta/dtau must be integer")
    sys_type = params["sys_type"] == "complex" ? ComplexF64 : Float64

    return GenericHubbard(
        (params["lx"], params["ly"], 1),
        (params["nup"], params["ndn"]),
        tmat,
        params["u"],
        0.0,
        params["beta"],
        time_slices,
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
        isLowrank=true,
        lrThld=params["lr_threshold"],
        saveRatio=false,
    )
end

function measure_batch(system, qmc)
    walker = Walker(system, qmc)
    ρup = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    ρdn = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    corr = CorrFuncSampler(system, qmc)

    nsites = system.V
    ntotal = sum(system.N)

    energies = zeros(Float64, qmc.nsamples, 3)
    docc = zeros(Float64, qmc.nsamples)
    charge = zeros(Float64, length(corr.δr), qmc.nsamples)
    spinz = zeros(Float64, length(corr.δr), qmc.nsamples)
    pair = zeros(Float64, length(corr.δr), qmc.nsamples)

    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

    for sample in 1:qmc.nsamples
        sweep!(system, qmc, walker, loop_number=qmc.measure_interval)
        update!(system, walker, ρup, 1)
        update!(system, walker, ρdn, ρup)

        energies[sample, :] .= real.(measure_Energy(system, ρup, ρdn))
        docc[sample] = sum(real.(diag(ρup.ρ₁) .* diag(ρdn.ρ₁))) / nsites

        measure_ChargeCorr(corr, ρup, ρdn)
        measure_SpinCorr(corr, ρup, ρdn)
        measure_PairCorr(corr, ρup, ρdn, addCount=true)

        charge[:, sample] .= real.(corr.nᵢ₊ᵣnᵢ[:, sample])
        spinz[:, sample] .= real.(corr.Sᵢ₊ᵣSᵢ[:, sample])
        pair[:, sample] .= real.(corr.Pₛ[:, sample])
    end

    return (
        deltas=collect(corr.δr),
        energies=energies ./ nsites,
        docc=docc,
        charge=charge,
        spinz=spinz,
        pair=pair,
        ntotal=ntotal,
        time_slices=system.L,
    )
end

stderr(v) = std(v; corrected=true) / sqrt(length(v))

function summarize(all_energy, all_docc, all_charge, all_spinz, all_pair)
    energy_mean = vec(mean(all_energy, dims=1))
    energy_err = vec(std(all_energy, dims=1; corrected=true)) ./ sqrt(size(all_energy, 1))
    docc_mean = mean(all_docc)
    docc_err = stderr(all_docc)

    nδ = size(all_charge, 1)
    charge_mean = [mean(view(all_charge, i, :)) for i in 1:nδ]
    charge_err = [stderr(vec(view(all_charge, i, :))) for i in 1:nδ]
    spinz_mean = [mean(view(all_spinz, i, :)) for i in 1:nδ]
    spinz_err = [stderr(vec(view(all_spinz, i, :))) for i in 1:nδ]
    pair_mean = [mean(view(all_pair, i, :)) for i in 1:nδ]
    pair_err = [stderr(vec(view(all_pair, i, :))) for i in 1:nδ]

    return (
        energy_mean=energy_mean,
        energy_err=energy_err,
        docc_mean=docc_mean,
        docc_err=docc_err,
        charge_mean=charge_mean,
        charge_err=charge_err,
        spinz_mean=spinz_mean,
        spinz_err=spinz_err,
        pair_mean=pair_mean,
        pair_err=pair_err,
    )
end

function write_metadata(outdir, params)
    metadata = Dict(
        "target" => "3x3 CE QMC vs low-temperature ED benchmark",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"], "ntotal" => params["nup"] + params["ndn"]),
        "model" => Dict(
            "u" => params["u"],
            "dtau" => params["dtau"],
            "beta" => params["beta"],
            "use_charge_hs" => params["use_charge_hs"],
            "sys_type" => params["sys_type"],
        ),
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
        "ed_reference_dir" => params["ed_dir"],
    )
    open(joinpath(outdir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function write_outputs(outdir, params, deltas, summary, nsamples, batches, time_slices)
    open(joinpath(outdir, "summary.tsv"), "w") do io
        println(io, "beta\tkinetic_per_site\tkinetic_stderr\tpotential_per_site\tpotential_stderr\ttotal_per_site\ttotal_stderr\tdouble_occupancy_per_site\tdouble_occupancy_stderr\ttime_slices\tnsamples\tbatches")
        println(io,
            string(
                params["beta"], '\t',
                summary.energy_mean[1], '\t',
                summary.energy_err[1], '\t',
                summary.energy_mean[2], '\t',
                summary.energy_err[2], '\t',
                summary.energy_mean[3], '\t',
                summary.energy_err[3], '\t',
                summary.docc_mean, '\t',
                summary.docc_err, '\t',
                time_slices, '\t',
                nsamples, '\t',
                batches
            )
        )
    end

    open(joinpath(outdir, "correlations.tsv"), "w") do io
        println(io, "dx\tdy\tcharge_corr\tcharge_stderr\tspin_z_corr\tspin_z_stderr\tpair_corr\tpair_stderr")
        for (i, (dx, dy)) in enumerate(deltas)
            println(io,
                string(
                    dx, '\t', dy, '\t',
                    summary.charge_mean[i], '\t',
                    summary.charge_err[i], '\t',
                    summary.spinz_mean[i], '\t',
                    summary.spinz_err[i], '\t',
                    summary.pair_mean[i], '\t',
                    summary.pair_err[i]
                )
            )
        end
    end
end

function main()
    params = parse_args(ARGS)
    root = normpath(joinpath(@__DIR__, "..", ".."))
    outdir = joinpath(root, params["output_dir"])
    mkpath(outdir)
    write_metadata(outdir, params)

    all_energy = Matrix{Float64}(undef, 0, 3)
    all_docc = Float64[]
    all_charge = Matrix{Float64}(undef, 0, 0)
    all_spinz = Matrix{Float64}(undef, 0, 0)
    all_pair = Matrix{Float64}(undef, 0, 0)
    deltas = Tuple{Int, Int}[]
    time_slices = 0

    for batch_idx in 1:params["max_batches"]
        Random.seed!(params["seed"] + batch_idx - 1)
        system = build_system(params)
        qmc = build_qmc(system, params)
        batch = measure_batch(system, qmc)

        if batch_idx == 1
            deltas = batch.deltas
            all_charge = batch.charge
            all_spinz = batch.spinz
            all_pair = batch.pair
        else
            all_charge = hcat(all_charge, batch.charge)
            all_spinz = hcat(all_spinz, batch.spinz)
            all_pair = hcat(all_pair, batch.pair)
        end
        all_energy = vcat(all_energy, batch.energies)
        append!(all_docc, batch.docc)
        time_slices = batch.time_slices

        summary = summarize(all_energy, all_docc, all_charge, all_spinz, all_pair)
        nsamples = size(all_energy, 1)
        write_outputs(outdir, params, deltas, summary, nsamples, batch_idx, time_slices)

        println(
            "batch=", batch_idx,
            " total/site=", summary.energy_mean[3],
            " stderr=", summary.energy_err[3],
            " docc/site=", summary.docc_mean,
            " nsamples=", nsamples,
        )

        summary.energy_err[3] <= params["stderr_target"] && break
    end
end

main()
