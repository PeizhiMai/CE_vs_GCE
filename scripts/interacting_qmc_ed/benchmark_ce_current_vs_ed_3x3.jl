#!/usr/bin/env julia

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
        "nwarmups" => 256,
        "batch_nsamples" => 128,
        "measure_interval" => 4,
        "stab_interval" => 10,
        "cluster_size" => 3,
        "num_fourier_points" => 10,
        "lr_threshold" => 1.0e-10,
        "is_lowrank" => false,
        "sys_type" => "complex",
        "seed" => 1234,
        "stderr_target" => 0.01,
        "max_batches" => 12,
        "use_charge_hs" => false,
        "output_dir" => joinpath("results", "interacting_qmc_ed", "ce_current_vs_ed_3x3_beta10_dtau0025"),
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
        elseif startswith(arg, "--is-lowrank=")
            params["is_lowrank"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sys-type=")
            params["sys_type"] = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            params["seed"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--stderr-target=")
            params["stderr_target"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--max-batches=")
            params["max_batches"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use-charge-hs=")
            params["use_charge_hs"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output-dir=")
            params["output_dir"] = split(arg, "=", limit=2)[2]
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
        isLowrank=params["is_lowrank"],
        lrThld=params["lr_threshold"],
        saveRatio=false,
    )
end

function stderr(v)
    return std(v; corrected=true) / sqrt(length(v))
end

function run_batch(system, qmc)
    walker = Walker(system, qmc)
    ρup = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    ρdn = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    qxmin = 2π / system.Ns[1]
    qymin = 2π / system.Ns[2]

    data = zeros(Float64, qmc.nsamples, 5)
    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

    for sample in 1:qmc.nsamples
        sweep!(system, qmc, walker, loop_number=qmc.measure_interval)
        update!(system, walker, ρup, 1)
        update!(system, walker, ρdn, ρup)

        Bup, Bdn = build_B_slices(system, walker)
        prefix_up, suffix_up = build_prefix_suffix(Bup)
        prefix_dn, suffix_dn = build_prefix_suffix(Bdn)

        λL = real(measure_current_response_unequaltime(system, ρup, ρdn, prefix_up, suffix_up, prefix_dn, suffix_dn, qx=qxmin, qy=0.0))
        λT = real(measure_current_response_unequaltime(system, ρup, ρdn, prefix_up, suffix_up, prefix_dn, suffix_dn, qx=0.0, qy=qymin))
        kx = real(measure_KxPerSite(system, ρup, ρdn))
        ρs = 0.25 * (λL - λT)
        ρs_dia = 0.25 * (-kx - λT)
        data[sample, :] .= (λL, λT, kx, ρs, ρs_dia)
    end
    return (data=data, time_slices=system.L)
end

function write_metadata(outdir, params)
    metadata = Dict(
        "target" => "3x3 CE-QMC current-response benchmark",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"], "ntotal" => params["nup"] + params["ndn"]),
        "model" => Dict(
            "u" => params["u"],
            "dtau" => params["dtau"],
            "beta" => params["beta"],
            "use_charge_hs" => params["use_charge_hs"],
            "sys_type" => params["sys_type"],
            "is_lowrank" => params["is_lowrank"],
        ),
        "run" => Dict(
            "batch_nsamples" => params["batch_nsamples"],
            "stderr_target" => params["stderr_target"],
            "max_batches" => params["max_batches"],
        ),
    )
    open(joinpath(outdir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function write_summary(outdir, params, means, errs, nsamples, batches, time_slices)
    open(joinpath(outdir, "summary.tsv"), "w") do io
        println(io, "beta\tlambda_longitudinal_qmin0\tlambda_longitudinal_stderr\tlambda_transverse_0qmin\tlambda_transverse_stderr\tKx_per_site\tKx_stderr\trho_s_current\trho_s_current_stderr\trho_s_diamagnetic\trho_s_diamagnetic_stderr\ttime_slices\tnsamples\tbatches")
        println(io,
            string(
                params["beta"], '\t',
                means[1], '\t', errs[1], '\t',
                means[2], '\t', errs[2], '\t',
                means[3], '\t', errs[3], '\t',
                means[4], '\t', errs[4], '\t',
                means[5], '\t', errs[5], '\t',
                time_slices, '\t', nsamples, '\t', batches
            )
        )
    end
end

function main()
    params = parse_args(ARGS)
    root = normpath(joinpath(@__DIR__, "..", ".."))
    outdir = joinpath(root, params["output_dir"])
    mkpath(outdir)
    write_metadata(outdir, params)

    all = Matrix{Float64}(undef, 0, 5)
    time_slices = 0

    for batch in 1:params["max_batches"]
        Random.seed!(params["seed"] + batch - 1)
        system = build_system(params)
        qmc = build_qmc(system, params)
        result = run_batch(system, qmc)
        all = vcat(all, result.data)
        time_slices = result.time_slices

        means = vec(mean(all, dims=1))
        errs = vec(std(all, dims=1; corrected=true)) ./ sqrt(size(all, 1))
        nsamples = size(all, 1)
        write_summary(outdir, params, means, errs, nsamples, batch, time_slices)
        println(
            "batch=", batch,
            " lambdaL=", means[1],
            " lambdaT=", means[2],
            " rho_s=", means[4],
            " rho_s_dia=", means[5],
            " nsamples=", nsamples,
        )
        errs[4] <= params["stderr_target"] && break
    end
end

main()
