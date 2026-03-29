using Distributed

function parse_main_args(args)
    params = Dict(
        "lx" => 6,
        "ly" => 6,
        "nup" => 18,
        "ndn" => 18,
        "u" => 4.0,
        "dtau" => 0.1,
        "beta_list" => "0.5,1.2,2.4,3.6,5.0,6.0",
        "nwarmups" => 128,
        "batch_nsamples" => 256,
        "measure_interval" => 4,
        "stab_interval" => 10,
        "cluster_size" => 4,
        "num_fourier_points" => 15,
        "lr_threshold" => 1.0e-10,
        "seed" => 1234,
        "output_dir" => joinpath("results", "fig3_ce_energy_scan"),
        "stderr_target" => 0.003,
        "max_batches" => 16,
        "max_workers" => 4,
    )

    for arg in args
        startswith(arg, "--") || error("Expected --key=value arguments, got: $arg")
        key, value = split(arg[3:end], "=", limit=2)
        haskey(params, key) || error("Unknown argument: --$key")
        default = params[key]
        if default isa Int
            params[key] = parse(Int, value)
        elseif default isa Float64
            params[key] = parse(Float64, value)
        else
            params[key] = value
        end
    end

    beta_list = [parse(Float64, strip(x)) for x in split(params["beta_list"], ",") if !isempty(strip(x))]
    return params, beta_list
end

params, beta_list = parse_main_args(ARGS)
worker_count = min(params["max_workers"], length(beta_list))
if worker_count > 1
    addprocs(worker_count)
end

@everywhere begin
    using CanEnsAFQMC
    using DelimitedFiles
    using Random
    using Statistics
    using TOML

    function build_system(params, beta)
        lx = params["lx"]
        ly = params["ly"]
        dtau = params["dtau"]
        tmat = hopping_matrix_Hubbard_2d(lx, ly, 1.0)
        time_slices = round(Int, beta / dtau)
        isapprox(beta / dtau, time_slices; atol=1e-10) || error("beta/dtau must be an integer, got beta=$beta dtau=$dtau")

        return GenericHubbard(
            (lx, ly, 1),
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
end

function write_outputs(params, beta_list, results)
    root = normpath(joinpath(@__DIR__, ".."))
    outdir = joinpath(root, params["output_dir"])
    mkpath(outdir)

    metadata = Dict(
        "target" => "Fig. 3 CE energy data for beta <= 6",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"], "ntotal" => params["nup"] + params["ndn"]),
        "model" => Dict("u" => params["u"], "dtau" => params["dtau"]),
        "beta_list" => beta_list,
        "run" => Dict(
            "batch_nsamples" => params["batch_nsamples"],
            "stderr_target" => params["stderr_target"],
            "max_batches" => params["max_batches"],
            "max_workers" => worker_count,
        ),
        "qmc" => Dict(
            "nwarmups" => params["nwarmups"],
            "measure_interval" => params["measure_interval"],
            "stab_interval" => params["stab_interval"],
            "cluster_size" => params["cluster_size"],
            "num_fourier_points" => params["num_fourier_points"],
            "lowrank_threshold" => params["lr_threshold"],
        ),
    )

    open(joinpath(outdir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end

    table = Matrix{Any}(undef, length(results) + 1, 10)
    table[1, :] = [
        "beta",
        "kinetic_per_particle",
        "kinetic_stderr",
        "potential_per_particle",
        "potential_stderr",
        "total_per_particle",
        "total_stderr",
        "time_slices",
        "nsamples",
        "batches",
    ]

    for (idx, result) in enumerate(results)
        table[idx + 1, :] = [
            result.beta,
            result.kinetic,
            result.kinetic_stderr,
            result.potential,
            result.potential_stderr,
            result.total,
            result.total_stderr,
            result.time_slices,
            result.nsamples,
            result.batches,
        ]
    end

    writedlm(joinpath(outdir, "ce_energy.tsv"), table, '\t')

    open(joinpath(outdir, "batch_progress.tsv"), "w") do io
        writedlm(io, [["beta", "batch", "seed", "nsamples", "kinetic", "kinetic_stderr", "potential", "potential_stderr", "total", "total_stderr", "time_slices"]], '\t')
        for result in results
            for batch in result.batch_summaries
                writedlm(io, [[
                    result.beta,
                    batch.batch,
                    batch.seed,
                    batch.nsamples,
                    batch.kinetic,
                    batch.kinetic_stderr,
                    batch.potential,
                    batch.potential_stderr,
                    batch.total,
                    batch.total_stderr,
                    batch.time_slices,
                ]], '\t')
            end
        end
    end
end

jobs = collect(enumerate(beta_list))
results = pmap(job -> run_beta_point(params, job[2], job[1]), jobs)
sort!(results, by=result -> result.beta)

for result in results
    println(
        "beta=", result.beta,
        " total/N=", result.total,
        " stderr=", result.total_stderr,
        " nsamples=", result.nsamples,
        " batches=", result.batches,
    )
end

write_outputs(params, beta_list, results)
println("Wrote adaptive CE energy scan to: ", joinpath(normpath(joinpath(@__DIR__, "..")), params["output_dir"]))
