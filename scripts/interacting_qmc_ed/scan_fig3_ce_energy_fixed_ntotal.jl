using Distributed

function parse_main_args(args)
    params = Dict(
        "lx" => 6,
        "ly" => 6,
        "ntotal" => 36,
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
        "output_dir" => joinpath("results", "interacting_qmc_ed", "fig3_ce_energy_fixed_ntotal"),
        "stderr_target" => 0.003,
        "max_batches" => 8,
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

    beta_list = sort([parse(Float64, strip(x)) for x in split(params["beta_list"], ",") if !isempty(strip(x))])
    return params, beta_list
end

params, beta_list = parse_main_args(ARGS)
ntotal = params["ntotal"]
iseven(ntotal) || error("This driver currently assumes an even ntotal, got $ntotal")

unique_sectors = [(nup, ntotal - nup) for nup in 0:div(ntotal, 2)]
jobs = [(sector_index, beta_index, sector[1], sector[2], beta) for (sector_index, sector) in enumerate(unique_sectors) for (beta_index, beta) in enumerate(beta_list)]

worker_count = min(params["max_workers"], length(jobs))
if worker_count > 1
    try
        addprocs(worker_count)
    catch err
        @warn "Falling back to serial execution because worker startup failed" exception=(err, catch_backtrace())
        global worker_count = 1
    end
end

@everywhere begin
    using CanEnsAFQMC
    using DelimitedFiles
    using Random
    using Statistics
    using TOML

    function build_system(params, beta, nup, ndn)
        lx = params["lx"]
        ly = params["ly"]
        dtau = params["dtau"]
        tmat = hopping_matrix_Hubbard_2d(lx, ly, 1.0)
        time_slices = round(Int, beta / dtau)
        isapprox(beta / dtau, time_slices; atol=1e-10) || error("beta/dtau must be an integer, got beta=$beta dtau=$dtau")

        return GenericHubbard(
            (lx, ly, 1),
            (nup, ndn),
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

    function populate_sector_density!(system, walker, rho, spin)
        nspin = system.N[spin]
        if nspin == 0
            fill!(rho.ρ₁, 0.0)
        elseif nspin == system.V
            fill!(rho.ρ₁, 0.0)
            @inbounds for i in 1:system.V
                rho.ρ₁[i, i] = 1.0
            end
        else
            update!(system, walker, rho, spin)
        end
        return nothing
    end

    function run_energy_batch(system, qmc)
        walker = Walker(system, qmc)
        rho_up = DensityMatrix(system, Nft=qmc.num_FourierPoints)
        rho_dn = DensityMatrix(system, Nft=qmc.num_FourierPoints)
        energies = zeros(Float64, qmc.nsamples, 3)

        sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

        for sample in 1:qmc.nsamples
            sweep!(system, qmc, walker, loop_number=qmc.measure_interval)
            populate_sector_density!(system, walker, rho_up, 1)
            populate_sector_density!(system, walker, rho_dn, 2)
            energies[sample, :] .= real.(measure_Energy(system, rho_up, rho_dn))
        end

        return energies ./ sum(system.N)
    end

    function run_sector_beta_point(params, beta, nup, ndn, job_index)
        all_energies = Matrix{Float64}(undef, 0, 3)
        batch_summaries = Vector{NamedTuple}()

        if nup == 0 || ndn == 0
            kinetic = 0.0
            potential = 0.0
            total = 0.0
            push!(batch_summaries, (
                batch=0,
                seed=params["seed"] + job_index - 1,
                nsamples=0,
                kinetic=kinetic,
                kinetic_stderr=0.0,
                potential=potential,
                potential_stderr=0.0,
                total=total,
                total_stderr=0.0,
                time_slices=round(Int, beta / params["dtau"]),
            ))
        else
            for batch_idx in 1:params["max_batches"]
                batch_seed = params["seed"] + 1000 * job_index + batch_idx - 1
                Random.seed!(batch_seed)
                system = build_system(params, beta, nup, ndn)
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
        end

        final = batch_summaries[end]
        return (
            beta=beta,
            nup=nup,
            ndn=ndn,
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

    function log_binomial(n, k)
        (0 <= k <= n) || error("Invalid binomial arguments n=$n k=$k")
        k′ = min(k, n - k)
        k′ == 0 && return 0.0
        return sum(log(float(n - k′ + i)) - log(float(i)) for i in 1:k′)
    end

    function sector_multiplicity(nup, ndn)
        return nup == ndn ? 1 : 2
    end

    function sector_logz0(volume, nup, ndn)
        return log_binomial(volume, nup) + log_binomial(volume, ndn)
    end

    function sector_energy0_per_particle(volume, ntotal, u, nup, ndn)
        return u * nup * ndn / volume / ntotal
    end

    function integrate_sector_logz(beta_list, rows, params)
        ntotal = params["ntotal"]
        volume = params["lx"] * params["ly"]
        logz = Dict{Float64, Float64}()

        prev_beta = 0.0
        prev_total = ntotal * sector_energy0_per_particle(volume, ntotal, params["u"], rows[1].nup, rows[1].ndn)
        accumulated = 0.0

        for row in rows
            current_total = ntotal * row.total
            accumulated += 0.5 * (prev_total + current_total) * (row.beta - prev_beta)
            logz[row.beta] = sector_logz0(volume, row.nup, row.ndn) - accumulated
            prev_beta = row.beta
            prev_total = current_total
        end

        return logz
    end

    function combine_fixed_ntotal(beta_list, results, params)
        grouped = Dict{Tuple{Int, Int}, Vector{typeof(results[1])}}()
        for result in results
            push!(get!(grouped, (result.nup, result.ndn), typeof(results[1])[]), result)
        end

        sector_logz = Dict{Tuple{Int, Int}, Dict{Float64, Float64}}()
        for (sector, rows) in grouped
            sort!(rows, by=row -> row.beta)
            sector_logz[sector] = integrate_sector_logz(beta_list, rows, params)
        end

        combined = Vector{NamedTuple}()
        for beta in beta_list
            weights = Dict{Tuple{Int, Int}, Float64}()
            log_terms = Float64[]
            sectors = Tuple{Int, Int}[]

            for sector in keys(grouped)
                nup, ndn = sector
                log_weight = sector_logz[sector][beta] + log(float(sector_multiplicity(nup, ndn)))
                push!(log_terms, log_weight)
                push!(sectors, sector)
            end

            log_norm = maximum(log_terms)
            norm_terms = exp.(log_terms .- log_norm)
            norm = sum(norm_terms)

            total = 0.0
            approx_variance = 0.0
            kinetic = 0.0
            potential = 0.0

            for (idx, sector) in enumerate(sectors)
                weight = norm_terms[idx] / norm
                weights[sector] = weight
                row = only(filter(result -> result.nup == sector[1] && result.ndn == sector[2] && result.beta == beta, grouped[sector]))
                kinetic += weight * row.kinetic
                potential += weight * row.potential
                total += weight * row.total
                approx_variance += (weight * row.total_stderr)^2
            end

            dominant_sector = sectors[1]
            dominant_weight = weights[dominant_sector]
            for sector in sectors[2:end]
                if weights[sector] > dominant_weight
                    dominant_sector = sector
                    dominant_weight = weights[sector]
                end
            end

            push!(combined, (
                beta=beta,
                kinetic=kinetic,
                potential=potential,
                total=total,
                total_stderr=sqrt(approx_variance),
                dominant_sector=dominant_sector,
                dominant_weight=dominant_weight,
                sector_weights=weights,
            ))
        end

        return combined, sector_logz
    end
end

function write_outputs(params, beta_list, results)
    root = normpath(joinpath(@__DIR__, ".."))
    outdir = joinpath(root, params["output_dir"])
    mkpath(outdir)

    combined, sector_logz = combine_fixed_ntotal(beta_list, results, params)

    metadata = Dict(
        "target" => "Fixed-total-N CE energy assembled from spin-resolved sectors",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("ntotal" => params["ntotal"]),
        "model" => Dict("u" => params["u"], "dtau" => params["dtau"]),
        "beta_list" => beta_list,
        "run" => Dict(
            "batch_nsamples" => params["batch_nsamples"],
            "stderr_target" => params["stderr_target"],
            "max_batches" => params["max_batches"],
            "max_workers" => worker_count,
            "unique_spin_sectors" => ["$(sector[1])+$(sector[2])" for sector in unique_sectors],
        ),
        "notes" => Dict(
            "weight_reconstruction" => "log Z_sector(beta) = log Z_sector(0) - integral_0^beta E_sector(beta') dbeta' using trapezoidal integration",
            "z0_reference" => "log Z_sector(0) from binomial state counts C(V,nup)C(V,ndn)",
            "beta0_energy" => "E_sector(0) = U * nup * ndn / V",
            "stderr_caveat" => "fixed-N stderr only propagates sector energy stderrs at fixed reconstructed weights",
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

    sector_table = Matrix{Any}(undef, length(results) + 1, 12)
    sector_table[1, :] = [
        "beta",
        "nup",
        "ndn",
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

    for (idx, result) in enumerate(sort(results, by=result -> (result.beta, result.nup)))
        sector_table[idx + 1, :] = [
            result.beta,
            result.nup,
            result.ndn,
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

    writedlm(joinpath(outdir, "sector_energy.tsv"), sector_table, '\t')

    fixed_table = Matrix{Any}(undef, length(combined) + 1, 7)
    fixed_table[1, :] = [
        "beta",
        "kinetic_per_particle",
        "potential_per_particle",
        "total_per_particle",
        "approx_total_stderr",
        "dominant_sector",
        "dominant_weight",
    ]

    for (idx, result) in enumerate(combined)
        fixed_table[idx + 1, :] = [
            result.beta,
            result.kinetic,
            result.potential,
            result.total,
            result.total_stderr,
            "$(result.dominant_sector[1])+$(result.dominant_sector[2])",
            result.dominant_weight,
        ]
    end

    writedlm(joinpath(outdir, "fixed_ntotal_energy.tsv"), fixed_table, '\t')

    open(joinpath(outdir, "sector_weights.tsv"), "w") do io
        writedlm(io, [["beta", "nup", "ndn", "multiplicity", "logz_sector", "normalized_weight"]], '\t')
        for row in combined
            for sector in sort(collect(keys(row.sector_weights)))
                nup, ndn = sector
                writedlm(io, [[
                    row.beta,
                    nup,
                    ndn,
                    sector_multiplicity(nup, ndn),
                    sector_logz[sector][row.beta],
                    row.sector_weights[sector],
                ]], '\t')
            end
        end
    end

    open(joinpath(outdir, "batch_progress.tsv"), "w") do io
        writedlm(io, [["beta", "nup", "ndn", "batch", "seed", "nsamples", "kinetic", "kinetic_stderr", "potential", "potential_stderr", "total", "total_stderr", "time_slices"]], '\t')
        for result in sort(results, by=result -> (result.beta, result.nup))
            for batch in result.batch_summaries
                writedlm(io, [[
                    result.beta,
                    result.nup,
                    result.ndn,
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

runner(job) = run_sector_beta_point(params, job[5], job[3], job[4], (job[1] - 1) * length(beta_list) + job[2])
results = worker_count > 1 ? pmap(runner, jobs) : map(runner, jobs)
sort!(results, by=result -> (result.beta, result.nup))

for result in results
    println(
        "beta=", result.beta,
        " sector=", result.nup, "+", result.ndn,
        " total/N=", result.total,
        " stderr=", result.total_stderr,
        " nsamples=", result.nsamples,
        " batches=", result.batches,
    )
end

write_outputs(params, beta_list, results)
