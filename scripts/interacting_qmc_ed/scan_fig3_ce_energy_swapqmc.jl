function parse_main_args(args)
    params = Dict(
        "lx" => 6,
        "ly" => 6,
        "nup" => 18,
        "ndn" => 18,
        "u" => 4.0,
        "dtau" => 0.1,
        "beta_list" => "0.5,1.2,2.4,3.6,5.0,6.0",
        "nwarmups" => 64,
        "batch_nsamples" => 128,
        "measure_interval" => 4,
        "stab_interval" => 1,
        "update_interval" => 1,
        "seed" => 1234,
        "output_dir" => joinpath("results", "interacting_qmc_ed", "fig3_ce_energy_swapqmc"),
        "stderr_target" => 0.003,
        "max_batches" => 8,
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
    sort!(beta_list)
    return params, beta_list
end

params, beta_list = parse_main_args(ARGS)

using DelimitedFiles
using Random
using Statistics
using TOML
using SwapQMC

struct DensityMatrix{T}
    ρ1::Matrix{T}
end

function DensityMatrix(G::AbstractMatrix)
    ρ1 = I - transpose(G)
    return DensityMatrix{eltype(ρ1)}(ρ1)
end

function update_density!(ρ::DensityMatrix, walker::HubbardWalker, spin::Int)
    ρ1 = ρ.ρ1
    G = walker.G[spin]
    transpose!(ρ1, G)
    @inbounds for i in eachindex(ρ1)
        ρ1[i] *= -1
    end
    @views ρ1[diagind(ρ1)] .+= 1
    return nothing
end

function measure_energy(system::Hubbard, ρup::DensityMatrix, ρdn::DensityMatrix)
    ρ1up = ρup.ρ1
    ρ1dn = ρdn.ρ1
    Tmat = system.T

    kinetic = zero(eltype(ρ1up))
    potential = zero(eltype(ρ1up))
    for i in eachindex(Tmat)
        if Tmat[i] != 0
            kinetic += Tmat[i] * (ρ1up[i] + ρ1dn[i])
        end
    end
    for i in 1:system.V
        potential += system.U * (ρ1up[i, i] * ρ1dn[i, i])
    end
    total = kinetic + potential
    return kinetic, potential, total
end

function build_system(params, beta)
    lx = params["lx"]
    ly = params["ly"]
    dtau = params["dtau"]
    tmat = hopping_matrix_Hubbard_2d(lx, ly, 1.0)
    time_slices = round(Int, beta / dtau)
    isapprox(beta / dtau, time_slices; atol=1e-10) || error("beta/dtau must be an integer")
    if isodd(time_slices)
        time_slices *= 2
    end

    return GenericHubbard(
        (lx, ly, 1),
        (params["nup"], params["ndn"]),
        tmat,
        params["u"],
        0.0,
        beta,
        time_slices,
        sys_type=ComplexF64,
        useChargeHST=true,
        useFirstOrderTrotter=false,
    )
end

function build_qmc(system, params)
    return QMC(
        system,
        params["nwarmups"],
        params["batch_nsamples"],
        params["measure_interval"],
        params["stab_interval"],
        params["update_interval"];
        forceSymmetry=true,
        saveRatio=false,
    )
end

function build_trial_wf(system)
    φ0up = trial_wf_free(system, 1, system.T)
    return [φ0up, copy(φ0up)]
end

function run_energy_batch(system, qmc, φ0)
    walker = HubbardWalker(system, qmc, φ0)
    ρup = DensityMatrix(walker.G[1])
    ρdn = DensityMatrix(walker.G[2])
    energies = zeros(Float64, qmc.nsamples, 3)

    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

    half_bins = max(1, div(qmc.measure_interval, 2))
    for sample in 1:qmc.nsamples
        sweep!(system, qmc, walker, loop_number=half_bins)
        update_density!(ρup, walker, 1)
        update_density!(ρdn, walker, 2)
        energies[sample, :] .= real.(measure_energy(system, ρup, ρdn))
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
        φ0 = build_trial_wf(system)
        batch = run_energy_batch(system, qmc, φ0)
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

function write_outputs(params, beta_list, results)
    root = normpath(joinpath(@__DIR__, ".."))
    outdir = joinpath(root, params["output_dir"])
    mkpath(outdir)

    metadata = Dict(
        "target" => "SwapQMC beta<=6 energy scan",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"], "ntotal" => params["nup"] + params["ndn"]),
        "model" => Dict("u" => params["u"], "dtau" => params["dtau"]),
        "beta_list" => beta_list,
        "run" => Dict(
            "batch_nsamples" => params["batch_nsamples"],
            "stderr_target" => params["stderr_target"],
            "max_batches" => params["max_batches"],
        ),
        "qmc" => Dict(
            "nwarmups" => params["nwarmups"],
            "measure_interval" => params["measure_interval"],
            "stab_interval" => params["stab_interval"],
            "update_interval" => params["update_interval"],
            "force_symmetry" => true,
            "use_charge_hst" => true,
            "odd_L_policy" => "double time slices when beta/dtau is odd",
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
end

results = map(job -> run_beta_point(params, job[2], job[1]), collect(enumerate(beta_list)))
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
