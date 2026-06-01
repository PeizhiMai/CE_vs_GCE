#!/usr/bin/env julia

using MPI
using TOML
using Printf

include(joinpath(@__DIR__, "benchmark_ce_green_matsubara_3x3.jl"))

function parse_mpi_args(args)
    params = Dict(
        "output_dir" => joinpath("results", "interacting_qmc_ed", "ce_green_tau_space_mpi_3x3"),
        "seed" => 1234,
        "combine" => true,
        "python" => get(ENV, "PYTHON", "python3"),
    )
    passthrough = String[]
    for arg in args
        if startswith(arg, "--output-dir=")
            params["output_dir"] = split(arg, "=", limit=2)[2]
            push!(passthrough, arg)
        elseif startswith(arg, "--seed=")
            params["seed"] = parse(Int, split(arg, "=", limit=2)[2])
            push!(passthrough, arg)
        elseif startswith(arg, "--combine=")
            params["combine"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--python=")
            params["python"] = split(arg, "=", limit=2)[2]
        else
            push!(passthrough, arg)
        end
    end
    return params, passthrough
end

function replace_or_push_arg!(args::Vector{String}, name::String, value)
    prefix = string("--", name, "=")
    newarg = string(prefix, value)
    idx = findfirst(arg -> startswith(arg, prefix), args)
    if idx === nothing
        push!(args, newarg)
    else
        args[idx] = newarg
    end
    return args
end

function write_mpi_metadata(root_outdir::String, mpi_params, nranks::Int)
    mkpath(root_outdir)
    metadata = Dict(
        "target" => "MPI independent-chain wrapper for 3x3 CE unequal-time G(r,tau)/G(k,tau)",
        "mpi" => Dict(
            "nranks" => nranks,
            "rank_model" => "one MPI rank = one independent Markov chain",
            "combined_by" => "combine_ce_green_tau_space_rank_outputs.py",
        ),
        "base_seed" => mpi_params["seed"],
    )
    open(joinpath(root_outdir, "mpi_metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function main_mpi(args=ARGS)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    mpi_params, passthrough = parse_mpi_args(args)

    root = normpath(joinpath(@__DIR__, "..", ".."))
    root_outdir = joinpath(root, mpi_params["output_dir"])
    rank_rel_outdir = joinpath(mpi_params["output_dir"], "ranks", @sprintf("rank_%05d", rank))
    rank_seed = mpi_params["seed"] + 1_000_003 * rank

    rank_args = copy(passthrough)
    replace_or_push_arg!(rank_args, "output-dir", rank_rel_outdir)
    replace_or_push_arg!(rank_args, "seed", rank_seed)
    replace_or_push_arg!(rank_args, "checkpoint-world-size", nranks)
    replace_or_push_arg!(rank_args, "checkpoint-root-dir", root_outdir)

    if rank == 0
        mkpath(root_outdir)
        mkpath(joinpath(root_outdir, "ranks"))
        write_mpi_metadata(root_outdir, mpi_params, nranks)
        println("MPI CE-QMC start: nranks=", nranks, " root_outdir=", root_outdir)
    end
    MPI.Barrier(comm)

    println("rank=", rank, " seed=", rank_seed, " output_dir=", rank_rel_outdir)
    flush(stdout)
    main(rank_args)
    MPI.Barrier(comm)

    if rank == 0 && mpi_params["combine"]
        combiner = joinpath(@__DIR__, "combine_ce_green_tau_space_rank_outputs.py")
        cmd = Cmd([
            string(mpi_params["python"]),
            combiner,
            "--root-dir", root_outdir,
            "--rank-glob", "ranks/rank_*",
            "--outdir", root_outdir,
        ])
        println("Combining rank outputs: ", cmd)
        run(cmd)
        println("MPI CE-QMC combined outputs written to ", root_outdir)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_mpi()
end
