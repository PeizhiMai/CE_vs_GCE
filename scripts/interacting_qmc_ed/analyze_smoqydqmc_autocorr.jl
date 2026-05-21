#!/usr/bin/env julia
# Estimate integrated autocorrelation times from SmoQyDQMC bin time series.
#
# For an autocorrelation probe, run the checkpoint driver with
#   MEASUREMENT_PROFILE=autocorr-current
# and N_BINS=N_MEASUREMENTS so each bin is one measured configuration.  This
# script then reports tau_int in both bin units and update-sweep units.

using HDF5
using Printf
using Statistics
using TOML

function usage()
    println("""
    Usage:
      julia --project=julia_env scripts/interacting_qmc_ed/analyze_smoqydqmc_autocorr.jl RUN_DIR [options]

    Options:
      --observable OBS       density | double_occ | sgn | rho_s_current | lambda_l | lambda_t | minus_kx
                             default: rho_s_current
      --max-lag N            maximum autocorrelation lag; default min(1000, n/2)
      --window-c C           self-consistent window parameter; default 5.0
      --sweep-spacing X      sweeps between adjacent stored bins; default from simulation_info
      --out PATH             output TSV; default RUN_DIR/autocorr_OBS.tsv

    Notes:
      tau_int_sweeps = tau_int_bins * sweep_spacing.
      If N_BINS=N_MEASUREMENTS, sweep_spacing=N_updates.
      For production measurement thinning, a conservative first choice is
      N_updates ≈ ceil(2 * max_tau_int_sweeps) for the slow observable.
    """)
end

function parse_cli(args)
    opts = Dict{String,String}(
        "observable" => "rho_s_current",
        "window-c" => "5.0",
    )
    positional = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--help" || a == "-h"
            usage()
            exit(0)
        elseif startswith(a, "--")
            key = replace(a[3:end], "_" => "-")
            i == length(args) && error("Missing value for option $a")
            opts[key] = args[i+1]
            i += 2
        else
            push!(positional, a)
            i += 1
        end
    end
    isempty(positional) && (usage(); error("RUN_DIR is required"))
    return abspath(positional[1]), opts
end

function read_metadata(run_dir::String)
    infos = sort(collect(filter(p -> occursin(r"simulation_info_sID-.*_pID-0\.toml$", basename(p)),
                                readdir(run_dir; join=true))))
    isempty(infos) && return Dict{String,Any}()
    return TOML.parsefile(infos[1])
end

function default_sweep_spacing(run_dir::String)
    info = read_metadata(run_dir)
    if haskey(info, "metadata")
        md = info["metadata"]
        if all(k -> haskey(md, k), ["N_updates", "N_measurements", "N_bins"])
            bin_size = Float64(md["N_measurements"]) / Float64(md["N_bins"])
            return Float64(md["N_updates"]) * bin_size
        end
    end
    return 1.0
end

function read_observable(path::String, observable::String)
    h5open(path, "r") do h5
        if observable in ("density", "double_occ", "sgn", "compressibility")
            return vec(real.(read(h5["GLOBAL/$observable"])))
        elseif observable == "minus_kx"
            hop = read(h5["LOCAL/hopping_energy"])
            return vec(-real.(hop[:, 1]))
        elseif observable in ("rho_s_current", "lambda_l", "lambda_t")
            current = read(h5["CORRELATIONS/STANDARD/INTEGRATED/current/MOMENTUM"])
            ndims(current) < 4 && error("Unexpected current MOMENTUM array shape in $path: $(size(current))")
            # Stored dimensions in bins files are [bin, K_2, K_1, HOPPING_PAIR].
            lambda_l = vec(real.(current[:, 1, 2, 1])) # K_1=1, K_2=0
            lambda_t = vec(real.(current[:, 2, 1, 1])) # K_1=0, K_2=1
            observable == "lambda_l" && return lambda_l
            observable == "lambda_t" && return lambda_t
            return 0.25 .* (lambda_l .- lambda_t)
        else
            error("Unknown observable $observable")
        end
    end
end

function autocorrelation(x::AbstractVector{<:Real}, max_lag::Int)
    n = length(x)
    n >= 3 || error("Need at least 3 samples for autocorrelation")
    y = Float64.(x) .- mean(x)
    v = sum(abs2, y) / n
    if v == 0
        return ones(1)
    end
    max_lag = min(max_lag, n - 1)
    acf = Vector{Float64}(undef, max_lag + 1)
    for lag in 0:max_lag
        acf[lag+1] = sum(@view(y[1:n-lag]) .* @view(y[1+lag:n])) / (n - lag) / v
    end
    return acf
end

function integrated_tau(acf::Vector{Float64}, c::Float64)
    tau = 0.5
    window = 0
    for lag in 1:(length(acf)-1)
        ρ = acf[lag+1]
        if ρ <= 0
            break
        end
        tau += ρ
        window = lag
        if lag > c * tau
            break
        end
    end
    return max(tau, 0.5), window
end

function main()
    run_dir, opts = parse_cli(ARGS)
    observable = opts["observable"]
    sweep_spacing = haskey(opts, "sweep-spacing") ? parse(Float64, opts["sweep-spacing"]) : default_sweep_spacing(run_dir)
    c = parse(Float64, opts["window-c"])
    out = get(opts, "out", joinpath(run_dir, "autocorr_$(observable).tsv"))

    bin_files = sort(collect(readdir(joinpath(run_dir, "bins"); join=true)))
    filter!(p -> occursin(r"bins_pID-\d+\.h5$", basename(p)), bin_files)
    isempty(bin_files) && error("No bins_pID-*.h5 files found under $(joinpath(run_dir, "bins"))")

    rows = []
    for bf in bin_files
        x = read_observable(bf, observable)
        n = length(x)
        max_lag = haskey(opts, "max-lag") ? parse(Int, opts["max-lag"]) : min(1000, max(1, n ÷ 2))
        acf = autocorrelation(x, max_lag)
        tau_bins, window = integrated_tau(acf, c)
        m = match(r"bins_pID-(\d+)\.h5$", basename(bf))
        pID = isnothing(m) ? -1 : parse(Int, m.captures[1])
        push!(rows, (
            pID = pID,
            n = n,
            mean = mean(x),
            std = std(x),
            tau_bins = tau_bins,
            tau_sweeps = tau_bins * sweep_spacing,
            window = window,
            acf1 = length(acf) >= 2 ? acf[2] : NaN,
            file = bf,
        ))
    end

    open(out, "w") do io
        println(io, "pID\tn\tmean\tstd\ttau_int_bins\ttau_int_sweeps\twindow_bins\tacf_lag1\tsweep_spacing\tfile")
        for r in rows
            @printf(io, "%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%d\t%.12g\t%.12g\t%s\n",
                    r.pID, r.n, r.mean, r.std, r.tau_bins, r.tau_sweeps, r.window, r.acf1, sweep_spacing, r.file)
        end
    end

    taus = [r.tau_sweeps for r in rows]
    @printf("observable          = %s\n", observable)
    @printf("chains              = %d\n", length(rows))
    @printf("sweep_spacing       = %.6g update sweeps per stored bin\n", sweep_spacing)
    @printf("tau_int_sweeps mean = %.6g\n", mean(taus))
    @printf("tau_int_sweeps max  = %.6g\n", maximum(taus))
    @printf("recommended N_updates first pass ≈ %d\n", max(1, ceil(Int, 2 * maximum(taus))))
    println("wrote $out")
end

main()
