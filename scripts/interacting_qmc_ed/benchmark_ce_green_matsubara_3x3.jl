#!/usr/bin/env julia

# The CE Markov acceptance step is dominated by dense non-Hermitian eigensolves
# used by the fixed-N partition-function recursion.  On CADES, MKL's LAPACK is
# substantially faster than the default OpenBLAS LAPACK for these V≈O(10^2)
# matrices.  Load it before LinearAlgebra/CanEnsAFQMC when available.  Set
# CE_USE_MKL=false to force the default BLAS/LAPACK backend.
if lowercase(get(ENV, "CE_USE_MKL", "true")) != "false" && Base.find_package("MKL") !== nothing
    try
        @eval using MKL
    catch err
        @warn "CE_USE_MKL requested, but MKL could not be loaded; falling back to default BLAS/LAPACK" exception=(err, catch_backtrace())
    end
end

using LinearAlgebra
using Random
using Serialization
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
        "dtau" => 0.1,
        "beta" => 10.0,
        "nwarmups" => 500,
        "batch_nsamples" => 100,
        "measure_interval" => 2,
        "stab_interval" => 10,
        "cluster_size" => 3,
        "num_fourier_points" => 10,
        "nfreq" => 10,
        "use_lowrank" => false,
        "lr_thld" => 1e-10,
        "force_symmetry" => true,
        "use_cluster_update" => true,
        "measure_greens" => true,
        "measure_bkt" => true,
        "measure_equal_time" => true,
        "measure_equal_time_correlations" => false,
        "phase_reweight" => false,
        "bkt_current_estimator" => "projected",
        "bkt_refresh_interval" => 10,
        "bkt_adaptive_refresh" => false,
        "bkt_refresh_tol" => 1e-6,
        "bkt_refresh_min" => 1,
        "bkt_refresh_max" => 20,
        "bkt_refresh_growth_patience" => 3,
        "bkt_current_drift_tol" => 1e-2,
        "allow_imbalanced_canonical_recursion" => false,
        "seed" => 1234,
        "max_batches" => 5,
        "use_charge_hs" => false,
        "sys_type" => "complex",
        "output_dir" => joinpath("results", "interacting_qmc_ed", "ce_green_matsubara_3x3"),
        "checkpoint_enable" => false,
        "checkpoint_file" => "checkpoint.jls",
        "checkpoint_every_batches" => 1,
        "checkpoint_freq_hours" => 0.0,
        "runtime_limit_hours" => 0.0,
        "checkpoint_exit_code" => 13,
        "checkpoint_keep" => false,
        "checkpoint_reset_accumulators" => false,
        "checkpoint_world_size" => 1,
        "checkpoint_root_dir" => "",
        "checkpoint_sync_timeout_seconds" => 300.0,
        "checkpoint_sync_poll_seconds" => 5.0,
        "checkpoint_warmup_chunk" => 10,
        "diagnose_metropolis" => false,
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
        elseif startswith(arg, "--batch-nsamples=")
            params["batch_nsamples"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--max-batches=")
            params["max_batches"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--nwarmups=")
            params["nwarmups"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--measure-interval=")
            params["measure_interval"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--stab-interval=")
            params["stab_interval"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--cluster-size=")
            params["cluster_size"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dtau=")
            params["dtau"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--u=")
            params["u"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--beta=")
            params["beta"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            params["seed"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num-fourier-points=")
            value = split(arg, "=", limit=2)[2]
            params["num_fourier_points"] = lowercase(value) == "auto" ? 0 : parse(Int, value)
        elseif startswith(arg, "--nfreq=")
            params["nfreq"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use-lowrank=")
            params["use_lowrank"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--lr-thld=")
            params["lr_thld"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--force-symmetry=")
            params["force_symmetry"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use-cluster-update=")
            params["use_cluster_update"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--measure-greens=")
            params["measure_greens"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--measure-bkt=")
            params["measure_bkt"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--measure-equal-time=")
            params["measure_equal_time"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--measure-equal-time-correlations=")
            params["measure_equal_time_correlations"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--phase-reweight=")
            params["phase_reweight"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--bkt-current-estimator=")
            params["bkt_current_estimator"] = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--allow-imbalanced-canonical-recursion=")
            params["allow_imbalanced_canonical_recursion"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--bkt-refresh-interval=")
            params["bkt_refresh_interval"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--bkt-adaptive-refresh=")
            params["bkt_adaptive_refresh"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--bkt-refresh-tol=")
            params["bkt_refresh_tol"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--bkt-refresh-min=")
            params["bkt_refresh_min"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--bkt-refresh-max=")
            params["bkt_refresh_max"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--bkt-refresh-growth-patience=")
            params["bkt_refresh_growth_patience"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--bkt-current-drift-tol=")
            params["bkt_current_drift_tol"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use-charge-hs=")
            params["use_charge_hs"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sys-type=")
            params["sys_type"] = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output-dir=")
            params["output_dir"] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--checkpoint-enable=")
            params["checkpoint_enable"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-file=")
            params["checkpoint_file"] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--checkpoint-every-batches=")
            params["checkpoint_every_batches"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-freq-hours=")
            params["checkpoint_freq_hours"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--runtime-limit-hours=")
            params["runtime_limit_hours"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-exit-code=")
            params["checkpoint_exit_code"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-keep=")
            params["checkpoint_keep"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-reset-accumulators=")
            params["checkpoint_reset_accumulators"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-world-size=")
            params["checkpoint_world_size"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-root-dir=")
            params["checkpoint_root_dir"] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--checkpoint-sync-timeout-seconds=")
            params["checkpoint_sync_timeout_seconds"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-sync-poll-seconds=")
            params["checkpoint_sync_poll_seconds"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--checkpoint-warmup-chunk=")
            params["checkpoint_warmup_chunk"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--diagnose-metropolis=")
            params["diagnose_metropolis"] = parse(Bool, split(arg, "=", limit=2)[2])
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
    if params["u"] > 0 && !params["use_charge_hs"] && params["force_symmetry"]
        error(
            "positive-U spin-HS canonical sampling requires --force-symmetry=false; " *
            "the symmetric sampler replaces the down-spin determinant weight by the up-spin weight",
        )
    end
    num_fourier_points = params["num_fourier_points"]
    if num_fourier_points <= 0
        # Discrete canonical projection needs enough roots of unity to avoid
        # aliasing coefficient c_N of a degree-V determinant polynomial:
        # Nft > max(N, V-N) for each spin sector.  This can be much smaller
        # than V+1 away from half filling; e.g. V=144,N=36 needs 109, not 145.
        num_fourier_points = maximum(max(N, system.V - N) for N in system.N) + 1
        params["num_fourier_points"] = num_fourier_points
    end
    return QMC(
        system,
        nwarmups=params["nwarmups"],
        nsamples=params["batch_nsamples"],
        measure_interval=params["measure_interval"],
        stab_interval=params["stab_interval"],
        useClusterUpdate=params["use_cluster_update"],
        cluster_size=params["cluster_size"],
        num_FourierPoints=num_fourier_points,
        forceSymmetry=params["force_symmetry"],
        isLowrank=params["use_lowrank"],
        lrThld=params["lr_thld"],
        saveRatio=params["diagnose_metropolis"],
    )
end

function write_metadata(outdir, params, L)
    metadata = Dict(
        "target" => "3x3 CE equal-time + BKT/current-response benchmark with optional unequal-time single-particle G(r,tau)/G(k,tau)",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"]),
        "model" => Dict(
            "u" => params["u"],
            "dtau" => params["dtau"],
            "beta" => params["beta"],
            "time_slices" => L,
            "use_charge_hs" => params["use_charge_hs"],
            "sys_type" => params["sys_type"],
            "use_lowrank" => params["use_lowrank"],
            "lr_thld" => params["lr_thld"],
            "force_symmetry" => params["force_symmetry"],
            "use_cluster_update" => params["use_cluster_update"],
            "cluster_size" => params["cluster_size"],
            "stab_interval" => params["stab_interval"],
        ),
        "unequal_time" => Dict(
            "enabled" => params["measure_greens"],
            "Ctau_branch" => "addition",
            "addition_backend" => "canonical_eigenbasis_coefficient_recursion",
            "also_write_removal_branch" => true,
            "realspace_definition" => "G_r_tau = V^-1 sum_j G[j+r,j;tau]",
            "momentum_definition" => "G_k_tau = sum_r exp(-i k dot r) G_r_tau",
            "physical_spin_hs_sector_normalization" => true,
        ),
        "bkt" => Dict(
            "enabled" => params["measure_bkt"],
            "output_file" => "bkt_observables_qmc.tsv",
            "lambda_longitudinal_qmin0" => "integral_0_beta d_tau <Jx(qmin,0,tau) Jx(-qmin,0,0)> / V",
            "lambda_transverse_0qmin" => "integral_0_beta d_tau <Jx(0,qmin,tau) Jx(0,-qmin,0)> / V",
            "rho_s_current" => "0.25 * (lambda_longitudinal_qmin0 - lambda_transverse_0qmin)",
            "rho_s_diamagnetic" => "0.25 * (-Kx_per_site - lambda_transverse_0qmin)",
            "bkt_universal_jump_2T_over_pi" => "2 / (pi * beta)",
            "bkt_residual" => "rho_s - 2/(pi*beta)",
            "current_estimator" => params["bkt_current_estimator"],
            "adaptive_refresh" => params["bkt_adaptive_refresh"],
            "refresh_interval" => params["bkt_refresh_interval"],
            "refresh_tol" => params["bkt_refresh_tol"],
            "refresh_min" => params["bkt_refresh_min"],
            "refresh_max" => params["bkt_refresh_max"],
            "refresh_growth_patience" => params["bkt_refresh_growth_patience"],
            "current_drift_tol" => params["bkt_current_drift_tol"],
            "refresh_tol_smoqydqmc_analogue" => "SmoQyDQMC δG_max threshold; CE compares propagated and stable LDR displaced Green functions at candidate refresh endpoints.",
        ),
        "equal_time" => Dict(
            "enabled" => params["measure_equal_time"],
            "output_file" => "equal_time_observables_qmc.tsv",
            "observables" => "kinetic_per_site, interaction_per_site, total_per_site, double_occupancy_per_site, local_moment_z, Kx_per_site",
            "correlations_enabled" => params["measure_equal_time_correlations"],
            "correlation_files" => "equal_time_charge_spin_wedge_qmc.tsv, equal_time_structure_factors_qmc.tsv, equal_time_neighbor_shells_qmc.tsv",
            "correlation_notes" => "charge and spin equal-time correlations only; no pairing measurements; charge structure factor is also written in connected form C_nn(r)-density^2; spin_s_s uses s=n_up-n_dn and spin_SzSz=spin_s_s/4",
            "phase_reweight" => params["phase_reweight"],
            "phase_definition" => "phase = sign(det_up) * sign(det_down); observables are sum(phase*O)/sum(phase)",
        ),
        "checkpoint" => Dict(
            "enabled" => params["checkpoint_enable"],
            "file" => params["checkpoint_file"],
            "every_batches" => params["checkpoint_every_batches"],
            "freq_hours" => params["checkpoint_freq_hours"],
            "runtime_limit_hours" => params["runtime_limit_hours"],
            "exit_code" => params["checkpoint_exit_code"],
            "keep_after_success" => params["checkpoint_keep"],
            "reset_accumulators_on_resume" => params["checkpoint_reset_accumulators"],
            "world_size" => params["checkpoint_world_size"],
            "root_dir" => params["checkpoint_root_dir"],
            "sync_timeout_seconds" => params["checkpoint_sync_timeout_seconds"],
            "sync_poll_seconds" => params["checkpoint_sync_poll_seconds"],
            "warmup_chunk" => params["checkpoint_warmup_chunk"],
            "format" => "Julia Serialization .jls with walker, RNG, completed-batch count, and measurement accumulator sums",
        ),
        "diagnostics" => Dict(
            "diagnose_metropolis" => params["diagnose_metropolis"],
            "metropolis_ratio_file" => params["diagnose_metropolis"] ? "metropolis_ratio_diagnostics.tsv" : "",
        ),
    )
    open(joinpath(outdir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function write_bkt_summary(outdir, params, means, errs, nsamples, batches, time_slices)
    temperature = 1.0 / params["beta"]
    jump = 2.0 * temperature / π
    open(joinpath(outdir, "bkt_observables_qmc.tsv"), "w") do io
        println(io, join([
            "beta",
            "temperature",
            "bkt_universal_jump_2T_over_pi",
            "lambda_longitudinal_qmin0",
            "lambda_longitudinal_stderr",
            "lambda_transverse_0qmin",
            "lambda_transverse_stderr",
            "Kx_per_site",
            "Kx_stderr",
            "diamagnetic_minus_Kx_per_site",
            "diamagnetic_minus_Kx_stderr",
            "rho_s_current",
            "rho_s_current_stderr",
            "rho_s_diamagnetic",
            "rho_s_diamagnetic_stderr",
            "bkt_residual_current",
            "bkt_residual_current_stderr",
            "bkt_residual_diamagnetic",
            "bkt_residual_diamagnetic_stderr",
            "time_slices",
            "nsamples",
            "batches",
        ], "\t"))
        println(io, string(
            params["beta"], '\t',
            temperature, '\t',
            jump, '\t',
            means[1], '\t', errs[1], '\t',
            means[2], '\t', errs[2], '\t',
            means[3], '\t', errs[3], '\t',
            -means[3], '\t', errs[3], '\t',
            means[4], '\t', errs[4], '\t',
            means[5], '\t', errs[5], '\t',
            means[4] - jump, '\t', errs[4], '\t',
            means[5] - jump, '\t', errs[5], '\t',
            time_slices, '\t',
            nsamples, '\t',
            batches
        ))
    end
end

function write_equal_time_summary(
    outdir, params, means, errs, nsamples, batches, time_slices;
    phase_sum::Float64=Float64(nsamples),
    average_phase::Float64=1.0,
    phase_reweighted::Bool=false,
    signed_sums=nothing,
)
    ntotal = params["nup"] + params["ndn"]
    nsites = params["lx"] * params["ly"]
    density = ntotal / nsites
    primitive_signed = signed_sums === nothing ? means .* phase_sum : signed_sums
    local_moment_signed = density * phase_sum - 2.0 * primitive_signed[4]
    diamagnetic_signed = -primitive_signed[5]
    open(joinpath(outdir, "equal_time_observables_qmc.tsv"), "w") do io
        println(io, join([
            "beta",
            "temperature",
            "nup",
            "ndn",
            "ntotal",
            "density",
            "kinetic_per_site",
            "kinetic_stderr",
            "interaction_per_site",
            "interaction_stderr",
            "total_per_site",
            "total_stderr",
            "double_occupancy_per_site",
            "double_occupancy_stderr",
            "local_moment_z",
            "local_moment_z_stderr",
            "Kx_per_site",
            "Kx_stderr",
            "diamagnetic_minus_Kx_per_site",
            "diamagnetic_minus_Kx_stderr",
            "time_slices",
            "nsamples",
            "batches",
            "phase_reweighted",
            "phase_sum",
            "average_phase",
            "kinetic_per_site_signed_sum",
            "interaction_per_site_signed_sum",
            "total_per_site_signed_sum",
            "double_occupancy_per_site_signed_sum",
            "local_moment_z_signed_sum",
            "Kx_per_site_signed_sum",
            "diamagnetic_minus_Kx_per_site_signed_sum",
        ], "\t"))
        println(io, string(
            params["beta"], '\t',
            1.0 / params["beta"], '\t',
            params["nup"], '\t',
            params["ndn"], '\t',
            ntotal, '\t',
            density, '\t',
            means[1], '\t', errs[1], '\t',
            means[2], '\t', errs[2], '\t',
            means[3], '\t', errs[3], '\t',
            means[4], '\t', errs[4], '\t',
            density - 2.0 * means[4], '\t', 2.0 * errs[4], '\t',
            means[5], '\t', errs[5], '\t',
            -means[5], '\t', errs[5], '\t',
            time_slices, '\t',
            nsamples, '\t',
            batches, '\t',
            phase_reweighted, '\t',
            phase_sum, '\t',
            average_phase, '\t',
            primitive_signed[1], '\t',
            primitive_signed[2], '\t',
            primitive_signed[3], '\t',
            primitive_signed[4], '\t',
            local_moment_signed, '\t',
            primitive_signed[5], '\t',
            diamagnetic_signed
        ))
    end
end

function measure_equal_time_observables(system, ρup, ρdn; kx_value=nothing)
    nsites = system.V
    energy = real.(ce_measure_Energy(system, ρup, ρdn)) ./ nsites
    docc = sum(real.(diag(ρup.ρ₁) .* diag(ρdn.ρ₁))) / nsites
    kx = kx_value === nothing ? real(ce_measure_KxPerSite(system, ρup, ρdn)) : real(kx_value)
    return (energy[1], energy[2], energy[3], docc, kx)
end

function reset_corr_sampler!(sampler::CorrFuncSampler)
    sampler.s_counter[] = 1
    fill!(sampler.nᵢ₊ᵣnᵢ, 0)
    fill!(sampler.Sᵢ₊ᵣSᵢ, 0)
    fill!(sampler.Sˣᵢ₊ᵣSˣᵢ, 0)
    return sampler
end

function measure_equal_time_charge_spin_wedge!(
    sampler::CorrFuncSampler,
    ρup::DensityMatrix,
    ρdn::DensityMatrix,
)
    reset_corr_sampler!(sampler)
    measure_ChargeCorr(sampler, ρup, ρdn)
    measure_SpinCorr(sampler, ρup, ρdn)
    charge = real.(sampler.nᵢ₊ᵣnᵢ[:, 1])
    spin_s_s = real.(sampler.Sᵢ₊ᵣSᵢ[:, 1])
    return charge, spin_s_s
end

function fold_displacement(d::Int, L::Int)
    dm = mod(d, L)
    return min(dm, mod(-dm, L))
end

function expand_wedge_to_full(
    values::AbstractVector{<:Real},
    delta_to_index::Dict{Tuple{Int,Int},Int},
    lx::Int,
    ly::Int,
)
    full = zeros(Float64, lx, ly)
    @inbounds for dx in 0:(lx - 1), dy in 0:(ly - 1)
        fdx = fold_displacement(dx, lx)
        fdy = fold_displacement(dy, ly)
        idx = delta_to_index[(fdx, fdy)]
        full[dx + 1, dy + 1] = Float64(values[idx])
    end
    return full
end

function d4_momentum_reps(lx::Int, ly::Int)
    if lx != ly
        return vec([(nx, ny) for nx in 0:div(lx, 2), ny in 0:div(ly, 2)])
    end
    half = div(lx, 2)
    reps = Tuple{Int,Int}[]
    for mx in 0:half
        for my in 0:mx
            push!(reps, (mx, my))
        end
    end
    return reps
end

function d4_momentum_star(mx::Int, my::Int, L::Int)
    seen = Set{Tuple{Int,Int}}()
    star = Tuple{Int,Int}[]
    for (a, b) in ((mx, my), (my, mx))
        for sx in (-1, 1), sy in (-1, 1)
            q = (mod(sx * a, L), mod(sy * b, L))
            if !(q in seen)
                push!(seen, q)
                push!(star, q)
            end
        end
    end
    return star
end

function structure_factor_from_full(full::AbstractMatrix{<:Real}, nx::Int, ny::Int, lx::Int, ly::Int)
    total = 0.0
    @inbounds for dx in 0:(lx - 1), dy in 0:(ly - 1)
        phase = 2π * (nx * dx / lx + ny * dy / ly)
        total += cos(phase) * Float64(full[dx + 1, dy + 1])
    end
    return total
end

function structure_samples_from_wedge(
    charge_wedge::AbstractVector{<:Real},
    spin_wedge::AbstractVector{<:Real},
    delta_to_index::Dict{Tuple{Int,Int},Int},
    q_reps::Vector{Tuple{Int,Int}},
    lx::Int,
    ly::Int,
    density::Float64,
)
    charge_full = expand_wedge_to_full(charge_wedge, delta_to_index, lx, ly)
    spin_full = expand_wedge_to_full(spin_wedge, delta_to_index, lx, ly)
    charge_connected_full = charge_full .- density^2

    nq = length(q_reps)
    charge_raw = zeros(Float64, nq)
    charge_connected = zeros(Float64, nq)
    spin_s_s = zeros(Float64, nq)
    for (iq, (mx, my)) in enumerate(q_reps)
        if lx == ly
            star = d4_momentum_star(mx, my, lx)
        else
            star = [(mx, my)]
        end
        for (nx, ny) in star
            charge_raw[iq] += structure_factor_from_full(charge_full, nx, ny, lx, ly)
            charge_connected[iq] += structure_factor_from_full(charge_connected_full, nx, ny, lx, ly)
            spin_s_s[iq] += structure_factor_from_full(spin_full, nx, ny, lx, ly)
        end
        charge_raw[iq] /= length(star)
        charge_connected[iq] /= length(star)
        spin_s_s[iq] /= length(star)
    end
    return charge_raw, charge_connected, spin_s_s
end

function neighbor_shell_definitions(lx::Int, ly::Int; nshells::Int=4)
    by_r2 = Dict{Int,Vector{Tuple{Int,Int}}}()
    for dx in 0:(lx - 1), dy in 0:(ly - 1)
        dx == 0 && dy == 0 && continue
        fx = fold_displacement(dx, lx)
        fy = fold_displacement(dy, ly)
        r2 = fx^2 + fy^2
        push!(get!(by_r2, r2, Tuple{Int,Int}[]), (dx, dy))
    end
    shells = Tuple{Int,Int,Vector{Tuple{Int,Int}}}[]
    r2_values = sort(collect(keys(by_r2)))
    for (ishell, r2) in enumerate(r2_values[1:min(nshells, length(r2_values))])
        push!(shells, (ishell, r2, sort(unique(by_r2[r2]))))
    end
    return shells
end

function shell_samples_from_wedge(
    charge_wedge::AbstractVector{<:Real},
    spin_wedge::AbstractVector{<:Real},
    delta_to_index::Dict{Tuple{Int,Int},Int},
    shells::Vector{Tuple{Int,Int,Vector{Tuple{Int,Int}}}},
    lx::Int,
    ly::Int,
    density::Float64,
)
    charge_full = expand_wedge_to_full(charge_wedge, delta_to_index, lx, ly)
    spin_full = expand_wedge_to_full(spin_wedge, delta_to_index, lx, ly)
    nshell = length(shells)
    charge_raw = zeros(Float64, nshell)
    charge_connected = zeros(Float64, nshell)
    spin_s_s = zeros(Float64, nshell)
    @inbounds for (ishell, (_, _, vectors)) in enumerate(shells)
        for (dx, dy) in vectors
            c = charge_full[dx + 1, dy + 1]
            s = spin_full[dx + 1, dy + 1]
            charge_raw[ishell] += c
            charge_connected[ishell] += c - density^2
            spin_s_s[ishell] += s
        end
        invn = 1.0 / length(vectors)
        charge_raw[ishell] *= invn
        charge_connected[ishell] *= invn
        spin_s_s[ishell] *= invn
    end
    return charge_raw, charge_connected, spin_s_s
end

function write_equal_time_wedge_summary(
    outdir,
    params,
    deltas::Vector{Tuple{Int,Int}},
    charge_mean,
    charge_err,
    spin_mean,
    spin_err,
    nsamples::Int,
    batches::Int,
    phase_sum::Float64=Float64(nsamples),
    average_phase::Float64=1.0,
    phase_reweighted::Bool=false,
    charge_signed_sum=nothing,
    spin_signed_sum=nothing,
)
    ntotal = params["nup"] + params["ndn"]
    density = ntotal / (params["lx"] * params["ly"])
    open(joinpath(outdir, "equal_time_charge_spin_wedge_qmc.tsv"), "w") do io
        println(io, "dx\tdy\tcharge_corr_raw\tcharge_corr_raw_stderr\tcharge_corr_connected\tcharge_corr_connected_stderr\tspin_corr_s_s\tspin_corr_s_s_stderr\tspin_corr_SzSz\tspin_corr_SzSz_stderr\tnsamples\tbatches\tphase_reweighted\tphase_sum\taverage_phase\tcharge_corr_raw_signed_sum\tcharge_corr_connected_signed_sum\tspin_corr_s_s_signed_sum\tspin_corr_SzSz_signed_sum")
        for (i, (dx, dy)) in enumerate(deltas)
            charge_signed = charge_signed_sum === nothing ? charge_mean[i] * phase_sum : charge_signed_sum[i]
            spin_signed = spin_signed_sum === nothing ? spin_mean[i] * phase_sum : spin_signed_sum[i]
            println(io, string(
                dx, '\t', dy, '\t',
                charge_mean[i], '\t', charge_err[i], '\t',
                charge_mean[i] - density^2, '\t', charge_err[i], '\t',
                spin_mean[i], '\t', spin_err[i], '\t',
                0.25 * spin_mean[i], '\t', 0.25 * spin_err[i], '\t',
                nsamples, '\t', batches, '\t',
                phase_reweighted, '\t', phase_sum, '\t', average_phase, '\t',
                charge_signed, '\t', charge_signed - density^2 * phase_sum, '\t',
                spin_signed, '\t', 0.25 * spin_signed,
            ))
        end
    end
end

function write_equal_time_structure_summary(
    outdir,
    params,
    q_reps::Vector{Tuple{Int,Int}},
    charge_raw_mean,
    charge_raw_err,
    charge_conn_mean,
    charge_conn_err,
    spin_mean,
    spin_err,
    nsamples::Int,
    batches::Int,
    phase_sum::Float64=Float64(nsamples),
    average_phase::Float64=1.0,
    phase_reweighted::Bool=false,
    charge_raw_signed_sum=nothing,
    charge_conn_signed_sum=nothing,
    spin_signed_sum=nothing,
)
    lx, ly = params["lx"], params["ly"]
    open(joinpath(outdir, "equal_time_structure_factors_qmc.tsv"), "w") do io
        println(io, "mx\tmy\tqx\tqy\tcharge_structure_raw\tcharge_structure_raw_stderr\tcharge_structure_connected\tcharge_structure_connected_stderr\tspin_structure_s_s\tspin_structure_s_s_stderr\tspin_structure_SzSz\tspin_structure_SzSz_stderr\tnsamples\tbatches\tphase_reweighted\tphase_sum\taverage_phase\tcharge_structure_raw_signed_sum\tcharge_structure_connected_signed_sum\tspin_structure_s_s_signed_sum\tspin_structure_SzSz_signed_sum")
        for (i, (mx, my)) in enumerate(q_reps)
            raw_signed = charge_raw_signed_sum === nothing ? charge_raw_mean[i] * phase_sum : charge_raw_signed_sum[i]
            conn_signed = charge_conn_signed_sum === nothing ? charge_conn_mean[i] * phase_sum : charge_conn_signed_sum[i]
            spin_signed = spin_signed_sum === nothing ? spin_mean[i] * phase_sum : spin_signed_sum[i]
            println(io, string(
                mx, '\t', my, '\t',
                2π * mx / lx, '\t', 2π * my / ly, '\t',
                charge_raw_mean[i], '\t', charge_raw_err[i], '\t',
                charge_conn_mean[i], '\t', charge_conn_err[i], '\t',
                spin_mean[i], '\t', spin_err[i], '\t',
                0.25 * spin_mean[i], '\t', 0.25 * spin_err[i], '\t',
                nsamples, '\t', batches, '\t',
                phase_reweighted, '\t', phase_sum, '\t', average_phase, '\t',
                raw_signed, '\t', conn_signed, '\t', spin_signed, '\t', 0.25 * spin_signed,
            ))
        end
    end
end

function write_equal_time_shell_summary(
    outdir,
    shells::Vector{Tuple{Int,Int,Vector{Tuple{Int,Int}}}},
    charge_raw_mean,
    charge_raw_err,
    charge_conn_mean,
    charge_conn_err,
    spin_mean,
    spin_err,
    nsamples::Int,
    batches::Int,
    phase_sum::Float64=Float64(nsamples),
    average_phase::Float64=1.0,
    phase_reweighted::Bool=false,
    charge_raw_signed_sum=nothing,
    charge_conn_signed_sum=nothing,
    spin_signed_sum=nothing,
)
    open(joinpath(outdir, "equal_time_neighbor_shells_qmc.tsv"), "w") do io
        println(io, "shell\tr2\tvectors\tcharge_corr_raw\tcharge_corr_raw_stderr\tcharge_corr_connected\tcharge_corr_connected_stderr\tspin_corr_s_s\tspin_corr_s_s_stderr\tspin_corr_SzSz\tspin_corr_SzSz_stderr\tnsamples\tbatches\tphase_reweighted\tphase_sum\taverage_phase\tcharge_corr_raw_signed_sum\tcharge_corr_connected_signed_sum\tspin_corr_s_s_signed_sum\tspin_corr_SzSz_signed_sum")
        for (i, (shell, r2, vectors)) in enumerate(shells)
            vecstr = join(["($(dx),$(dy))" for (dx, dy) in vectors], ";")
            raw_signed = charge_raw_signed_sum === nothing ? charge_raw_mean[i] * phase_sum : charge_raw_signed_sum[i]
            conn_signed = charge_conn_signed_sum === nothing ? charge_conn_mean[i] * phase_sum : charge_conn_signed_sum[i]
            spin_signed = spin_signed_sum === nothing ? spin_mean[i] * phase_sum : spin_signed_sum[i]
            println(io, string(
                shell, '\t', r2, '\t', vecstr, '\t',
                charge_raw_mean[i], '\t', charge_raw_err[i], '\t',
                charge_conn_mean[i], '\t', charge_conn_err[i], '\t',
                spin_mean[i], '\t', spin_err[i], '\t',
                0.25 * spin_mean[i], '\t', 0.25 * spin_err[i], '\t',
                nsamples, '\t', batches, '\t',
                phase_reweighted, '\t', phase_sum, '\t', average_phase, '\t',
                raw_signed, '\t', conn_signed, '\t', spin_signed, '\t', 0.25 * spin_signed,
            ))
        end
    end
end

function measure_bkt_observables(
    system,
    ρup,
    ρdn,
    Bup,
    Bdn,
    prefix_up,
    suffix_up,
    prefix_dn,
    suffix_dn;
    estimator::String="projected",
    refresh_interval::Int=10,
    adaptive_refresh::Bool=false,
    refresh_tol::Float64=1e-6,
    refresh_min::Int=1,
    refresh_max::Int=max(refresh_interval, refresh_min),
    refresh_growth_patience::Int=3,
    current_drift_tol::Float64=1e-2,
    allow_imbalanced_canonical_recursion::Bool=false,
    walker=nothing,
)
    lx, ly, lz = system.Ns
    lz == 1 || error("BKT stiffness observables assume a 2D lattice")
    qxmin = 2π / lx
    qymin = 2π / ly
    momenta = [(qxmin, 0.0), (0.0, qymin)]
    if estimator == "projected"
        λ = measure_current_responses_unequaltime(
            system, ρup, ρdn, prefix_up, suffix_up, prefix_dn, suffix_dn,
            momenta,
        )
    elseif estimator == "canonical-recursion"
        λ = measure_current_responses_unequaltime_canonical_recursion(
            system, ρup, ρdn, Bup, Bdn, prefix_up, suffix_up, prefix_dn, suffix_dn,
            momenta;
            allow_imbalanced=allow_imbalanced_canonical_recursion,
            drift_tol=current_drift_tol,
        )
    elseif estimator == "canonical-stable"
        λ = measure_current_responses_unequaltime_canonical_stable(
            system, ρup, ρdn, prefix_up, prefix_dn,
            momenta;
            allow_imbalanced=allow_imbalanced_canonical_recursion,
        )
    elseif estimator in ("canonical-accumulator", "canonical-smoqy", "canonical-structured")
        walker === nothing && error("$(estimator) requires the current walker so dense B/prefix construction can be skipped")
        try
            λ = measure_current_responses_unequaltime_structured_accumulator(
                system, ρup, ρdn, walker,
                momenta;
                allow_imbalanced=allow_imbalanced_canonical_recursion,
                drift_tol=current_drift_tol,
            )
        catch err
            msg = sprint(showerror, err)
            if occursin("structured canonical current accumulator drift too large", msg) ||
               occursin("structured canonical current accumulator produced non-finite", msg)
                @warn(
                    "canonical-smoqy structured current accumulator failed guard; " *
                    "falling back to stable fixed-N no-Fourier estimator for this sample",
                    error = msg,
                )
                fallback_Bup = Bup
                fallback_Bdn = Bdn
                fallback_prefix_up = prefix_up
                fallback_suffix_up = suffix_up
                fallback_prefix_dn = prefix_dn
                fallback_suffix_dn = suffix_dn
                if fallback_Bup === nothing || fallback_Bdn === nothing
                    fallback_Bup, fallback_Bdn = build_B_slices(system, walker)
                    fallback_prefix_up, fallback_suffix_up = build_prefix_suffix(fallback_Bup)
                    fallback_prefix_dn, fallback_suffix_dn = build_prefix_suffix(fallback_Bdn)
                end
                λ = measure_current_responses_unequaltime_canonical_stable(
                    system, ρup, ρdn,
                    fallback_prefix_up, fallback_prefix_dn,
                    momenta;
                    allow_imbalanced=allow_imbalanced_canonical_recursion,
                )
            else
                rethrow()
            end
        end
    elseif estimator == "propagated"
        λ = measure_current_responses_unequaltime_propagated(
            system, ρup, ρdn, Bup, Bdn, prefix_up, suffix_up, prefix_dn, suffix_dn,
            momenta;
            refresh_interval=refresh_interval,
            adaptive_refresh=adaptive_refresh,
            refresh_tol=refresh_tol,
            refresh_min=refresh_min,
            refresh_max=refresh_max,
            refresh_growth_patience=refresh_growth_patience,
        )
    else
        error("unknown --bkt-current-estimator=$(estimator); expected projected, propagated, canonical-recursion, canonical-stable, canonical-accumulator, canonical-smoqy, or canonical-structured")
    end
    λL = real(λ[1])
    λT = real(λ[2])
    kx = real(ce_measure_KxPerSite(system, ρup, ρdn))
    ρs_current = 0.25 * (λL - λT)
    ρs_diamagnetic = 0.25 * (-kx - λT)
    return (λL, λT, kx, ρs_current, ρs_diamagnetic)
end

function write_tau_summary(outdir, params, Cτ_mean, Cτ_err, nsamples, batches)
    open(joinpath(outdir, "greens_tau0_qmc.tsv"), "w") do io
        println(io, "slice\ttau\tCtau_mean\tCtau_stderr\tnsamples\tbatches")
        for l in eachindex(Cτ_mean)
            println(io, string(l - 1, '\t', (l - 1) * params["dtau"], '\t', Cτ_mean[l], '\t', Cτ_err[l], '\t', nsamples, '\t', batches))
        end
    end
end

function write_tau_remove_summary(outdir, params, Rτ_mean, Rτ_err, nsamples, batches)
    open(joinpath(outdir, "greens_tau0_remove_qmc.tsv"), "w") do io
        println(io, "slice\ttau\tRtau_mean\tRtau_stderr\tnsamples\tbatches")
        for l in eachindex(Rτ_mean)
            println(io, string(l - 1, '\t', (l - 1) * params["dtau"], '\t', Rτ_mean[l], '\t', Rτ_err[l], '\t', nsamples, '\t', batches))
        end
    end
end

function write_realspace_tau_summary(outdir, filename, params, G_mean, G_re_err, G_im_err, nsamples, batches)
    ntau, lx, ly = size(G_mean)
    open(joinpath(outdir, filename), "w") do io
        println(io, "slice\ttau\tdx\tdy\tGreal_mean\tGreal_stderr\tGimag_mean\tGimag_stderr\tnsamples\tbatches")
        for l in 1:ntau, dy in 0:(ly - 1), dx in 0:(lx - 1)
            val = G_mean[l, dx + 1, dy + 1]
            println(io, string(
                l - 1, '\t', (l - 1) * params["dtau"], '\t',
                dx, '\t', dy, '\t',
                real(val), '\t', G_re_err[l, dx + 1, dy + 1], '\t',
                imag(val), '\t', G_im_err[l, dx + 1, dy + 1], '\t',
                nsamples, '\t', batches
            ))
        end
    end
end

function write_momentum_tau_summary(outdir, filename, params, G_mean, G_re_err, G_im_err, nsamples, batches)
    ntau, lx, ly = size(G_mean)
    open(joinpath(outdir, filename), "w") do io
        println(io, "slice\ttau\tnx\tny\tkx\tky\tGreal_mean\tGreal_stderr\tGimag_mean\tGimag_stderr\tnsamples\tbatches")
        for l in 1:ntau, ny in 0:(ly - 1), nx in 0:(lx - 1)
            kx = 2π * nx / lx
            ky = 2π * ny / ly
            val = G_mean[l, nx + 1, ny + 1]
            println(io, string(
                l - 1, '\t', (l - 1) * params["dtau"], '\t',
                nx, '\t', ny, '\t', kx, '\t', ky, '\t',
                real(val), '\t', G_re_err[l, nx + 1, ny + 1], '\t',
                imag(val), '\t', G_im_err[l, nx + 1, ny + 1], '\t',
                nsamples, '\t', batches
            ))
        end
    end
end

function mean_reim_err(all_values::Array{ComplexF64,4})
    nsamples = size(all_values, 1)
    mean_values = dropdims(mean(all_values, dims=1), dims=1)
    re_err = dropdims(std(real.(all_values), dims=1; corrected=true), dims=1) ./ sqrt(nsamples)
    im_err = dropdims(std(imag.(all_values), dims=1; corrected=true), dims=1) ./ sqrt(nsamples)
    return mean_values, re_err, im_err
end

function stderr_from_sums(sum_values::AbstractArray{Float64}, sumsq_values::AbstractArray{Float64}, nsamples::Int)
    if nsamples <= 1
        return fill(NaN, size(sum_values))
    end
    mean_values = sum_values ./ nsamples
    var_values = max.(sumsq_values .- nsamples .* mean_values .^ 2, 0.0) ./ (nsamples - 1)
    return sqrt.(var_values ./ nsamples)
end

function canonical_configuration_phase(walker; imag_tol::Float64=1e-8)
    z = ComplexF64(walker.sign[1] * walker.sign[2])
    magnitude = abs(z)
    isfinite(magnitude) && magnitude > 0 || error("invalid canonical determinant phase $(z)")
    unit_phase = z / magnitude
    abs(imag(unit_phase)) <= imag_tol || error(
        "positive-U spin-HS run produced a non-real determinant phase $(unit_phase); " *
        "the real-sign reweighting accumulator cannot represent it",
    )
    phase = real(unit_phase)
    abs(abs(phase) - 1.0) <= 10 * imag_tol || error("canonical determinant sign is not ±1: $(phase)")
    return phase
end

function ratio_mean_err_from_sums(
    weighted_sum::AbstractArray{Float64},
    raw_sum::AbstractArray{Float64},
    sumsq_values::AbstractArray{Float64},
    phase_sum::Float64,
    nsamples::Int,
)
    nsamples > 0 || return fill(NaN, size(weighted_sum)), fill(NaN, size(weighted_sum))
    average_phase = phase_sum / nsamples
    # A rank-local running sign sum can cross exactly through zero at small
    # sample counts.  This is normal and must not terminate the Markov chain.
    # The ratio is temporarily undefined; raw signed numerators are written
    # separately and pooled across ranks by the MPI combiner.
    if abs(average_phase) <= 10 * eps(Float64)
        return fill(NaN, size(weighted_sum)), fill(NaN, size(weighted_sum))
    end
    mean_values = weighted_sum ./ phase_sum
    if nsamples <= 1
        return mean_values, fill(NaN, size(weighted_sum))
    end
    # Delta-method/influence-variable error for R=<p O>/<p>.  For real
    # determinant signs p=±1, [p(O-R)]²=(O-R)², so the raw first and second
    # moments below are sufficient and checkpoint-friendly.
    centered_ss = max.(
        sumsq_values .- 2.0 .* mean_values .* raw_sum .+ nsamples .* mean_values .^ 2,
        0.0,
    )
    influence_var = centered_ss ./ (nsamples - 1)
    stderr = sqrt.(influence_var ./ nsamples) ./ abs(average_phase)
    return mean_values, stderr
end

function complex_mean_reim_err_from_sums(
    sum_values::AbstractArray{ComplexF64},
    sumsq_re::AbstractArray{Float64},
    sumsq_im::AbstractArray{Float64},
    nsamples::Int,
)
    mean_values = sum_values ./ nsamples
    if nsamples <= 1
        return mean_values, fill(NaN, size(sum_values)), fill(NaN, size(sum_values))
    end
    mean_re = real.(mean_values)
    mean_im = imag.(mean_values)
    var_re = max.(sumsq_re .- nsamples .* mean_re .^ 2, 0.0) ./ (nsamples - 1)
    var_im = max.(sumsq_im .- nsamples .* mean_im .^ 2, 0.0) ./ (nsamples - 1)
    return mean_values, sqrt.(var_re ./ nsamples), sqrt.(var_im ./ nsamples)
end

function add_complex_sample!(sum_values, sumsq_re, sumsq_im, sample_values)
    sum_values .+= sample_values
    sumsq_re .+= real.(sample_values) .^ 2
    sumsq_im .+= imag.(sample_values) .^ 2
    return nothing
end

mutable struct MetropolisRatioStats
    n_total::Int
    n_finite_nonnegative::Int
    n_nonfinite_or_negative::Int
    n_ge_one::Int
    n_lt_one::Int
    n_lt_1e_minus_2::Int
    n_lt_1e_minus_6::Int
    n_lt_1e_minus_12::Int
    sum_accept_prob::Float64
    sum_log10_ratio::Float64
    min_log10_ratio::Float64
    max_log10_ratio::Float64
end

function MetropolisRatioStats()
    return MetropolisRatioStats(0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, Inf, -Inf)
end

function update_metropolis_ratio_stats!(stats::MetropolisRatioStats, walker)
    isempty(walker.tmp_r) && return stats
    tiny = floatmin(Float64)
    @inbounds for raw_r in walker.tmp_r
        stats.n_total += 1
        r = real(raw_r)
        if isfinite(r) && r >= 0
            stats.n_finite_nonnegative += 1
            accept_prob = min(1.0, r)
            stats.sum_accept_prob += accept_prob
            if r >= 1
                stats.n_ge_one += 1
            else
                stats.n_lt_one += 1
            end
            r < 1e-2 && (stats.n_lt_1e_minus_2 += 1)
            r < 1e-6 && (stats.n_lt_1e_minus_6 += 1)
            r < 1e-12 && (stats.n_lt_1e_minus_12 += 1)
            log10r = log10(max(r, tiny))
            stats.sum_log10_ratio += log10r
            stats.min_log10_ratio = min(stats.min_log10_ratio, log10r)
            stats.max_log10_ratio = max(stats.max_log10_ratio, log10r)
        else
            stats.n_nonfinite_or_negative += 1
        end
    end
    empty!(walker.tmp_r)
    return stats
end

function write_metropolis_ratio_diagnostics(
    outdir::AbstractString,
    stats::MetropolisRatioStats,
    nsamples_total::Int,
    completed_batches::Int,
    warmups_completed::Int,
)
    path = joinpath(outdir, "metropolis_ratio_diagnostics.tsv")
    mkpath(dirname(path))
    finite = stats.n_finite_nonnegative
    expected_acceptance = finite > 0 ? stats.sum_accept_prob / finite : NaN
    mean_log10_ratio = finite > 0 ? stats.sum_log10_ratio / finite : NaN
    min_log10_ratio = isfinite(stats.min_log10_ratio) ? stats.min_log10_ratio : NaN
    max_log10_ratio = isfinite(stats.max_log10_ratio) ? stats.max_log10_ratio : NaN
    frac_ge_one = finite > 0 ? stats.n_ge_one / finite : NaN
    frac_lt_one = finite > 0 ? stats.n_lt_one / finite : NaN
    frac_lt_1e_minus_2 = finite > 0 ? stats.n_lt_1e_minus_2 / finite : NaN
    frac_lt_1e_minus_6 = finite > 0 ? stats.n_lt_1e_minus_6 / finite : NaN
    frac_lt_1e_minus_12 = finite > 0 ? stats.n_lt_1e_minus_12 / finite : NaN
    open(path, "w") do io
        println(io, join([
            "warmups_completed",
            "completed_batches",
            "nsamples",
            "proposal_count",
            "finite_nonnegative_count",
            "nonfinite_or_negative_count",
            "expected_acceptance_mean_min1r",
            "mean_log10_ratio",
            "min_log10_ratio",
            "max_log10_ratio",
            "frac_ratio_ge_1",
            "frac_ratio_lt_1",
            "frac_ratio_lt_1e_minus_2",
            "frac_ratio_lt_1e_minus_6",
            "frac_ratio_lt_1e_minus_12",
        ], "\t"))
        println(io, join(string.(Any[
            warmups_completed,
            completed_batches,
            nsamples_total,
            stats.n_total,
            stats.n_finite_nonnegative,
            stats.n_nonfinite_or_negative,
            expected_acceptance,
            mean_log10_ratio,
            min_log10_ratio,
            max_log10_ratio,
            frac_ge_one,
            frac_lt_one,
            frac_lt_1e_minus_2,
            frac_lt_1e_minus_6,
            frac_lt_1e_minus_12,
        ]), "\t"))
    end
    return path
end

const CHECKPOINT_VERSION = 1
const CHECKPOINT_CORE_PARAM_KEYS = [
    "lx",
    "ly",
    "nup",
    "ndn",
    "u",
    "dtau",
    "beta",
    "nwarmups",
    "batch_nsamples",
    "measure_interval",
    "stab_interval",
    "cluster_size",
    "num_fourier_points",
    "nfreq",
    "use_lowrank",
    "lr_thld",
    "force_symmetry",
    "use_cluster_update",
    "measure_greens",
    "measure_bkt",
    "measure_equal_time",
    "seed",
    "max_batches",
    "use_charge_hs",
    "sys_type",
]

function checkpoint_core_params(params)
    return Dict(key => params[key] for key in CHECKPOINT_CORE_PARAM_KEYS)
end

function resolve_checkpoint_path(outdir::AbstractString, params)
    checkpoint_file = String(params["checkpoint_file"])
    if isempty(checkpoint_file)
        checkpoint_file = "checkpoint.jls"
    end
    return isabspath(checkpoint_file) ? checkpoint_file : joinpath(outdir, checkpoint_file)
end

function save_checkpoint(path::AbstractString, state)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid())
    open(tmp, "w") do io
        serialize(io, state)
    end
    mv(tmp, path; force=true)
    return nothing
end

checkpoint_status_path(path::AbstractString) = string(path, ".status")

function save_checkpoint_status(
    path::AbstractString,
    completed_batches::Int,
    warmups_completed::Int,
    nsamples_total::Int,
    reason::AbstractString,
)
    status_path = checkpoint_status_path(path)
    mkpath(dirname(status_path))
    tmp = string(status_path, ".tmp.", getpid())
    open(tmp, "w") do io
        println(io, "completed_batches=$(completed_batches)")
        println(io, "warmups_completed=$(warmups_completed)")
        println(io, "nsamples=$(nsamples_total)")
        println(io, "reason=$(reason)")
        println(io, "unix_time=$(time())")
    end
    mv(tmp, status_path; force=true)
    return nothing
end

function read_checkpoint_progress(status_path::AbstractString)
    completed_batches = -1
    warmups_completed = -1
    try
        for line in eachline(status_path)
            if startswith(line, "completed_batches=")
                completed_batches = parse(Int, split(line, "=", limit=2)[2])
            elseif startswith(line, "warmups_completed=")
                warmups_completed = parse(Int, split(line, "=", limit=2)[2])
            end
        end
    catch err
        return (-1, -1)
    end
    return (completed_batches, warmups_completed)
end

function count_peer_checkpoints(
    root_dir::AbstractString,
    checkpoint_file::AbstractString,
    min_completed_batches::Int,
    min_warmups_completed::Int,
)
    isempty(root_dir) && return 1
    ranks_dir = joinpath(root_dir, "ranks")
    isdir(ranks_dir) || return 0
    rel_checkpoint = isempty(checkpoint_file) ? "checkpoint.jls" : checkpoint_file
    if isabspath(rel_checkpoint)
        status_path = checkpoint_status_path(rel_checkpoint)
        if isfile(status_path)
            done_batches, done_warmups = read_checkpoint_progress(status_path)
            return (done_batches >= min_completed_batches && done_warmups >= min_warmups_completed) ? 1 : 0
        end
        return 0
    end

    count = 0
    for rank_dir in readdir(ranks_dir; join=true)
        isdir(rank_dir) || continue
        status_path = checkpoint_status_path(joinpath(rank_dir, rel_checkpoint))
        if isfile(status_path)
            done_batches, done_warmups = read_checkpoint_progress(status_path)
            if done_batches >= min_completed_batches && done_warmups >= min_warmups_completed
                count += 1
            end
        end
    end
    return count
end

function wait_for_peer_checkpoints(params, min_completed_batches::Int, min_warmups_completed::Int)
    world_size = params["checkpoint_world_size"]
    root_dir = String(params["checkpoint_root_dir"])
    if world_size <= 1 || isempty(root_dir)
        return true
    end
    timeout = params["checkpoint_sync_timeout_seconds"]
    poll = max(params["checkpoint_sync_poll_seconds"], 0.25)
    deadline = time() + timeout
    while true
        count = count_peer_checkpoints(
            root_dir,
            String(params["checkpoint_file"]),
            min_completed_batches,
            min_warmups_completed,
        )
        if count >= world_size
            println("checkpoint_peer_sync count=$(count)/$(world_size) min_completed_batches=$(min_completed_batches) min_warmups_completed=$(min_warmups_completed)")
            flush(stdout)
            return true
        end
        if time() >= deadline
            println("checkpoint_peer_sync_timeout count=$(count)/$(world_size) min_completed_batches=$(min_completed_batches) min_warmups_completed=$(min_warmups_completed)")
            flush(stdout)
            return false
        end
        sleep(poll)
    end
end

function load_checkpoint(path::AbstractString)
    return open(deserialize, path)
end

function restore_default_rng!(rng)
    copy!(Random.default_rng(), rng)
    return nothing
end

function validate_checkpoint_state(state, params, system)
    version = get(state, "version", nothing)
    version == CHECKPOINT_VERSION || error("checkpoint version mismatch: got $(version), expected $(CHECKPOINT_VERSION)")

    saved_core = get(state, "core_params", nothing)
    current_core = checkpoint_core_params(params)
    saved_core isa Dict || error("checkpoint is missing core_params")
    mismatches = String[]
    for key in CHECKPOINT_CORE_PARAM_KEYS
        if !haskey(saved_core, key) || saved_core[key] != current_core[key]
            saved_value = haskey(saved_core, key) ? saved_core[key] : "<missing>"
            if key == "max_batches" && saved_value isa Integer && current_core[key] isa Integer
                # Safe resume policy: max_batches is a stopping target, not part
                # of the Markov state or accumulated observables. Allow changing
                # it between checkpointed continuation legs; the completed_batches
                # <= max_batches check below still protects against resuming past
                # the requested target.
                println("checkpoint_parameter_notice max_batches saved=$(saved_value) current=$(current_core[key])")
                continue
            elseif key == "num_fourier_points" && saved_value isa Integer && current_core[key] isa Integer
                min_nft = maximum(max(N, system.V - N) for N in system.N) + 1
                if saved_value >= min_nft && current_core[key] >= min_nft
                    # Both quadratures are alias-free for the fixed-N projection,
                    # so changing Nft only changes numerical roundoff. This lets
                    # optimized continuations use the minimum alias-free Nft while
                    # preserving accumulated samples from a larger-Nft run.
                    println("checkpoint_parameter_notice num_fourier_points saved=$(saved_value) current=$(current_core[key]) min_alias_free=$(min_nft)")
                    continue
                end
            end
            push!(mismatches, string(key, " saved=", saved_value, " current=", current_core[key]))
        end
    end
    isempty(mismatches) || error("checkpoint parameter mismatch:\n  " * join(mismatches, "\n  "))

    saved_phase_reweight = Bool(get(state, "phase_reweight", false))
    current_phase_reweight = params["phase_reweight"]
    saved_phase_reweight == current_phase_reweight || error(
        "checkpoint phase-reweight mismatch: saved=$(saved_phase_reweight) current=$(current_phase_reweight)",
    )

    get(state, "time_slices", system.L) == system.L || error("checkpoint time-slice mismatch")
    get(state, "volume", system.V) == system.V || error("checkpoint volume mismatch")
    return nothing
end

function checkpoint_due(params, last_checkpoint_time::Float64, completed_batches::Int)
    params["checkpoint_enable"] || return false
    every = params["checkpoint_every_batches"]
    freq_seconds = 3600.0 * params["checkpoint_freq_hours"]
    if every <= 0 && freq_seconds <= 0
        return false
    end
    if every > 0 && completed_batches % every != 0
        return false
    end
    return freq_seconds <= 0 || (time() - last_checkpoint_time) >= freq_seconds
end

function runtime_limit_reached(params, start_time::Float64)
    limit_seconds = 3600.0 * params["runtime_limit_hours"]
    return params["checkpoint_enable"] && limit_seconds > 0 && (time() - start_time) >= limit_seconds
end

function main(args=ARGS)
    params = parse_args(args)
    root = normpath(joinpath(@__DIR__, "..", ".."))
    outdir = joinpath(root, params["output_dir"])
    mkpath(outdir)
    run_start_time = time()

    system = build_system(params)
    qmc = build_qmc(system, params)
    write_metadata(outdir, params, system.L)

    ntau = system.L
    lx, ly, _ = system.Ns
    nsamples_total = 0

    measure_greens = params["measure_greens"]
    measure_bkt = params["measure_bkt"]
    measure_equal_time = params["measure_equal_time"]
    measure_equal_time_correlations = measure_equal_time && params["measure_equal_time_correlations"]
    phase_reweight = params["phase_reweight"]
    if phase_reweight && (measure_greens || measure_bkt || !measure_equal_time)
        error("--phase-reweight=true is currently supported only for equal-time-only runs")
    end
    structured_bkt_estimator = params["bkt_current_estimator"] in ("canonical-accumulator", "canonical-smoqy", "canonical-structured")
    checkpoint_enabled = params["checkpoint_enable"]
    checkpoint_path = resolve_checkpoint_path(outdir, params)
    if params["checkpoint_every_batches"] < 0
        error("--checkpoint-every-batches must be non-negative")
    end
    if params["checkpoint_freq_hours"] < 0
        error("--checkpoint-freq-hours must be non-negative")
    end
    if params["runtime_limit_hours"] < 0
        error("--runtime-limit-hours must be non-negative")
    end
    if params["checkpoint_warmup_chunk"] <= 0
        error("--checkpoint-warmup-chunk must be positive")
    end

    sum_Cτ = measure_greens ? zeros(Float64, ntau) : zeros(Float64, 0)
    sumsq_Cτ = measure_greens ? zeros(Float64, ntau) : zeros(Float64, 0)
    sum_Rτ = measure_greens ? zeros(Float64, ntau) : zeros(Float64, 0)
    sumsq_Rτ = measure_greens ? zeros(Float64, ntau) : zeros(Float64, 0)

    sum_add_r = measure_greens ? zeros(ComplexF64, ntau, lx, ly) : zeros(ComplexF64, 0, 0, 0)
    sumsq_add_r_re = measure_greens ? zeros(Float64, ntau, lx, ly) : zeros(Float64, 0, 0, 0)
    sumsq_add_r_im = measure_greens ? zeros(Float64, ntau, lx, ly) : zeros(Float64, 0, 0, 0)
    sum_rem_r = measure_greens ? zeros(ComplexF64, ntau, lx, ly) : zeros(ComplexF64, 0, 0, 0)
    sumsq_rem_r_re = measure_greens ? zeros(Float64, ntau, lx, ly) : zeros(Float64, 0, 0, 0)
    sumsq_rem_r_im = measure_greens ? zeros(Float64, ntau, lx, ly) : zeros(Float64, 0, 0, 0)
    sum_add_k = measure_greens ? zeros(ComplexF64, ntau, lx, ly) : zeros(ComplexF64, 0, 0, 0)
    sumsq_add_k_re = measure_greens ? zeros(Float64, ntau, lx, ly) : zeros(Float64, 0, 0, 0)
    sumsq_add_k_im = measure_greens ? zeros(Float64, ntau, lx, ly) : zeros(Float64, 0, 0, 0)
    sum_rem_k = measure_greens ? zeros(ComplexF64, ntau, lx, ly) : zeros(ComplexF64, 0, 0, 0)
    sumsq_rem_k_re = measure_greens ? zeros(Float64, ntau, lx, ly) : zeros(Float64, 0, 0, 0)
    sumsq_rem_k_im = measure_greens ? zeros(Float64, ntau, lx, ly) : zeros(Float64, 0, 0, 0)

    sum_bkt = zeros(Float64, 5)
    sumsq_bkt = zeros(Float64, 5)
    sum_equal_time = zeros(Float64, 5)
    sum_raw_equal_time = zeros(Float64, 5)
    sumsq_equal_time = zeros(Float64, 5)
    phase_sum = 0.0
    abs_phase_sum = 0.0
    corr_sampler = measure_equal_time_correlations ? CorrFuncSampler(system, qmc; nsamples=1) : nothing
    corr_deltas = measure_equal_time_correlations ? copy(corr_sampler.δr) : Tuple{Int,Int}[]
    corr_delta_to_index = Dict{Tuple{Int,Int},Int}(d => i for (i, d) in enumerate(corr_deltas))
    q_reps = measure_equal_time_correlations ? d4_momentum_reps(lx, ly) : Tuple{Int,Int}[]
    shells = measure_equal_time_correlations ? neighbor_shell_definitions(lx, ly; nshells=4) : Tuple{Int,Int,Vector{Tuple{Int,Int}}}[]
    density_fixed = (params["nup"] + params["ndn"]) / (lx * ly)
    sum_charge_wedge = zeros(Float64, length(corr_deltas))
    sum_raw_charge_wedge = zeros(Float64, length(corr_deltas))
    sumsq_charge_wedge = zeros(Float64, length(corr_deltas))
    sum_spin_wedge = zeros(Float64, length(corr_deltas))
    sum_raw_spin_wedge = zeros(Float64, length(corr_deltas))
    sumsq_spin_wedge = zeros(Float64, length(corr_deltas))
    sum_charge_structure_raw = zeros(Float64, length(q_reps))
    sum_raw_charge_structure_raw = zeros(Float64, length(q_reps))
    sumsq_charge_structure_raw = zeros(Float64, length(q_reps))
    sum_charge_structure_connected = zeros(Float64, length(q_reps))
    sum_raw_charge_structure_connected = zeros(Float64, length(q_reps))
    sumsq_charge_structure_connected = zeros(Float64, length(q_reps))
    sum_spin_structure = zeros(Float64, length(q_reps))
    sum_raw_spin_structure = zeros(Float64, length(q_reps))
    sumsq_spin_structure = zeros(Float64, length(q_reps))
    sum_charge_shell_raw = zeros(Float64, length(shells))
    sum_raw_charge_shell_raw = zeros(Float64, length(shells))
    sumsq_charge_shell_raw = zeros(Float64, length(shells))
    sum_charge_shell_connected = zeros(Float64, length(shells))
    sum_raw_charge_shell_connected = zeros(Float64, length(shells))
    sumsq_charge_shell_connected = zeros(Float64, length(shells))
    sum_spin_shell = zeros(Float64, length(shells))
    sum_raw_spin_shell = zeros(Float64, length(shells))
    sumsq_spin_shell = zeros(Float64, length(shells))
    metropolis_ratio_stats = MetropolisRatioStats()

    completed_batches = 0
    warmups_completed = 0
    resume_loaded = false
    if checkpoint_enabled && isfile(checkpoint_path)
        checkpoint_state = load_checkpoint(checkpoint_path)
        validate_checkpoint_state(checkpoint_state, params, system)
        completed_batches = Int(checkpoint_state["completed_batches"])
        warmups_completed = Int(get(checkpoint_state, "warmups_completed", qmc.nwarmups))
        nsamples_total = Int(checkpoint_state["nsamples_total"])
        max_batches = params["max_batches"]
        (params["checkpoint_reset_accumulators"] || completed_batches <= max_batches) ||
            error("checkpoint completed_batches=$(completed_batches) exceeds max_batches=$(max_batches)")
        warmups_completed <= qmc.nwarmups || error("checkpoint warmups_completed=$(warmups_completed) exceeds nwarmups=$(qmc.nwarmups)")

        sum_Cτ = checkpoint_state["sum_Ctau"]
        sumsq_Cτ = checkpoint_state["sumsq_Ctau"]
        sum_Rτ = checkpoint_state["sum_Rtau"]
        sumsq_Rτ = checkpoint_state["sumsq_Rtau"]

        sum_add_r = checkpoint_state["sum_add_r"]
        sumsq_add_r_re = checkpoint_state["sumsq_add_r_re"]
        sumsq_add_r_im = checkpoint_state["sumsq_add_r_im"]
        sum_rem_r = checkpoint_state["sum_rem_r"]
        sumsq_rem_r_re = checkpoint_state["sumsq_rem_r_re"]
        sumsq_rem_r_im = checkpoint_state["sumsq_rem_r_im"]
        sum_add_k = checkpoint_state["sum_add_k"]
        sumsq_add_k_re = checkpoint_state["sumsq_add_k_re"]
        sumsq_add_k_im = checkpoint_state["sumsq_add_k_im"]
        sum_rem_k = checkpoint_state["sum_rem_k"]
        sumsq_rem_k_re = checkpoint_state["sumsq_rem_k_re"]
        sumsq_rem_k_im = checkpoint_state["sumsq_rem_k_im"]

        sum_bkt = checkpoint_state["sum_bkt"]
        sumsq_bkt = checkpoint_state["sumsq_bkt"]
        sum_equal_time = checkpoint_state["sum_equal_time"]
        sum_raw_equal_time = get(checkpoint_state, "sum_raw_equal_time", phase_reweight ? sum_raw_equal_time : copy(sum_equal_time))
        sumsq_equal_time = checkpoint_state["sumsq_equal_time"]
        phase_sum = Float64(get(checkpoint_state, "phase_sum", phase_reweight ? NaN : nsamples_total))
        abs_phase_sum = Float64(get(checkpoint_state, "abs_phase_sum", phase_reweight ? NaN : nsamples_total))
        if phase_reweight && (!isfinite(phase_sum) || !isfinite(abs_phase_sum))
            error("phase-reweighted checkpoint is missing determinant-phase accumulators")
        end
        sum_charge_wedge = get(checkpoint_state, "sum_charge_wedge", sum_charge_wedge)
        sum_raw_charge_wedge = get(checkpoint_state, "sum_raw_charge_wedge", phase_reweight ? sum_raw_charge_wedge : copy(sum_charge_wedge))
        sumsq_charge_wedge = get(checkpoint_state, "sumsq_charge_wedge", sumsq_charge_wedge)
        sum_spin_wedge = get(checkpoint_state, "sum_spin_wedge", sum_spin_wedge)
        sum_raw_spin_wedge = get(checkpoint_state, "sum_raw_spin_wedge", phase_reweight ? sum_raw_spin_wedge : copy(sum_spin_wedge))
        sumsq_spin_wedge = get(checkpoint_state, "sumsq_spin_wedge", sumsq_spin_wedge)
        sum_charge_structure_raw = get(checkpoint_state, "sum_charge_structure_raw", sum_charge_structure_raw)
        sum_raw_charge_structure_raw = get(checkpoint_state, "sum_raw_charge_structure_raw", phase_reweight ? sum_raw_charge_structure_raw : copy(sum_charge_structure_raw))
        sumsq_charge_structure_raw = get(checkpoint_state, "sumsq_charge_structure_raw", sumsq_charge_structure_raw)
        sum_charge_structure_connected = get(checkpoint_state, "sum_charge_structure_connected", sum_charge_structure_connected)
        sum_raw_charge_structure_connected = get(checkpoint_state, "sum_raw_charge_structure_connected", phase_reweight ? sum_raw_charge_structure_connected : copy(sum_charge_structure_connected))
        sumsq_charge_structure_connected = get(checkpoint_state, "sumsq_charge_structure_connected", sumsq_charge_structure_connected)
        sum_spin_structure = get(checkpoint_state, "sum_spin_structure", sum_spin_structure)
        sum_raw_spin_structure = get(checkpoint_state, "sum_raw_spin_structure", phase_reweight ? sum_raw_spin_structure : copy(sum_spin_structure))
        sumsq_spin_structure = get(checkpoint_state, "sumsq_spin_structure", sumsq_spin_structure)
        sum_charge_shell_raw = get(checkpoint_state, "sum_charge_shell_raw", sum_charge_shell_raw)
        sum_raw_charge_shell_raw = get(checkpoint_state, "sum_raw_charge_shell_raw", phase_reweight ? sum_raw_charge_shell_raw : copy(sum_charge_shell_raw))
        sumsq_charge_shell_raw = get(checkpoint_state, "sumsq_charge_shell_raw", sumsq_charge_shell_raw)
        sum_charge_shell_connected = get(checkpoint_state, "sum_charge_shell_connected", sum_charge_shell_connected)
        sum_raw_charge_shell_connected = get(checkpoint_state, "sum_raw_charge_shell_connected", phase_reweight ? sum_raw_charge_shell_connected : copy(sum_charge_shell_connected))
        sumsq_charge_shell_connected = get(checkpoint_state, "sumsq_charge_shell_connected", sumsq_charge_shell_connected)
        sum_spin_shell = get(checkpoint_state, "sum_spin_shell", sum_spin_shell)
        sum_raw_spin_shell = get(checkpoint_state, "sum_raw_spin_shell", phase_reweight ? sum_raw_spin_shell : copy(sum_spin_shell))
        sumsq_spin_shell = get(checkpoint_state, "sumsq_spin_shell", sumsq_spin_shell)
        metropolis_ratio_stats = get(checkpoint_state, "metropolis_ratio_stats", metropolis_ratio_stats)

        walker = checkpoint_state["walker"]
        restore_default_rng!(checkpoint_state["rng"])
        if params["checkpoint_reset_accumulators"]
            loaded_batches = completed_batches
            loaded_nsamples = nsamples_total
            completed_batches = 0
            nsamples_total = 0
            for arr in (
                sum_Cτ, sumsq_Cτ, sum_Rτ, sumsq_Rτ,
                sum_add_r, sumsq_add_r_re, sumsq_add_r_im,
                sum_rem_r, sumsq_rem_r_re, sumsq_rem_r_im,
                sum_add_k, sumsq_add_k_re, sumsq_add_k_im,
                sum_rem_k, sumsq_rem_k_re, sumsq_rem_k_im,
                sum_bkt, sumsq_bkt,
                sum_equal_time, sum_raw_equal_time, sumsq_equal_time,
                sum_charge_wedge, sum_raw_charge_wedge, sumsq_charge_wedge,
                sum_spin_wedge, sum_raw_spin_wedge, sumsq_spin_wedge,
                sum_charge_structure_raw, sum_raw_charge_structure_raw, sumsq_charge_structure_raw,
                sum_charge_structure_connected, sum_raw_charge_structure_connected, sumsq_charge_structure_connected,
                sum_spin_structure, sum_raw_spin_structure, sumsq_spin_structure,
                sum_charge_shell_raw, sum_raw_charge_shell_raw, sumsq_charge_shell_raw,
                sum_charge_shell_connected, sum_raw_charge_shell_connected, sumsq_charge_shell_connected,
                sum_spin_shell, sum_raw_spin_shell, sumsq_spin_shell,
            )
                fill!(arr, zero(eltype(arr)))
            end
            phase_sum = 0.0
            abs_phase_sum = 0.0
            metropolis_ratio_stats = MetropolisRatioStats()
            println("checkpoint_accumulators_reset path=$(checkpoint_path) loaded_completed_batches=$(loaded_batches) loaded_nsamples=$(loaded_nsamples) warmups_completed=$(warmups_completed)")
            flush(stdout)
        end
        resume_loaded = true
        println("checkpoint_loaded path=$(checkpoint_path) warmups_completed=$(warmups_completed) completed_batches=$(completed_batches) nsamples=$(nsamples_total)")
        flush(stdout)
    else
        Random.seed!(params["seed"])
        walker = Walker(system, qmc)
    end

    ρup = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    ρdn = DensityMatrix(system, Nft=qmc.num_FourierPoints)

    function make_checkpoint_state(done_batches::Int)
        return Dict(
            "version" => CHECKPOINT_VERSION,
            "created_unix_time" => time(),
            "core_params" => checkpoint_core_params(params),
            "time_slices" => system.L,
            "volume" => system.V,
            "warmups_completed" => warmups_completed,
            "completed_batches" => done_batches,
            "nsamples_total" => nsamples_total,
            "phase_reweight" => phase_reweight,
            "phase_sum" => phase_sum,
            "abs_phase_sum" => abs_phase_sum,
            "sum_Ctau" => sum_Cτ,
            "sumsq_Ctau" => sumsq_Cτ,
            "sum_Rtau" => sum_Rτ,
            "sumsq_Rtau" => sumsq_Rτ,
            "sum_add_r" => sum_add_r,
            "sumsq_add_r_re" => sumsq_add_r_re,
            "sumsq_add_r_im" => sumsq_add_r_im,
            "sum_rem_r" => sum_rem_r,
            "sumsq_rem_r_re" => sumsq_rem_r_re,
            "sumsq_rem_r_im" => sumsq_rem_r_im,
            "sum_add_k" => sum_add_k,
            "sumsq_add_k_re" => sumsq_add_k_re,
            "sumsq_add_k_im" => sumsq_add_k_im,
            "sum_rem_k" => sum_rem_k,
            "sumsq_rem_k_re" => sumsq_rem_k_re,
            "sumsq_rem_k_im" => sumsq_rem_k_im,
            "sum_bkt" => sum_bkt,
            "sumsq_bkt" => sumsq_bkt,
            "sum_equal_time" => sum_equal_time,
            "sum_raw_equal_time" => sum_raw_equal_time,
            "sumsq_equal_time" => sumsq_equal_time,
            "sum_charge_wedge" => sum_charge_wedge,
            "sum_raw_charge_wedge" => sum_raw_charge_wedge,
            "sumsq_charge_wedge" => sumsq_charge_wedge,
            "sum_spin_wedge" => sum_spin_wedge,
            "sum_raw_spin_wedge" => sum_raw_spin_wedge,
            "sumsq_spin_wedge" => sumsq_spin_wedge,
            "sum_charge_structure_raw" => sum_charge_structure_raw,
            "sum_raw_charge_structure_raw" => sum_raw_charge_structure_raw,
            "sumsq_charge_structure_raw" => sumsq_charge_structure_raw,
            "sum_charge_structure_connected" => sum_charge_structure_connected,
            "sum_raw_charge_structure_connected" => sum_raw_charge_structure_connected,
            "sumsq_charge_structure_connected" => sumsq_charge_structure_connected,
            "sum_spin_structure" => sum_spin_structure,
            "sum_raw_spin_structure" => sum_raw_spin_structure,
            "sumsq_spin_structure" => sumsq_spin_structure,
            "sum_charge_shell_raw" => sum_charge_shell_raw,
            "sum_raw_charge_shell_raw" => sum_raw_charge_shell_raw,
            "sumsq_charge_shell_raw" => sumsq_charge_shell_raw,
            "sum_charge_shell_connected" => sum_charge_shell_connected,
            "sum_raw_charge_shell_connected" => sum_raw_charge_shell_connected,
            "sumsq_charge_shell_connected" => sumsq_charge_shell_connected,
            "sum_spin_shell" => sum_spin_shell,
            "sum_raw_spin_shell" => sum_raw_spin_shell,
            "sumsq_spin_shell" => sumsq_spin_shell,
            "metropolis_ratio_stats" => metropolis_ratio_stats,
            "corr_deltas" => corr_deltas,
            "q_reps" => q_reps,
            "neighbor_shells" => shells,
            "walker" => walker,
            "rng" => copy(Random.default_rng()),
        )
    end

    function write_current_checkpoint(done_batches::Int, reason::AbstractString)
        save_checkpoint(checkpoint_path, make_checkpoint_state(done_batches))
        save_checkpoint_status(checkpoint_path, done_batches, warmups_completed, nsamples_total, reason)
        println("checkpoint_written path=$(checkpoint_path) warmups_completed=$(warmups_completed) completed_batches=$(done_batches) nsamples=$(nsamples_total) reason=$(reason)")
        flush(stdout)
        return time()
    end

    last_checkpoint_time = time()
    if checkpoint_enabled && !resume_loaded
        last_checkpoint_time = write_current_checkpoint(completed_batches, "initial")
    end

    while warmups_completed < qmc.nwarmups
        chunk = min(params["checkpoint_warmup_chunk"], qmc.nwarmups - warmups_completed)
        sweep!(system, qmc, walker, loop_number=chunk)
        params["diagnose_metropolis"] && update_metropolis_ratio_stats!(metropolis_ratio_stats, walker)
        warmups_completed += chunk
        println("warmup_progress warmups_completed=$(warmups_completed)/$(qmc.nwarmups)")
        flush(stdout)

        if checkpoint_due(params, last_checkpoint_time, completed_batches)
            last_checkpoint_time = write_current_checkpoint(completed_batches, "warmup_periodic")
        end
        if runtime_limit_reached(params, run_start_time)
            if params["diagnose_metropolis"]
                write_metropolis_ratio_diagnostics(outdir, metropolis_ratio_stats, nsamples_total, completed_batches, warmups_completed)
            end
            last_checkpoint_time = write_current_checkpoint(completed_batches, "runtime_limit_warmup")
            wait_for_peer_checkpoints(params, completed_batches, warmups_completed)
            println("checkpoint_runtime_stop warmups_completed=$(warmups_completed) completed_batches=$(completed_batches) nsamples=$(nsamples_total) elapsed_hours=$((time() - run_start_time) / 3600.0)")
            flush(stdout)
            exit(params["checkpoint_exit_code"])
        end
    end

    if checkpoint_enabled && (warmups_completed == qmc.nwarmups) && (nsamples_total == 0) && (completed_batches == 0)
        last_checkpoint_time = write_current_checkpoint(completed_batches, "post_warmup")
    end

    for batch in (completed_batches + 1):params["max_batches"]
        for sample in 1:qmc.nsamples
            sweep!(system, qmc, walker, loop_number=qmc.measure_interval)
            params["diagnose_metropolis"] && update_metropolis_ratio_stats!(metropolis_ratio_stats, walker)
            ce_update_density_matrices!(system, walker, ρup, ρdn)
            sample_phase = phase_reweight ? canonical_configuration_phase(walker) : 1.0
            phase_sum += sample_phase
            abs_phase_sum += abs(sample_phase)
            nsamples_total += 1

            need_unequal_time = measure_greens || (measure_bkt && !structured_bkt_estimator)
            Bup = Bdn = nothing
            prefix_up = suffix_up = prefix_dn = suffix_dn = nothing
            if need_unequal_time
                Bup, Bdn = build_B_slices(system, walker)
                prefix_up, suffix_up = build_prefix_suffix(Bup)
                prefix_dn, suffix_dn = build_prefix_suffix(Bdn)
            end

            bkt_vals = nothing
            if measure_bkt
                bkt_vals = collect(measure_bkt_observables(
                    system, ρup, ρdn, Bup, Bdn, prefix_up, suffix_up, prefix_dn, suffix_dn,
                    estimator=params["bkt_current_estimator"],
                    refresh_interval=params["bkt_refresh_interval"],
                    adaptive_refresh=params["bkt_adaptive_refresh"],
                    refresh_tol=params["bkt_refresh_tol"],
                    refresh_min=params["bkt_refresh_min"],
                    refresh_max=params["bkt_refresh_max"],
                    refresh_growth_patience=params["bkt_refresh_growth_patience"],
                    current_drift_tol=params["bkt_current_drift_tol"],
                    allow_imbalanced_canonical_recursion=params["allow_imbalanced_canonical_recursion"],
                    walker=walker,
                ))
                sum_bkt .+= bkt_vals
                sumsq_bkt .+= bkt_vals .^ 2
            end

            if measure_equal_time
                equal_vals = collect(measure_equal_time_observables(
                    system, ρup, ρdn, kx_value=(bkt_vals === nothing ? nothing : bkt_vals[3])
                ))
                sum_equal_time .+= sample_phase .* equal_vals
                sum_raw_equal_time .+= equal_vals
                sumsq_equal_time .+= equal_vals .^ 2
                if measure_equal_time_correlations
                    charge_wedge, spin_wedge = measure_equal_time_charge_spin_wedge!(corr_sampler, ρup, ρdn)
                    sum_charge_wedge .+= sample_phase .* charge_wedge
                    sum_raw_charge_wedge .+= charge_wedge
                    sumsq_charge_wedge .+= charge_wedge .^ 2
                    sum_spin_wedge .+= sample_phase .* spin_wedge
                    sum_raw_spin_wedge .+= spin_wedge
                    sumsq_spin_wedge .+= spin_wedge .^ 2

                    charge_struct_raw, charge_struct_connected, spin_struct = structure_samples_from_wedge(
                        charge_wedge, spin_wedge, corr_delta_to_index, q_reps, lx, ly, density_fixed,
                    )
                    sum_charge_structure_raw .+= sample_phase .* charge_struct_raw
                    sum_raw_charge_structure_raw .+= charge_struct_raw
                    sumsq_charge_structure_raw .+= charge_struct_raw .^ 2
                    sum_charge_structure_connected .+= sample_phase .* charge_struct_connected
                    sum_raw_charge_structure_connected .+= charge_struct_connected
                    sumsq_charge_structure_connected .+= charge_struct_connected .^ 2
                    sum_spin_structure .+= sample_phase .* spin_struct
                    sum_raw_spin_structure .+= spin_struct
                    sumsq_spin_structure .+= spin_struct .^ 2

                    charge_shell_raw, charge_shell_connected, spin_shell = shell_samples_from_wedge(
                        charge_wedge, spin_wedge, corr_delta_to_index, shells, lx, ly, density_fixed,
                    )
                    sum_charge_shell_raw .+= sample_phase .* charge_shell_raw
                    sum_raw_charge_shell_raw .+= charge_shell_raw
                    sumsq_charge_shell_raw .+= charge_shell_raw .^ 2
                    sum_charge_shell_connected .+= sample_phase .* charge_shell_connected
                    sum_raw_charge_shell_connected .+= charge_shell_connected
                    sumsq_charge_shell_connected .+= charge_shell_connected .^ 2
                    sum_spin_shell .+= sample_phase .* spin_shell
                    sum_raw_spin_shell .+= spin_shell
                    sumsq_spin_shell .+= spin_shell .^ 2
                end
            end

            if measure_greens
                cache = CanonicalUnequalTimeGreenCache(system, prefix_up, 1)
                for slice in 0:(system.L - 1)
                    add_mat, rem_mat = canonical_unequal_time_greens(
                        system, cache, prefix_up, suffix_up, slice,
                        physical_normalization=true, spin=1,
                    )
                    add_r = green_realspace_average(system, add_mat)
                    rem_r = green_realspace_average(system, rem_mat)
                    add_k = green_momentum_from_realspace(add_r)
                    rem_k = green_momentum_from_realspace(rem_r)

                    cτ = real(tr(add_mat) / system.V)
                    rτ = real(tr(rem_mat) / system.V)
                    idx = slice + 1
                    sum_Cτ[idx] += cτ
                    sumsq_Cτ[idx] += cτ^2
                    sum_Rτ[idx] += rτ
                    sumsq_Rτ[idx] += rτ^2
                    add_complex_sample!(@view(sum_add_r[idx, :, :]), @view(sumsq_add_r_re[idx, :, :]), @view(sumsq_add_r_im[idx, :, :]), add_r)
                    add_complex_sample!(@view(sum_rem_r[idx, :, :]), @view(sumsq_rem_r_re[idx, :, :]), @view(sumsq_rem_r_im[idx, :, :]), rem_r)
                    add_complex_sample!(@view(sum_add_k[idx, :, :]), @view(sumsq_add_k_re[idx, :, :]), @view(sumsq_add_k_im[idx, :, :]), add_k)
                    add_complex_sample!(@view(sum_rem_k[idx, :, :]), @view(sumsq_rem_k_re[idx, :, :]), @view(sumsq_rem_k_im[idx, :, :]), rem_k)
                end
            end
        end

        status_parts = String["batch=$(batch)", "nsamples=$(nsamples_total)"]

        if measure_greens
            Cτ_mean = sum_Cτ ./ nsamples_total
            Cτ_err = stderr_from_sums(sum_Cτ, sumsq_Cτ, nsamples_total)
            Rτ_mean = sum_Rτ ./ nsamples_total
            Rτ_err = stderr_from_sums(sum_Rτ, sumsq_Rτ, nsamples_total)
            add_r_mean, add_r_re_err, add_r_im_err = complex_mean_reim_err_from_sums(sum_add_r, sumsq_add_r_re, sumsq_add_r_im, nsamples_total)
            rem_r_mean, rem_r_re_err, rem_r_im_err = complex_mean_reim_err_from_sums(sum_rem_r, sumsq_rem_r_re, sumsq_rem_r_im, nsamples_total)
            add_k_mean, add_k_re_err, add_k_im_err = complex_mean_reim_err_from_sums(sum_add_k, sumsq_add_k_re, sumsq_add_k_im, nsamples_total)
            rem_k_mean, rem_k_re_err, rem_k_im_err = complex_mean_reim_err_from_sums(sum_rem_k, sumsq_rem_k_re, sumsq_rem_k_im, nsamples_total)

            write_tau_summary(outdir, params, Cτ_mean, Cτ_err, nsamples_total, batch)
            write_tau_remove_summary(outdir, params, Rτ_mean, Rτ_err, nsamples_total, batch)
            write_realspace_tau_summary(outdir, "greens_r_tau_add_qmc.tsv", params, add_r_mean, add_r_re_err, add_r_im_err, nsamples_total, batch)
            write_realspace_tau_summary(outdir, "greens_r_tau_remove_qmc.tsv", params, rem_r_mean, rem_r_re_err, rem_r_im_err, nsamples_total, batch)
            write_momentum_tau_summary(outdir, "greens_k_tau_add_qmc.tsv", params, add_k_mean, add_k_re_err, add_k_im_err, nsamples_total, batch)
            write_momentum_tau_summary(outdir, "greens_k_tau_remove_qmc.tsv", params, rem_k_mean, rem_k_re_err, rem_k_im_err, nsamples_total, batch)
            push!(status_parts, string("Gadd(k=0,tau=0)=", add_k_mean[1, 1, 1]))
        end

        if measure_equal_time
            average_phase = phase_sum / nsamples_total
            equal_mean, equal_err = ratio_mean_err_from_sums(
                sum_equal_time, sum_raw_equal_time, sumsq_equal_time, phase_sum, nsamples_total,
            )
            write_equal_time_summary(
                outdir, params, equal_mean, equal_err, nsamples_total, batch, system.L;
                phase_sum=phase_sum,
                average_phase=average_phase,
                phase_reweighted=phase_reweight,
                signed_sums=sum_equal_time,
            )
            if measure_equal_time_correlations
                charge_wedge_mean, charge_wedge_err = ratio_mean_err_from_sums(
                    sum_charge_wedge, sum_raw_charge_wedge, sumsq_charge_wedge, phase_sum, nsamples_total,
                )
                spin_wedge_mean, spin_wedge_err = ratio_mean_err_from_sums(
                    sum_spin_wedge, sum_raw_spin_wedge, sumsq_spin_wedge, phase_sum, nsamples_total,
                )
                write_equal_time_wedge_summary(
                    outdir, params, corr_deltas,
                    charge_wedge_mean, charge_wedge_err,
                    spin_wedge_mean, spin_wedge_err,
                    nsamples_total, batch,
                    phase_sum, average_phase, phase_reweight,
                    sum_charge_wedge, sum_spin_wedge,
                )

                charge_struct_raw_mean, charge_struct_raw_err = ratio_mean_err_from_sums(
                    sum_charge_structure_raw, sum_raw_charge_structure_raw, sumsq_charge_structure_raw, phase_sum, nsamples_total,
                )
                charge_struct_conn_mean, charge_struct_conn_err = ratio_mean_err_from_sums(
                    sum_charge_structure_connected, sum_raw_charge_structure_connected, sumsq_charge_structure_connected, phase_sum, nsamples_total,
                )
                spin_struct_mean, spin_struct_err = ratio_mean_err_from_sums(
                    sum_spin_structure, sum_raw_spin_structure, sumsq_spin_structure, phase_sum, nsamples_total,
                )
                write_equal_time_structure_summary(
                    outdir, params, q_reps,
                    charge_struct_raw_mean, charge_struct_raw_err,
                    charge_struct_conn_mean, charge_struct_conn_err,
                    spin_struct_mean, spin_struct_err,
                    nsamples_total, batch,
                    phase_sum, average_phase, phase_reweight,
                    sum_charge_structure_raw, sum_charge_structure_connected, sum_spin_structure,
                )

                charge_shell_raw_mean, charge_shell_raw_err = ratio_mean_err_from_sums(
                    sum_charge_shell_raw, sum_raw_charge_shell_raw, sumsq_charge_shell_raw, phase_sum, nsamples_total,
                )
                charge_shell_conn_mean, charge_shell_conn_err = ratio_mean_err_from_sums(
                    sum_charge_shell_connected, sum_raw_charge_shell_connected, sumsq_charge_shell_connected, phase_sum, nsamples_total,
                )
                spin_shell_mean, spin_shell_err = ratio_mean_err_from_sums(
                    sum_spin_shell, sum_raw_spin_shell, sumsq_spin_shell, phase_sum, nsamples_total,
                )
                write_equal_time_shell_summary(
                    outdir, shells,
                    charge_shell_raw_mean, charge_shell_raw_err,
                    charge_shell_conn_mean, charge_shell_conn_err,
                    spin_shell_mean, spin_shell_err,
                    nsamples_total, batch,
                    phase_sum, average_phase, phase_reweight,
                    sum_charge_shell_raw, sum_charge_shell_connected, sum_spin_shell,
                )
                push!(status_parts, string("Sspin(qpi,pi)≈", isempty(spin_struct_mean) ? "NA" : maximum(spin_struct_mean)))
            end
            push!(status_parts, string("total/site=", equal_mean[3]))
            push!(status_parts, string("docc/site=", equal_mean[4]))
            push!(status_parts, string("average_phase=", average_phase))
        end

        if measure_bkt
            bkt_mean = sum_bkt ./ nsamples_total
            bkt_err = stderr_from_sums(sum_bkt, sumsq_bkt, nsamples_total)
            write_bkt_summary(outdir, params, bkt_mean, bkt_err, nsamples_total, batch, system.L)
            push!(status_parts, string("rho_s_current=", bkt_mean[4]))
            push!(status_parts, string("rho_s_dia=", bkt_mean[5]))
        end
        if params["diagnose_metropolis"]
            write_metropolis_ratio_diagnostics(outdir, metropolis_ratio_stats, nsamples_total, batch, warmups_completed)
            finite = metropolis_ratio_stats.n_finite_nonnegative
            if finite > 0
                push!(status_parts, string("metro_accept≈", metropolis_ratio_stats.sum_accept_prob / finite))
            end
        end
        println(join(status_parts, " "))
        flush(stdout)

        completed_batches = batch
        if checkpoint_due(params, last_checkpoint_time, completed_batches)
            last_checkpoint_time = write_current_checkpoint(completed_batches, "periodic")
        end
        if completed_batches < params["max_batches"] && runtime_limit_reached(params, run_start_time)
            if params["diagnose_metropolis"]
                write_metropolis_ratio_diagnostics(outdir, metropolis_ratio_stats, nsamples_total, completed_batches, warmups_completed)
            end
            last_checkpoint_time = write_current_checkpoint(completed_batches, "runtime_limit")
            wait_for_peer_checkpoints(params, completed_batches, warmups_completed)
            println("checkpoint_runtime_stop completed_batches=$(completed_batches) nsamples=$(nsamples_total) elapsed_hours=$((time() - run_start_time) / 3600.0)")
            flush(stdout)
            exit(params["checkpoint_exit_code"])
        end
    end

    if checkpoint_enabled
        complete_path = joinpath(outdir, "checkpoint_complete.txt")
        open(complete_path, "w") do io
            println(io, "completed_batches=$(completed_batches)")
            println(io, "nsamples=$(nsamples_total)")
            println(io, "completed_unix_time=$(time())")
        end
        if !params["checkpoint_keep"]
            if isfile(checkpoint_path)
                rm(checkpoint_path; force=true)
            end
            status_path = checkpoint_status_path(checkpoint_path)
            if isfile(status_path)
                rm(status_path; force=true)
            end
            println("checkpoint_removed path=$(checkpoint_path)")
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
