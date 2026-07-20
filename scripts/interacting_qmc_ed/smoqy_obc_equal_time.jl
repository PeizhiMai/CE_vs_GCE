module SmoQyOBCEqualTime

using MPI
using Printf
using Statistics
using SmoQyDQMC

using ..SquareLatticeGeometry

export OBCEqualTimeAccumulator,
       configuration_phase,
       record_obc_equal_time!,
       write_obc_rank_accumulator,
       write_obc_pooled_outputs,
       smoqy_hopping_matrix,
       smoqy_provenance

const ACCUMULATOR_VERSION = 1
const VALUE_NAMES = (
    "kinetic_per_site",
    "double_occupancy_per_site",
    "nn_spin_s_s",
    "nn_charge_raw",
    "nn_charge_connected_instantaneous",
    "nnn_spin_s_s",
    "nnn_charge_raw",
    "nnn_charge_connected_instantaneous",
    "density",
)

mutable struct OBCEqualTimeAccumulator
    version::Int
    nsites::Int
    nsamples::Int
    phase_sum::ComplexF64
    abs_phase_sum::Float64
    signed_sum::Vector{ComplexF64}
    raw_sum::Vector{Float64}
    raw_sumsq::Vector{Float64}
    signed_site_density_sum::Vector{ComplexF64}
    raw_site_density_sum::Vector{Float64}
end

function OBCEqualTimeAccumulator(nsites::Int)
    return OBCEqualTimeAccumulator(
        ACCUMULATOR_VERSION,
        nsites,
        0,
        0.0 + 0.0im,
        0.0,
        zeros(ComplexF64, length(VALUE_NAMES)),
        zeros(Float64, length(VALUE_NAMES)),
        zeros(Float64, length(VALUE_NAMES)),
        zeros(ComplexF64, nsites),
        zeros(Float64, nsites),
    )
end

function configuration_phase(
    fermion_path_integral_up,
    sgndetGup,
    sgndetGdn,
)
    Sb = fermion_path_integral_up.Sb
    weight_phase = isreal(Sb) ?
        inv(sgndetGup) * inv(sgndetGdn) :
        exp(-1im * imag(Sb)) * inv(sgndetGup) * inv(sgndetGdn)
    return ComplexF64(sign(weight_phase))
end

function record_obc_equal_time!(
    accumulator::OBCEqualTimeAccumulator,
    geometry::SquareGeometry,
    Gup,
    Gdn,
    phase;
    U::Real,
)
    accumulator.version == ACCUMULATOR_VERSION || error("OBC accumulator version mismatch")
    accumulator.nsites == geometry.nsites || error("OBC accumulator geometry mismatch")
    rho_up = density_matrix_from_smoqy_green(Gup)
    rho_dn = density_matrix_from_smoqy_green(Gdn)
    obs = measure_equal_time_observables(geometry, rho_up, rho_dn; U=U)
    values = Float64[
        obs.kinetic_per_site,
        obs.double_occupancy_per_site,
        obs.nn_spin,
        obs.nn_charge_raw,
        obs.nn_charge_connected,
        obs.nnn_spin,
        obs.nnn_charge_raw,
        obs.nnn_charge_connected,
        obs.density,
    ]
    site_density = obs.density_up_site .+ obs.density_dn_site
    p = ComplexF64(phase)
    accumulator.nsamples += 1
    accumulator.phase_sum += p
    accumulator.abs_phase_sum += abs(p)
    accumulator.signed_sum .+= p .* values
    accumulator.raw_sum .+= values
    accumulator.raw_sumsq .+= values .^ 2
    accumulator.signed_site_density_sum .+= p .* site_density
    accumulator.raw_site_density_sum .+= site_density
    return obs
end

function _git_commit_for_source(path::AbstractString)
    root = dirname(dirname(path))
    try
        return readchomp(pipeline(`git -C $root rev-parse HEAD`; stderr=devnull))
    catch
        return get(ENV, "SMOQYDQMC_GIT_COMMIT", "unknown")
    end
end

function smoqy_provenance()
    return (
        upstream_version=string(SmoQyDQMC.SMOQYDQMC_VERSION),
        fork_commit=_git_commit_for_source(pathof(SmoQyDQMC)),
        fork_branch="obc-v2.0.12",
    )
end

"""Reconstruct the one-body Hamiltonian from SmoQyDQMC tight-binding parameters."""
function smoqy_hopping_matrix(parameters)
    n = length(parameters.ϵ)
    K = zeros(ComplexF64, n, n)
    for edge in axes(parameters.neighbor_table, 2)
        i, j = parameters.neighbor_table[:, edge]
        t = parameters.t[edge]
        K[i, j] -= t
        K[j, i] -= conj(t)
    end
    return K
end

function _pool(acc::OBCEqualTimeAccumulator, comm::MPI.Comm)
    nsamples = [acc.nsamples]
    MPI.Allreduce!(nsamples, MPI.SUM, comm)
    phase_sum = [acc.phase_sum]
    MPI.Allreduce!(phase_sum, MPI.SUM, comm)
    abs_phase_sum = [acc.abs_phase_sum]
    MPI.Allreduce!(abs_phase_sum, MPI.SUM, comm)
    signed_sum = copy(acc.signed_sum)
    raw_sum = copy(acc.raw_sum)
    raw_sumsq = copy(acc.raw_sumsq)
    site_signed = copy(acc.signed_site_density_sum)
    site_raw = copy(acc.raw_site_density_sum)
    MPI.Allreduce!(signed_sum, MPI.SUM, comm)
    MPI.Allreduce!(raw_sum, MPI.SUM, comm)
    MPI.Allreduce!(raw_sumsq, MPI.SUM, comm)
    MPI.Allreduce!(site_signed, MPI.SUM, comm)
    MPI.Allreduce!(site_raw, MPI.SUM, comm)
    return (
        nsamples=nsamples[1], phase_sum=phase_sum[1], abs_phase_sum=abs_phase_sum[1],
        signed_sum=signed_sum, raw_sum=raw_sum, raw_sumsq=raw_sumsq,
        signed_site_density_sum=site_signed, raw_site_density_sum=site_raw,
        nranks=MPI.Comm_size(comm),
    )
end

function _ratio_stats(pooled)
    n = pooled.nsamples
    n > 0 || error("cannot summarize an empty OBC accumulator")
    abs(pooled.phase_sum) > 100eps(Float64) * max(pooled.abs_phase_sum, 1.0) ||
        error("numerical-zero phase denominator in OBC equal-time pooling")
    means = real.(pooled.signed_sum ./ pooled.phase_sum)
    if n <= 1
        errors = fill(NaN, length(means))
    else
        raw_mean = pooled.raw_sum ./ n
        raw_var = max.(pooled.raw_sumsq .- n .* raw_mean .^ 2, 0.0) ./ (n - 1)
        # This is the raw-observable SEM. Rank-level/seed-level benchmark code
        # performs the final ratio jackknife used for acceptance tests.
        errors = sqrt.(raw_var ./ n)
    end
    site_mean = real.(pooled.signed_site_density_sum ./ pooled.phase_sum)
    return means, errors, site_mean
end

function _connected(raw, bonds, site_mean)
    isempty(bonds) && return NaN
    return raw - sum(site_mean[i] * site_mean[j] for (i, j) in bonds) / length(bonds)
end

function _rank_jackknife_errors(
    accumulator::OBCEqualTimeAccumulator,
    geometry::SquareGeometry,
    comm::MPI.Comm,
)
    nranks = MPI.Comm_size(comm)
    nranks >= 2 || return nothing
    width = 1 + length(VALUE_NAMES) + geometry.nsites
    local_values = vcat(
        ComplexF64[accumulator.phase_sum],
        accumulator.signed_sum,
        accumulator.signed_site_density_sum,
    )
    gathered = MPI.Allgather(local_values, comm)
    length(gathered) == width * nranks || error("OBC rank-jackknife gather size mismatch")
    by_rank = reshape(gathered, width, nranks)
    total = vec(sum(by_rank, dims=2))
    leave_values = Matrix{Float64}(undef, length(VALUE_NAMES), nranks)
    leave_nn_connected = Vector{Float64}(undef, nranks)
    leave_nnn_connected = Vector{Float64}(undef, nranks)
    for rank in 1:nranks
        leave = total .- by_rank[:, rank]
        denominator = leave[1]
        abs(denominator) > 100eps(Float64) || error(
            "numerical-zero leave-one-rank phase denominator in OBC pooling",
        )
        values = real.(leave[2:1 + length(VALUE_NAMES)] ./ denominator)
        site_mean = real.(leave[2 + length(VALUE_NAMES):end] ./ denominator)
        leave_values[:, rank] .= values
        leave_nn_connected[rank] = _connected(values[4], geometry.nn_bonds, site_mean)
        leave_nnn_connected[rank] = _connected(values[7], geometry.nnn_bonds, site_mean)
    end
    function jackknife(values)
        center = mean(values)
        return sqrt((nranks - 1) / nranks * sum(abs2, values .- center))
    end
    return (
        value_errors=[jackknife(view(leave_values, i, :)) for i in axes(leave_values, 1)],
        nn_connected_error=jackknife(leave_nn_connected),
        nnn_connected_error=jackknife(leave_nnn_connected),
    )
end

function write_obc_rank_accumulator(
    datafolder::AbstractString,
    accumulator::OBCEqualTimeAccumulator,
    pID::Int,
    geometry::SquareGeometry,
)
    accumulator.nsites == geometry.nsites || error("rank accumulator geometry mismatch")
    path = joinpath(datafolder, @sprintf("obc_equal_time_rank_pID-%d.tsv", pID))
    open(path, "w") do io
        println(io, "name\tnsamples\tphase_sum_real\tphase_sum_imag\tabs_phase_sum\tsigned_sum_real\tsigned_sum_imag\traw_sum\traw_sumsq")
        for i in eachindex(VALUE_NAMES)
            println(io, join((
                VALUE_NAMES[i], accumulator.nsamples,
                real(accumulator.phase_sum), imag(accumulator.phase_sum), accumulator.abs_phase_sum,
                real(accumulator.signed_sum[i]), imag(accumulator.signed_sum[i]),
                accumulator.raw_sum[i], accumulator.raw_sumsq[i],
            ), '\t'))
        end
    end
    site_path = joinpath(datafolder, @sprintf("obc_equal_time_site_rank_pID-%d.tsv", pID))
    open(site_path, "w") do io
        println(io, "site\tx\ty\tnsamples\tphase_sum_real\tphase_sum_imag\tabs_phase_sum\tdensity_signed_sum_real\tdensity_signed_sum_imag\tdensity_raw_sum")
        for site in 1:geometry.nsites
            x, y = geometry.coordinates[site]
            println(io, join((
                site, x, y, accumulator.nsamples,
                real(accumulator.phase_sum), imag(accumulator.phase_sum),
                accumulator.abs_phase_sum,
                real(accumulator.signed_site_density_sum[site]),
                imag(accumulator.signed_site_density_sum[site]),
                accumulator.raw_site_density_sum[site],
            ), '\t'))
        end
    end
    return path
end

function _write_primary(path, boundary, beta, observable, value, stderr, pooled, provenance)
    open(path, "w") do io
        println(io, "boundary\tbeta\ttemperature\tobservable\tvalue\tstderr\tnsamples\tnranks\tphase_sum_real\tphase_sum_imag\taverage_phase_abs\tsmoqydqmc_version\tsmoqydqmc_commit")
        println(io, join((
            boundary, beta, 1 / beta, observable, value, stderr, pooled.nsamples, pooled.nranks,
            real(pooled.phase_sum), imag(pooled.phase_sum), abs(pooled.phase_sum) / pooled.nsamples,
            provenance.upstream_version, provenance.fork_commit,
        ), '\t'))
    end
end

function write_obc_pooled_outputs(
    comm::MPI.Comm,
    datafolder::AbstractString,
    accumulator::OBCEqualTimeAccumulator,
    geometry::SquareGeometry;
    beta::Real,
    U::Real,
)
    pooled = _pool(accumulator, comm)
    means, errors, site_mean = _ratio_stats(pooled)
    jackknife = _rank_jackknife_errors(accumulator, geometry, comm)
    if jackknife !== nothing
        errors = jackknife.value_errors
    end
    nn_connected = _connected(means[4], geometry.nn_bonds, site_mean)
    nnn_connected = _connected(means[7], geometry.nnn_bonds, site_mean)
    nn_connected_error = jackknife === nothing ? errors[5] : jackknife.nn_connected_error
    nnn_connected_error = jackknife === nothing ? errors[8] : jackknife.nnn_connected_error
    provenance = smoqy_provenance()
    if MPI.Comm_rank(comm) == 0
        open(joinpath(datafolder, "equal_time_bond_observables_qmc.tsv"), "w") do io
            println(io, "shell\tbond_count\tcharge_corr_raw\tcharge_corr_raw_stderr\tcharge_corr_connected\tcharge_corr_connected_stderr\tspin_corr_s_s\tspin_corr_s_s_stderr\tspin_corr_SzSz\tspin_corr_SzSz_stderr\tnsamples\tphase_sum_real\tphase_sum_imag\taverage_phase_abs\tnormalization")
            for row in (
                ("NN", length(geometry.nn_bonds), 4, nn_connected, nn_connected_error, 3),
                ("NNN", length(geometry.nnn_bonds), 7, nnn_connected, nnn_connected_error, 6),
            )
                name, count, raw_i, connected_value, connected_error, spin_i = row
                println(io, join((
                    name, count, means[raw_i], errors[raw_i], connected_value, connected_error,
                    means[spin_i], errors[spin_i], 0.25 * means[spin_i], 0.25 * errors[spin_i],
                    pooled.nsamples, real(pooled.phase_sum), imag(pooled.phase_sum),
                    abs(pooled.phase_sum) / pooled.nsamples,
                    "existing undirected physical bonds",
                ), '\t'))
            end
        end
        _write_primary(
            joinpath(datafolder, "equal_time_kinetic_per_site_qmc.tsv"), "open", beta,
            "kinetic_per_site", means[1], errors[1], pooled, provenance,
        )
        _write_primary(
            joinpath(datafolder, "equal_time_double_occupancy_per_site_qmc.tsv"), "open", beta,
            "double_occupancy_per_site", means[2], errors[2], pooled, provenance,
        )
        _write_primary(
            joinpath(datafolder, "equal_time_nn_spin_qmc.tsv"), "open", beta,
            "nn_spin_s_s", means[3], errors[3], pooled, provenance,
        )
        _write_primary(
            joinpath(datafolder, "equal_time_nn_connected_charge_qmc.tsv"), "open", beta,
            "nn_connected_charge", nn_connected, nn_connected_error, pooled, provenance,
        )
        open(joinpath(datafolder, "equal_time_observables_obc_qmc.tsv"), "w") do io
            println(io, "beta\ttemperature\tdensity\tachieved_N\tkinetic_per_site\tkinetic_stderr\tinteraction_per_site\ttotal_per_site\tdouble_occupancy_per_site\tdouble_occupancy_stderr\tlocal_moment\tnsamples")
            density = means[9]
            interaction = U * means[2]
            println(io, join((
                beta, 1 / beta, density, density * geometry.nsites,
                means[1], errors[1], interaction, means[1] + interaction,
                means[2], errors[2], density - 2 * means[2], pooled.nsamples,
            ), '\t'))
        end
    end
    MPI.Barrier(comm)
    return (means=means, errors=errors, site_density=site_mean, nn_connected=nn_connected, nnn_connected=nnn_connected)
end

end # module
