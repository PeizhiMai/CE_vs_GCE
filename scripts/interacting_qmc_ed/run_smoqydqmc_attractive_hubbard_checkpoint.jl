# # Attractive square Hubbard Model with Checkpointing
# In this tutorial we demonstrate how to introduce checkpointing to the previous
# [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial, allowing for simulations to be
# resumed if terminated prior to completion.

# ## Import Packages
# No changes need to made to this section of the code from the previous
# [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial.

# Load MKL before SmoQyDQMC/LinearAlgebra when the active environment has it.
# This preserves the production default while allowing the frozen v2.0.11
# compatibility environment to be audited without resolving its old manifest.
if lowercase(get(ENV, "SMOQY_USE_MKL", "true")) != "false" && Base.find_package("MKL") !== nothing
    try
        @eval using MKL
    catch err
        @warn "SMOQY_USE_MKL requested but MKL could not be loaded; using the default BLAS/LAPACK" exception=(err, catch_backtrace())
    end
end
using LinearAlgebra
@info "BLAS/LAPACK configuration" LinearAlgebra.BLAS.get_config()

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

using Random
using Printf
using MPI
using JLD2

include(joinpath(@__DIR__, "square_lattice_geometry.jl"))
using .SquareLatticeGeometry
include(joinpath(@__DIR__, "smoqy_obc_equal_time.jl"))
using .SmoQyOBCEqualTime

function safe_jld2_checkpoint_write(
    simulation_info::SimulationInfo;
    model_geometry,
    measurement_container,
    kwargs...
)
    datafolder = simulation_info.datafolder
    pID = simulation_info.pID
    isdir(datafolder) || mkpath(datafolder)

    checkpoint_fn = joinpath(datafolder, "checkpoint_pID-$(pID).jld2")
    checkpoint_fn_old = joinpath(datafolder, "checkpoint_old_pID-$(pID).jld2")
    tmp_fn = joinpath(datafolder, "checkpoint_tmp_pID-$(pID)_pid$(getpid())_$(round(Int, time()*1000)).jld2")
    rm(tmp_fn, force = true)

    bin_files = simulation_info.bin_files
    filtered_measurement_container = (;
        (k => v for (k, v) in pairs(measurement_container) if k != :pfft!)...
    )

    try
        jldsave(
            tmp_fn;
            bin_files = bin_files,
            model_geometry = model_geometry,
            measurement_container = filtered_measurement_container,
            kwargs...
        )
        if !isfile(tmp_fn)
            error("JLD2 checkpoint write returned without creating temporary file $(tmp_fn)")
        end
        if filesize(tmp_fn) <= 0
            error("JLD2 checkpoint temporary file is empty: $(tmp_fn)")
        end
        # Open/close the file once before publishing it.  This catches many
        # incomplete JLD2 writes before replacing the previous checkpoint.
        jldopen(tmp_fn, "r") do f
            haskey(f, "model_geometry") || error("checkpoint missing model_geometry: $(tmp_fn)")
            haskey(f, "measurement_container") || error("checkpoint missing measurement_container: $(tmp_fn)")
        end

        if isfile(checkpoint_fn)
            try
                mv(checkpoint_fn, checkpoint_fn_old, force = true)
            catch err
                # A stale existence check or external cleanup can make the old
                # checkpoint disappear between isfile() and mv().  Do not fail
                # the new verified checkpoint publication because of that.
                @warn "old checkpoint vanished before rotation; continuing with new checkpoint" pID checkpoint_fn exception=(err, catch_backtrace())
            end
        end
        try
            mv(tmp_fn, checkpoint_fn, force = true)
        catch err
            # Preserve the last known-good checkpoint if publishing the new one
            # fails after moving the old file aside.
            if isfile(checkpoint_fn_old) && !isfile(checkpoint_fn)
                mv(checkpoint_fn_old, checkpoint_fn, force = true)
            end
            rethrow(err)
        end
        rm(checkpoint_fn_old, force = true)
    catch err
        rm(tmp_fn, force = true)
        @error "safe JLD2 checkpoint write failed" pID datafolder checkpoint_fn exception=(err, catch_backtrace())
        rethrow(err)
    end

    return time()
end

function write_jld2_checkpoint_mpi_serial(
    comm::MPI.Comm,
    simulation_info::SimulationInfo;
    checkpoint_timestamp::T = 0.0,
    checkpoint_freq::T = 0.0,
    start_timestamp::T = 0.0,
    runtime_limit::T = Inf,
    error_code::Int = 13,
    force_checkpoint::Bool = false,
    model_geometry,
    measurement_container,
    kwargs...
) where {T<:AbstractFloat}

    if checkpoint_freq <= 0 && !force_checkpoint
        return checkpoint_timestamp
    end

    pID = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    local_checkpoint_timestamp = checkpoint_timestamp
    now_timestamp = time()
    elapsed_runtime = now_timestamp - start_timestamp
    runtime_stop_due = elapsed_runtime >= runtime_limit
    due_to_write = force_checkpoint || ((now_timestamp - checkpoint_timestamp) >= checkpoint_freq) || runtime_stop_due
    due_flag = [due_to_write ? 1 : 0]
    MPI.Allreduce!(due_flag, MPI.MAX, comm)
    due_to_write = due_flag[1] != 0

    # JLD2 writes from many Julia MPI ranks can fail on shared filesystems.  We
    # serialize writes and publish each rank's checkpoint through a unique temp
    # file that is verified before it replaces the previous checkpoint.  This
    # avoids the SmoQyDQMC package writer's checkpoint_new -> checkpoint rename
    # failure mode where checkpoint_new was sometimes absent after jldsave.
    #
    # Important: decide the runtime-limit stop from elapsed time in the current
    # Slurm allocation, not from the previous checkpoint timestamp plus the next
    # checkpoint interval.  After a resume, checkpoint_timestamp is from an older
    # allocation, so a `(checkpoint_timestamp + checkpoint_freq - start)` rule
    # can exit immediately when checkpoint_freq is very large or equals the
    # runtime limit.  Measurement-count checkpoints force a write but should not
    # by themselves cause a runtime-limit exit.
    if due_to_write
        for write_pID in 0:(comm_size-1)
            MPI.Barrier(comm)
            if pID == write_pID
                local_checkpoint_timestamp = safe_jld2_checkpoint_write(
                    simulation_info;
                    model_geometry = model_geometry,
                    measurement_container = measurement_container,
                    kwargs...
                )
            end
        end
        MPI.Barrier(comm)
    end

    exit_flag = [(time() - start_timestamp) >= runtime_limit ? 1 : 0]
    MPI.Allreduce!(exit_flag, MPI.MAX, comm)
    if exit_flag[1] != 0
        MPI.Barrier(comm)
        MPI.Finalize()
        exit(error_code)
    end

    return local_checkpoint_timestamp
end

# ## Specify simulation parameters
# Compared to the previous [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial, we have added
# two new keyword arguments to the `run_simulation` function:
# - `checkpoint_freq`: When going to write a new checkpoint file, only write one if more than `checkpoint_freq` hours have passed since the last checkpoint file was written.
# - `runtime_limit = Inf`: If after writing a new checkpoint file more than `runtime_limit` hours have passed since the simulation started, terminate the simulation.
# The `runtime_limit = Inf` default behavior means there is no runtime limit for the simulation.

## Top-level function to run simulation.
function run_simulation(
    comm::MPI.Comm; # MPI communicator.
    ## KEYWORD ARGUMENTS
    sID, # Simulation ID.
    U, # Hubbard interaction.
    t′, # Next-nearest-neighbor hopping amplitude.
    μ, # Chemical potential.
    ph_sym_form = true, # Use particle-hole symmetric Hubbard interaction convention.
    L, # System size in x.
    Ly = L, # System size in y.
    boundary = "periodic", # periodic or open in both spatial directions.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_measurements, # Total number of measurements.
    N_bins, # Number of times bin-averaged measurements are written to file.
    N_updates, # Number of updates between measurements.
    checkpoint_freq, # Frequency with which checkpoint files are written in hours.
    runtime_limit = Inf, # Simulation runtime limit in hours.
    Δτ = 0.05, # Discretization in imaginary time.
    n_stab = 10, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    symmetric = false, # Whether symmetric propagator definition is used.
    checkerboard = false, # Whether checkerboard approximation is used.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = joinpath(@__DIR__, "..", "..", "results", "interacting_qmc_ed", "smoqydqmc_attractive_hubbard_checkpoint"), # Filepath to where data folder will be created.
    measurement_profile = "full", # "full" production measurements, "equal-time-only" for CE/GCE equal-time comparisons, "density-only" for chemical-potential tuning, or autocorrelation probes.
    use_reflection_update = false, # Match older three-band jobs: no reflection/global update by default.
    update_stabilization_frequency = false, # If true, adapt n_stab downward when δG exceeds δG_max.
    n_stab_min = 1, # Lower bound for adaptive n_stab reductions.
    checkpoint_every_n_measurements = 0 # If >0, force a checkpoint after every N completed measurement iterations.
)

# ## Initialize simulation
# We need to make a few modifications to this portion of the code as compared to the previous tutorial
# in order for checkpointing to work. First, we record need to record the simulation start time,
# which we do by initializing a variable `start_timestamp = time()`.
# Second, we need to convert the `checkpoint_freq` and `runtime_limit` from hours to seconds.

    isfinite(U) || throw(ArgumentError("Require finite Hubbard interaction U. Got U=$(U)."))
    N_measurements % N_bins == 0 || throw(ArgumentError("N_measurements must be divisible by N_bins"))
    1 <= n_stab_min <= n_stab || throw(ArgumentError("Require 1 <= n_stab_min <= n_stab. Got n_stab_min=$(n_stab_min), n_stab=$(n_stab)."))
    measurement_profile in ("full", "equal-time-only", "density-only", "autocorr-current", "autocorr-global") ||
        throw(ArgumentError("measurement_profile must be one of: full, equal-time-only, density-only, autocorr-current, autocorr-global"))
    checkpoint_every_n_measurements >= 0 ||
        throw(ArgumentError("checkpoint_every_n_measurements must be non-negative. Got $(checkpoint_every_n_measurements)."))
    boundary = lowercase(String(boundary))
    boundary in ("periodic", "open") || throw(ArgumentError("boundary must be periodic or open"))
    if boundary == "open"
        checkerboard && throw(ArgumentError("OBC checkerboard propagation is not validated; use checkerboard=false"))
        measurement_profile in ("equal-time-only", "density-only") ||
            throw(ArgumentError("OBC supports only equal-time-only or density-only profiles"))
        occursin("_obc_", filepath) ||
            throw(ArgumentError("OBC requires a fresh filepath containing '_obc_': $(filepath)"))
    end

    β_requested = β
    Lτ = round(Int, β / Δτ)
    Lτ > 0 || throw(ArgumentError("Require β/Δτ > 0. Got β=$(β), Δτ=$(Δτ)."))
    β_commensurate = Lτ * Δτ
    if !(β_commensurate ≈ β)
        @warn "Rounding β to make Lτ*Δτ exactly commensurate" β_requested β_commensurate Δτ Lτ T_requested=(1/β_requested) T_effective=(1/β_commensurate)
        β = β_commensurate
    end

    ## Record when the simulation began.
    start_timestamp = time()

    ## Convert runtime limit from hours to seconds.
    runtime_limit = runtime_limit * 60.0^2

    ## Convert checkpoint frequency from hours to seconds.
    checkpoint_freq = checkpoint_freq * 60.0^2

    ## Construct the foldername the data will be written to.
    datafolder_prefix = if boundary == "open"
        @sprintf "attractive_hubbard_obc_rect_U%.2f_tp%.2f_mu%.2f_Lx%d_Ly%d_b%.2f" U t′ μ L Ly β
    else
        @sprintf "attractive_hubbard_rect_U%.2f_tp%.2f_mu%.2f_Lx%d_Ly%d_b%.2f" U t′ μ L Ly β
    end

    ## Get MPI process ID.
    pID = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    square_geometry = build_square_geometry(
        L, Ly; boundary=boundary, t=1.0, tprime=t′,
    )
    provenance = smoqy_provenance()

    ## Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = (L > 10),
        sID = sID,
        pID = pID
    )

    # Decide collectively whether this is a fresh run or a checkpoint resume.
    #
    # SmoQyDQMC's SimulationInfo marks a run as "resuming" if the data
    # directory already exists.  Under MPI that can race on a fresh launch:
    # one rank may create the directory before a slower rank constructs its
    # SimulationInfo, causing that slower rank to try to read a checkpoint that
    # does not exist yet.  Only resume if checkpoint files exist for every MPI
    # rank; an existing empty/stale directory is treated as a fresh run.
    resume_flag = [0] # 0: fresh, 1: resume, -1: partial/corrupt checkpoint set
    if iszero(pID)
        rank_has_checkpoint = [
            any(isfile(joinpath(simulation_info.datafolder, filename)) for filename in (
                "checkpoint_pID-$(pid).jld2",
                "checkpoint_new_pID-$(pid).jld2",
                "checkpoint_old_pID-$(pid).jld2",
            ))
            for pid in 0:(comm_size-1)
        ]
        any_checkpoint = any(rank_has_checkpoint)
        all_checkpoints = all(rank_has_checkpoint)
        resume_flag[1] = any_checkpoint ? (all_checkpoints ? 1 : -1) : 0
    end
    MPI.Bcast!(resume_flag, 0, comm)
    if resume_flag[1] == -1
        error("Partial checkpoint set found in $(simulation_info.datafolder); refusing to resume. Move/remove the incomplete directory or restore missing checkpoint files.")
    end
    simulation_info.resuming = (resume_flag[1] == 1)

    ## Initialize the directory the data will be written to if one does not already exist.
    initialize_datafolder(comm, simulation_info)

# ## Initialize simulation metadata
# At this point we need to introduce branching logic to handle whether a new simulation is being started,
# or a previous simulation is being resumed.
# We do this by checking the `simulation_info.resuming` boolean value.
# If `simulation_info.resuming = true`, then we are resuming a previous simulation, while
# `simulation_info.resuming = false` indicates we are starting a new simulation.
# Therefore, the section of code immediately below handles the case that we are starting a new simulation.

# We also introduce and initialize two new variables `n_therm = 1` and `n_updates = 1` which will keep track
# of how many rounds of thermalization and measurement updates have been performed. These two variables will
# needed to be included in the checkpoint files we write later in the simulation, as they will indicate
# where to resume a previously terminated simulation.

    ## If starting a new simulation i.e. not resuming a previous simulation.
    if !simulation_info.resuming

        ## Begin thermalization updates from start.
        n_therm = 1

        ## Begin measurements from start.
        n_measurements = 1

        ## Initialize random number generator.  Offset by MPI rank so a fixed
        ## base seed gives reproducible but independent Markov chains.
        rank_seed = seed + pID
        rng = Xoshiro(rank_seed)

        ## Initialize metadata dictionary
        metadata = Dict()

        ## Record simulation parameters.
        metadata["N_therm"] = N_therm
        metadata["N_measurements"] = N_measurements
        metadata["N_updates"] = N_updates
        metadata["N_bins"] = N_bins
        metadata["beta_requested"] = β_requested
        metadata["beta"] = β
        metadata["T_effective"] = 1 / β
        metadata["L_tau"] = Lτ
        metadata["n_stab_init"] = n_stab
        metadata["dG_max"] = δG_max
        metadata["symmetric"] = symmetric
        metadata["checkerboard"] = checkerboard
        metadata["ph_sym_form"] = ph_sym_form
        metadata["measurement_profile"] = measurement_profile
        metadata["use_reflection_update"] = use_reflection_update
        metadata["update_stabilization_frequency"] = update_stabilization_frequency
        metadata["n_stab_min"] = n_stab_min
        metadata["Ly"] = Ly
        metadata["seed"] = seed
        metadata["rank_seed"] = rank_seed
        metadata["local_acceptance_rate"] = 0.0
        metadata["reflection_acceptance_rate"] = 0.0
        metadata["boundary"] = boundary
        metadata["smoqydqmc_version"] = provenance.upstream_version
        metadata["smoqydqmc_commit"] = provenance.fork_commit
        metadata["smoqydqmc_branch"] = provenance.fork_branch
        for (key, value) in geometry_metadata(square_geometry)
            metadata["geometry_$(key)"] = value
        end
        metadata["obc_estimator_tables"] = boundary == "open" ?
            "equal_time_kinetic_per_site_qmc.tsv,equal_time_double_occupancy_per_site_qmc.tsv,equal_time_nn_spin_qmc.tsv,equal_time_nn_connected_charge_qmc.tsv" : ""

# ## Initialize Model
# No changes need to made to this section of the code from the previous
# [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial.

        ## Define unit cell.
        unit_cell = lu.UnitCell(
            lattice_vecs = [[1.0, 0.0],
                            [0.0, 1.0]],
            basis_vecs = [[0.0, 0.0]]
        )

        ## Define the finite lattice with the requested boundary conditions.
        lattice = lu.Lattice(
            L = [L, Ly],
            periodic = collect(boundary_flags(boundary))
        )

        ## Initialize model geometry.
        model_geometry = ModelGeometry(
            unit_cell, lattice
        )

        ## Define the nearest-neighbor bond in +x direction.
        bond_px = lu.Bond(
            orbitals = (1,1),
            displacement = [1, 0]
        )

        ## Add this bond definition to the model, by adding it the model_geometry.
        bond_px_id = add_bond!(model_geometry, bond_px)

        ## Define the nearest-neighbor bond in +y direction.
        bond_py = lu.Bond(
            orbitals = (1,1),
            displacement = [0, 1]
        )

        ## Add this bond definition to the model, by adding it the model_geometry.
        bond_py_id = add_bond!(model_geometry, bond_py)

        ## Define the nearest-neighbor bond in -x direction.
        ## Will be used to make measurements later in this tutorial.
        bond_nx = lu.Bond(
            orbitals = (1,1),
            displacement = [-1, 0]
        )

        ## Add this bond definition to the model, by adding it the model_geometry.
        bond_nx_id = add_bond!(model_geometry, bond_nx)

        ## Define the nearest-neighbor bond in -y direction.
        ## Will be used to make measurements later in this tutorial.
        bond_ny = lu.Bond(
            orbitals = (1,1),
            displacement = [0, -1]
        )

        ## Add this bond definition to the model, by adding it the model_geometry.
        bond_ny_id = add_bond!(model_geometry, bond_ny)

        ## Define the next-nearest-neighbor bond in +x+y direction.
        bond_pxpy = lu.Bond(
            orbitals = (1,1),
            displacement = [1, 1]
        )

        ## Add this bond definition to the model, by adding it the model_geometry.
        bond_pxpy_id = add_bond!(model_geometry, bond_pxpy)

        ## Define the next-nearest-neighbor bond in +x-y direction.
        bond_pxny = lu.Bond(
            orbitals = (1,1),
            displacement = [1, -1]
        )

        ## Add this bond definition to the model, by adding it the model_geometry.
        bond_pxny_id = add_bond!(model_geometry, bond_pxny)

        ## Set nearest-neighbor hopping amplitude to unity,
        ## setting the energy scale in the model.
        t = 1.0

        ## Define the non-interacting tight-binding model.
        tight_binding_model = TightBindingModel(
            model_geometry = model_geometry,
            t_bonds = [bond_px, bond_py, bond_pxpy, bond_pxny], # defines hopping
            t_mean = [t, t, t′, t′], # defines corresponding mean hopping amplitude
            t_std = [0., 0., 0., 0.], # defines corresponding standard deviation in hopping amplitude
            ϵ_mean = [0.], # set mean on-site energy for each orbital in unit cell
            ϵ_std = [0.], # set standard deviation of on-site energy or each orbital in unit cell
            μ = μ # set chemical potential
        )

        ## Define the Hubbard interaction in the model.
        hubbard_model = HubbardModel(
            ph_sym_form = ph_sym_form, # if particle-hole symmetric form for Hubbard interaction is used.
            U_orbital = [1], # orbitals in unit cell with Hubbard interaction.
            U_mean = [U], # mean Hubbard interaction strength for corresponding orbital species in unit cell.
            U_std = [0.], # standard deviation of Hubbard interaction strength for corresponding orbital species in unit cell.
        )

        ## Write model summary TOML file specifying Hamiltonian that will be simulated.
        model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = (hubbard_model,)
        )

# ## Initialize model parameters
# No changes need to made to this section of the code from the previous
# [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial.

        ## Initialize tight-binding parameters.
        tight_binding_parameters = TightBindingParameters(
            tight_binding_model = tight_binding_model,
            model_geometry = model_geometry,
            rng = rng
        )

        ## Initialize Hubbard interaction parameters.
        hubbard_parameters = HubbardParameters(
            model_geometry = model_geometry,
            hubbard_model = hubbard_model,
            rng = rng
        )

        ## Apply density-channel Hubbard-Stratonovich (HS) transformation to decouple the attractive Hubbard interaction,
        ## and initialize the corresponding HS fields that will be sampled in the DQMC simulation.
        hst_parameters = HubbardDensityHirschHST(
            β = β, Δτ = Δτ,
            hubbard_parameters = hubbard_parameters,
            rng = rng
        )

# ## Initialize measurements
# No changes need to made to this section of the code from the previous
# [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial.

        ## Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

        ## Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        ## Initialize the Hubbard interaction related measurements.
        initialize_measurements!(measurement_container, hubbard_model)

        if measurement_profile == "full"
            ## Initialize the single-particle electron Green's function measurement.
            initialize_correlation_measurements!(
                measurement_container = measurement_container,
                model_geometry = model_geometry,
                correlation = "greens",
                time_displaced = true,
                pairs = [(1, 1)]
            )
        end

        if boundary == "periodic" && measurement_profile in ("full", "equal-time-only")
            ## Initialize density correlation function measurement.
            initialize_correlation_measurements!(
                measurement_container = measurement_container,
                model_geometry = model_geometry,
                correlation = "density",
                time_displaced = false,
                integrated = (measurement_profile == "full"),
                pairs = [(1, 1)]
            )

            ## Initialize the spin-z correlation function measurement.
            initialize_correlation_measurements!(
                measurement_container = measurement_container,
                model_geometry = model_geometry,
                correlation = "spin_z",
                time_displaced = false,
                integrated = (measurement_profile == "full"),
                pairs = [(1, 1)]
            )
        end

        if measurement_profile == "full"
            ## Initialize the pair correlation function measurement.
            initialize_correlation_measurements!(
                measurement_container = measurement_container,
                model_geometry = model_geometry,
                correlation = "pair",
                time_displaced = false,
                integrated = true,
                pairs = [(1, 1)]
            )
        end

        ## Initialize the x-current/current correlation measurement needed for superfluid density.
        ## HOPPING_ID 1 is the +x nearest-neighbor hopping because tight_binding_model.t_bonds
        ## is [bond_px, bond_py, bond_pxpy, bond_pxny].
        if measurement_profile in ("full", "autocorr-current")
            initialize_correlation_measurements!(
                measurement_container = measurement_container,
                model_geometry = model_geometry,
                correlation = "current",
                time_displaced = false,
                integrated = true,
                pairs = [(1, 1)]
            )
        end

        if measurement_profile == "full"
            ## Initialize the d-wave pair susceptibility measurement.
            initialize_composite_correlation_measurement!(
                measurement_container = measurement_container,
                model_geometry = model_geometry,
                name = "d-wave",
                correlation = "pair",
                ids = [bond_px_id, bond_nx_id, bond_py_id, bond_ny_id],
                coefficients = [0.5, 0.5, -0.5, -0.5],
                time_displaced = false,
                integrated = true
            )
        end

        obc_equal_time_accumulator = (
            boundary == "open" && measurement_profile == "equal-time-only"
        ) ? OBCEqualTimeAccumulator(square_geometry.nsites) : nothing

# ## Write first checkpoint
# This section of code needs to be added so that a first checkpoint file is written before
# beginning a new simulation. We do this using the [`write_jld2_checkpoint`](@ref) function.
# This function all return the epoch timestamp `checkpoint_timestamp` corresponding to when
# the checkpoint file was written.

        ## Write initial checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint_mpi_serial(
            comm,
            simulation_info;
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            ## Contents of checkpoint file below.
            n_therm, n_measurements,
            tight_binding_parameters, hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng,
            obc_equal_time_accumulator
        )

# ## Load checkpoint
# If we are resuming a simulation that was previously terminated prior to completion, then
# we need to load the most recent checkpoint file using the [`read_jld2_checkpoint`](@ref) function.
# The contents of the checkpoint file are returned as a dictionary `checkpoint` by the [`read_jld2_checkpoint`](@ref) function.
# We then extract the contents of the checkpoint file from the `checkpoint` dictionary.

    ## If resuming a previous simulation.
    else

        ## Load the checkpoint file.
        checkpoint, checkpoint_timestamp = read_jld2_checkpoint(simulation_info)

        ## Unpack contents of checkpoint dictionary.
        tight_binding_parameters = checkpoint["tight_binding_parameters"]
        hubbard_parameters = checkpoint["hubbard_parameters"]
        hst_parameters = checkpoint["hst_parameters"]
        measurement_container = checkpoint["measurement_container"]
        model_geometry = checkpoint["model_geometry"]
        metadata = checkpoint["metadata"]
        obc_equal_time_accumulator = get(checkpoint, "obc_equal_time_accumulator", nothing)
        get(metadata, "boundary", "periodic") == boundary || error("checkpoint boundary mismatch")
        get(metadata, "smoqydqmc_version", "unknown") == provenance.upstream_version || error("checkpoint SmoQyDQMC version mismatch")
        get(metadata, "smoqydqmc_commit", "unknown") == provenance.fork_commit || error("checkpoint SmoQyDQMC fork commit mismatch")
        if boundary == "open" && measurement_profile == "equal-time-only"
            obc_equal_time_accumulator isa OBCEqualTimeAccumulator ||
                error("OBC checkpoint is missing the direct equal-time accumulator")
        end
        rng = checkpoint["rng"]
        n_therm = checkpoint["n_therm"]
        n_measurements = checkpoint["n_measurements"]
    end

    if boundary == "open"
        reconstructed = smoqy_hopping_matrix(tight_binding_parameters)
        isapprox(reconstructed, square_geometry.hopping; atol=1e-13, rtol=0) ||
            error("SmoQyDQMC/shared OBC one-body Hamiltonian mismatch")
    end

# ## Setup DQMC simulation
# No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

    ## Allocate FermionPathIntegral type for both the spin-up and spin-down electrons.
    ## Density-channel Hirsch fields are real for U <= 0 and complex for U > 0.
    fermion_path_integral_up = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
        forced_complex_potential = (U > 0), forced_complex_kinetic = false
    )
    fermion_path_integral_dn = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
        forced_complex_potential = (U > 0), forced_complex_kinetic = false
    )

    ## Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for Hubbard interaction.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)

    ## Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for the current
    ## Hubbard-Stratonovich field configuration.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hst_parameters)

    ## Initialize imaginary-time propagators for all imaginary-time slices for spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    ## Initialize FermionGreensCalculator type for spin-up and spin-down electrons.
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    ## Initialize alternate FermionGreensCalculator type for performing reflection updates.
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

    ## Allocate matrices for spin-up and spin-down electron Green's function matrices.
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))

    ## Initialize the spin-up and spin-down electron Green's function matrices, also
    ## calculating their respective determinants as the same time.
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    ## Allocate matrices for various time-displaced Green's function matrices.
    Gup_ττ = similar(Gup) # Gup(τ,τ)
    Gup_τ0 = similar(Gup) # Gup(τ,0)
    Gup_0τ = similar(Gup) # Gup(0,τ)
    Gdn_ττ = similar(Gdn) # Gdn(τ,τ)
    Gdn_τ0 = similar(Gdn) # Gdn(τ,0)
    Gdn_0τ = similar(Gdn) # Gdn(0,τ)

    ## Initialize diagnostic parameters to asses numerical stability.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

# ## Thermalize system
# The first change we need to make to this section is to have the for-loop iterate from `n_therm:N_therm` instead of `1:N_therm`.
# The other change we need make to this section of the code from the previous [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial
# is to add a call to the [`write_jld2_checkpoint`](@ref) function at the end of each iteration of the
# for-loop in which we perform the thermalization updates.
# When calling this function we need to pass it the timestamp for the previous checkpoint `checkpoint_timestamp`
# so that the function can determine if a new checkpoint file needs to be written.
# If a new checkpoint file is written then the `checkpoint_timestamp` variable will be updated to reflect this,
# otherwise it will remain unchanged.

    ## Iterate over number of thermalization updates to perform.
    for update in n_therm:N_therm

        if use_reflection_update
            ## Perform reflection update for HS fields with randomly chosen site.
            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn, rng = rng
            )

            ## Record whether reflection update was accepted or not.
            metadata["reflection_acceptance_rate"] += accepted
        end

        ## Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = update_stabilization_frequency && (fermion_greens_calculator_up.n_stab > n_stab_min)
        )

        ## Record acceptance rate for sweep.
        metadata["local_acceptance_rate"] += acceptance_rate

        ## Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint_mpi_serial(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            ## Contents of checkpoint file below.
            n_therm = update + 1,
            n_measurements = 1,
            tight_binding_parameters, hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng,
            obc_equal_time_accumulator
        )
    end

# ## Make measurements
# Again, we need to modify the for-loop so that it runs from `n_updates:N_measurements` instead of `1:N_measurements`.
# The only other change we need to make to this section of the code from the previous
# [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial
# is to add a call to the [`write_jld2_checkpoint`](@ref) function at the end of each iteration of the
# for-loop in which we perform updates and measurements.
# Note that we set `n_therm = N_therm + 1` when writing the checkpoint file to ensure that when the simulation
# is resumed the thermalization updates are not repeated.

    ## Reset diagnostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    ## Calculate the bin size.
    bin_size = N_measurements ÷ N_bins

    # Ranks can resume from old checkpoints with different n_measurements values.
    # A rank that already reached N_measurements must not skip the MPI control
    # flow while slower ranks finish: checkpoint writing and finalization are
    # collective operations.  Keep finished ranks in this loop as idle
    # participants until every rank reports done.
    global_measurement_step = 0

    ## Iterate over measurements collectively until every MPI rank is done.
    while true

        local_done = n_measurements > N_measurements
        done_count = [local_done ? 1 : 0]
        MPI.Allreduce!(done_count, MPI.SUM, comm)
        if done_count[1] == MPI.Comm_size(comm)
            break
        end

        global_measurement_step += 1

        if !local_done
            measurement = n_measurements

            ## Iterate over updates between measurements.
            for update in 1:N_updates

                if use_reflection_update
                    ## Perform reflection update for HS fields with randomly chosen site.
                    (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
                        Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                        hst_parameters,
                        fermion_path_integral_up = fermion_path_integral_up,
                        fermion_path_integral_dn = fermion_path_integral_dn,
                        fermion_greens_calculator_up = fermion_greens_calculator_up,
                        fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                        fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                        fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                        Bup = Bup, Bdn = Bdn, rng = rng
                    )

                    ## Record whether reflection update was accepted or not.
                    metadata["reflection_acceptance_rate"] += accepted
                end

                ## Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
                (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                    Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                    hst_parameters,
                    fermion_path_integral_up = fermion_path_integral_up,
                    fermion_path_integral_dn = fermion_path_integral_dn,
                    fermion_greens_calculator_up = fermion_greens_calculator_up,
                    fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                    Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
                    update_stabilization_frequency = update_stabilization_frequency && (fermion_greens_calculator_up.n_stab > n_stab_min)
                )

                ## Record acceptance rate.
                metadata["local_acceptance_rate"] += acceptance_rate
            end

            ## Make measurements.
            (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
                measurement_container,
                logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
                logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = (hubbard_parameters, hst_parameters),
                update_stabilization_frequency = update_stabilization_frequency && (fermion_greens_calculator_up.n_stab > n_stab_min)
            )

            if obc_equal_time_accumulator !== nothing
                phase = configuration_phase(
                    fermion_path_integral_up, sgndetGup, sgndetGdn,
                )
                record_obc_equal_time!(
                    obc_equal_time_accumulator, square_geometry, Gup, Gdn, phase; U=U,
                )
            end

            ## Write the bin-averaged measurements to file if update ÷ bin_size == 0.
            write_measurements!(
                measurement_container = measurement_container,
                simulation_info = simulation_info,
                model_geometry = model_geometry,
                measurement = measurement,
                bin_size = bin_size,
                Δτ = Δτ
            )

            n_measurements = measurement + 1
        end

        # Measurement-count checkpointing must be collective.  If only one rank
        # forces a checkpoint while another keeps measuring, the serialized
        # writer barriers no longer line up.  Use the collective loop counter so
        # all ranks enter the writer together every N collective iterations.
        force_measurement_checkpoint = checkpoint_every_n_measurements > 0 &&
            (global_measurement_step % checkpoint_every_n_measurements == 0)

        ## Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint_mpi_serial(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            force_checkpoint = force_measurement_checkpoint,
            ## Contents of checkpoint file below.
            n_therm  = N_therm + 1,
            n_measurements = n_measurements,
            tight_binding_parameters, hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng,
            obc_equal_time_accumulator
        )
    end

    if obc_equal_time_accumulator !== nothing
        write_obc_rank_accumulator(
            simulation_info.datafolder, obc_equal_time_accumulator, pID, square_geometry,
        )
        MPI.Barrier(comm)
        write_obc_pooled_outputs(
            comm, simulation_info.datafolder, obc_equal_time_accumulator, square_geometry;
            beta=β, U=U,
        )
    end

# ## Merge binned data
# No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

    ## Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

# ## Record simulation metadata
# No changes need to made to this section of the code from the previous [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial.

    ## Normalize acceptance rate.
    metadata["local_acceptance_rate"] /=  (N_therm + N_measurements * N_updates)
    if use_reflection_update
        metadata["reflection_acceptance_rate"] /=  (N_therm + N_measurements * N_updates)
    end

    ## Record final stabilization frequency used at end of simulation.
    metadata["n_stab_final"] = fermion_greens_calculator_up.n_stab

    ## Record largest numerical error.
    metadata["dG"] = δG

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

# ## Post-process results
# From the last [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial, we now need to add
# a call to the [`rename_complete_simulation`](@ref) function once the results are processed.
# This function renames the data folder to begin with `complete_*`, making it simple to identify which
# simulations ran to completion and which ones need to be resumed from the last checkpoint file.
# This function also deletes the checkpoint files that were written during the simulation.

    ## Process the simulation results, calculating final error bars for all measurements.
    ## writing final statistics to CSV files.
    process_measurements(
        comm;
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = false,
        decimals = 7,
        delimiter = " "
    )

    ## Calculate AFM correlation ratio only for square lattices.
    ## The q-neighbor shortcut below is not robust for rectangular small clusters.
    if L == Ly && measurement_profile == "full"
        ## Calculate AFM correlation ratio.
        Rafm, ΔRafm = compute_correlation_ratio(
            comm;
            datafolder = simulation_info.datafolder,
            correlation = "spin_z",
            type = "equal-time",
            id_pairs = [(1, 1)],
            id_pair_coefficients = [1.0],
            q_point = (L÷2, Ly÷2),
            q_neighbors = [
                (L÷2+1, Ly÷2), (L÷2-1, Ly÷2),
                (L÷2, Ly÷2+1), (L÷2, Ly÷2-1)
            ]
        )

        ## Record the AFM correlation ratio mean and standard deviation.
        metadata["Rafm_mean_real"] = real(Rafm)
        metadata["Rafm_mean_imag"] = imag(Rafm)
        metadata["Rafm_std"] = ΔRafm

    end

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    ## Rename the data folder to indicate the simulation is complete.
    simulation_info = rename_complete_simulation(
        comm, simulation_info,
        delete_jld2_checkpoints = true
    )

    return nothing
end # end of run_simulation function

# ## Execute script
# To execute the script, we have added two new command line arguments allowing for the assignment of both
# the `checkpoint_freq` and `runtime_limit` values.
# Therefore, a simulation can be run with the command
# ```bash
# ./scripts/run_julia_local.sh --project=julia_env scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl 0 -4.0 0.0 0.0 8 4.0 2000 2000 40 5 1.0
# ```
# or
# ```bash
# srun julia hubbard_square_checkpoint.jl 1 5.0 -0.25 -2.0 4 4.0 2000 2000 40 5 1.0
# ```
# Refer to the previous [1b) Square Hubbard Model with MPI Parallelization](@ref) tutorial for more details on how to run the simulation
# script using MPI.

# In the example calls above the code will write a new checkpoint if more than 1 hour has passed since the last checkpoint file was written.
# Note that these same commands are used to both begin a new simulation and also resume a previous simulation.
# This is a useful feature when submitting jobs on a cluster, as it allows the same job file to be used for
# both starting new simulations and resuming ones that still need to finish.

function parse_driver_cli(args)
    positional = String[]
    boundary_flag = nothing
    for arg in args
        if startswith(arg, "--boundary=")
            isnothing(boundary_flag) || throw(ArgumentError("--boundary may be specified only once"))
            boundary_flag = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--")
            throw(ArgumentError("unknown option: $(arg)"))
        else
            push!(positional, arg)
        end
    end
    length(positional) >= 11 || throw(ArgumentError("expected at least 11 positional arguments"))
    positional_boundary = length(positional) >= 25 ? lowercase(positional[25]) : "periodic"
    if !isnothing(boundary_flag) && length(positional) >= 25 && boundary_flag != positional_boundary
        throw(ArgumentError("conflicting positional and --boundary values"))
    end
    return positional, something(boundary_flag, positional_boundary)
end

if abspath(PROGRAM_FILE) == @__FILE__

    cli_args, cli_boundary = parse_driver_cli(ARGS)

    ## Initialize MPI
    MPI.Init()

    ## Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    ## Run the simulation, reading in command line arguments.
    run_simulation(
        comm;
        sID = parse(Int, cli_args[1]), # Simulation ID.
        U = parse(Float64, cli_args[2]), # Hubbard interaction.
        t′ = parse(Float64, cli_args[3]), # Next-nearest-neighbor hopping amplitude.
        μ = parse(Float64, cli_args[4]), # Chemical potential.
        L = parse(Int, cli_args[5]), # System size.
        β = parse(Float64, cli_args[6]), # Inverse temperature.
        N_therm = parse(Int, cli_args[7]), # Number of thermalization updates.
        N_measurements = parse(Int, cli_args[8]), # Total number of measurements and measurement updates.
        N_bins = parse(Int, cli_args[9]), # Number of times bin-averaged measurements are written to file.
        N_updates = parse(Int, cli_args[10]), # Number of updates between measurements.
        checkpoint_freq = parse(Float64, cli_args[11]), # Frequency with which checkpoint files are written in hours.
        runtime_limit = length(cli_args) >= 12 ? parse(Float64, cli_args[12]) : Inf, # Runtime limit in hours.
        ph_sym_form = length(cli_args) >= 13 ? parse(Bool, cli_args[13]) : true,
        filepath = length(cli_args) >= 14 ? cli_args[14] : joinpath(@__DIR__, "..", "..", "results", "interacting_qmc_ed", "smoqydqmc_attractive_hubbard_checkpoint"),
        Ly = length(cli_args) >= 15 ? parse(Int, cli_args[15]) : parse(Int, cli_args[5]),
        measurement_profile = length(cli_args) >= 16 ? cli_args[16] : "full",
        Δτ = length(cli_args) >= 17 ? parse(Float64, cli_args[17]) : 0.05,
        n_stab = length(cli_args) >= 18 ? parse(Int, cli_args[18]) : 10,
        δG_max = length(cli_args) >= 19 ? parse(Float64, cli_args[19]) : 1e-6,
        use_reflection_update = length(cli_args) >= 20 ? parse(Bool, cli_args[20]) : false,
        update_stabilization_frequency = length(cli_args) >= 21 ? parse(Bool, cli_args[21]) : false,
        n_stab_min = length(cli_args) >= 22 ? parse(Int, cli_args[22]) : 1,
        seed = length(cli_args) >= 23 ? parse(Int, cli_args[23]) : abs(rand(Int)),
        checkpoint_every_n_measurements = length(cli_args) >= 24 ? parse(Int, cli_args[24]) : 0,
        boundary = cli_boundary,
    )

    ## Finalize MPI.
    MPI.Finalize()
end
