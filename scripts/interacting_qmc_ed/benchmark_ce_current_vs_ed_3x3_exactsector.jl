#!/usr/bin/env julia

using LinearAlgebra
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
        "dtau" => 0.025,
        "beta" => 10.0,
        "nwarmups" => 64,
        "batch_nsamples" => 8,
        "measure_interval" => 2,
        "stab_interval" => 10,
        "cluster_size" => 3,
        "seed" => 1234,
        "stderr_target" => 0.02,
        "max_batches" => 4,
        "num_origins" => 1,
        "use_charge_hs" => false,
        "is_lowrank" => true,
        "sys_type" => "complex",
        "output_dir" => joinpath("results", "interacting_qmc_ed", "ce_current_vs_ed_3x3_exactsector"),
    )
    for arg in args
        if startswith(arg, "--batch-nsamples=")
            params["batch_nsamples"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--max-batches=")
            params["max_batches"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--nwarmups=")
            params["nwarmups"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--measure-interval=")
            params["measure_interval"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--u=")
            params["u"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dtau=")
            params["dtau"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--beta=")
            params["beta"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            params["seed"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output-dir=")
            params["output_dir"] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--stderr-target=")
            params["stderr_target"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use-charge-hs=")
            params["use_charge_hs"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--is-lowrank=")
            params["is_lowrank"] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sys-type=")
            params["sys_type"] = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num-origins=")
            params["num_origins"] = parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return params
end

function build_system(params)
    tmat = hopping_matrix_Hubbard_2d(params["lx"], params["ly"], 1.0)
    time_slices = round(Int, params["beta"] / params["dtau"])
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
        num_FourierPoints=system.V + 1,
        forceSymmetry=true,
        isLowrank=params["is_lowrank"],
        lrThld=1e-10,
        saveRatio=false,
    )
end

function gen_basis(nsites::Int, nparticles::Int)
    basis = Int[]
    function rec(start::Int, left::Int, state::Int)
        if left == 0
            push!(basis, state)
            return
        end
        for i in start:(nsites - left + 1)
            rec(i + 1, left - 1, state | (1 << (i - 1)))
        end
    end
    rec(1, nparticles, 0)
    return basis
end

@inline occ(state::Int, site::Int) = (state >> (site - 1)) & 1

function cdagc(state::Int, i::Int, j::Int)
    if occ(state, j) == 0 || occ(state, i) == 1
        return nothing
    end
    mask_j = 1 << (j - 1)
    st = state & ~mask_j
    sign1 = isodd(count_ones(state & (mask_j - 1))) ? -1 : 1
    mask_i = 1 << (i - 1)
    sign2 = isodd(count_ones(st & (mask_i - 1))) ? -1 : 1
    return sign1 * sign2, st | mask_i
end

function bilinear_sector_matrix(A::AbstractMatrix{ComplexF64}, basis::Vector{Int})
    dim = length(basis)
    index = Dict(state => idx for (idx, state) in enumerate(basis))
    M = zeros(ComplexF64, dim, dim)
    V = size(A, 1)
    for (col, state) in enumerate(basis)
        for i in 1:V, j in 1:V
            abs(A[i, j]) < 1e-14 && continue
            out = cdagc(state, i, j)
            out === nothing && continue
            sign, new_state = out
            row = index[new_state]
            M[row, col] += A[i, j] * sign
        end
    end
    return M
end

function occupied_sites(state::Int, nsites::Int)
    out = Vector{Int}(undef, count_ones(state))
    k = 1
    for i in 1:nsites
        if occ(state, i) == 1
            out[k] = i
            k += 1
        end
    end
    return out
end

function manybody_propagator_sector(B::AbstractMatrix{ComplexF64}, occ_lists::Vector{Vector{Int}})
    dim = length(occ_lists)
    M = zeros(ComplexF64, dim, dim)
    for row in 1:dim
        rows = occ_lists[row]
        for col in 1:dim
            cols = occ_lists[col]
            M[row, col] = det(@view B[rows, cols])
        end
    end
    return M
end

function build_B_slices(system, walker; shift::Int=0)
    V, L = system.V, system.L
    Bup = [zeros(ComplexF64, V, V) for _ in 1:L]
    Bdn = [zeros(ComplexF64, V, V) for _ in 1:L]
    tmp = zeros(ComplexF64, V, V)
    for l in 1:L
        σ = @view walker.auxfield[:, mod1(l + shift, L)]
        CanEnsAFQMC.imagtime_propagator!(Bup[l], Bdn[l], σ, system, tmpmat=tmp)
    end
    return Bup, Bdn
end

function sample_current_response_exactsector(system, walker, basis, occ_lists, Jq, Jm; shift::Int=0)
    Bup, Bdn = build_B_slices(system, walker, shift=shift)
    mup = [manybody_propagator_sector(B, occ_lists) for B in Bup]
    mdn = [manybody_propagator_sector(B, occ_lists) for B in Bdn]

    function mixed_derivatives_product(Ms, A, B)
        dim = size(Ms[1], 1)
        P00 = Matrix{ComplexF64}(I, dim, dim)
        P10 = zeros(ComplexF64, dim, dim)
        P01 = zeros(ComplexF64, dim, dim)
        P11 = zeros(ComplexF64, dim, dim)
        AB = A * B

        for M in Ms
            Q00 = M * P00
            Q10 = M * P10
            Q01 = M * P01
            Q11 = M * P11

            P00 = Q00
            P10 = A * Q00 + Q10
            P01 = B * Q00 + Q01
            P11 = AB * Q00 + A * Q01 + B * Q10 + Q11

            scale = maximum(abs, P00)
            scale = max(scale, maximum(abs, P10))
            scale = max(scale, maximum(abs, P01))
            scale = max(scale, maximum(abs, P11))
            if scale > 0
                P00 ./= scale
                P10 ./= scale
                P01 ./= scale
                P11 ./= scale
            end
        end

        return tr(P00), tr(P10), tr(P01), tr(P11)
    end

    Zup, Zup_q, Zup_m, Zup_qm = mixed_derivatives_product(mup, Jq, Jm)
    Zdn, Zdn_q, Zdn_m, Zdn_qm = mixed_derivatives_product(mdn, Jq, Jm)
    total = Zup_qm * Zdn + Zup_q * Zdn_m + Zup_m * Zdn_q + Zup * Zdn_qm
    return real((system.β / system.L) * total / (Zup * Zdn) / system.L / system.V)
end

function write_metadata(outdir, params)
    metadata = Dict(
        "target" => "3x3 CE-QMC exact-sector current benchmark",
        "lattice" => Dict("lx" => params["lx"], "ly" => params["ly"]),
        "particles" => Dict("nup" => params["nup"], "ndn" => params["ndn"]),
        "model" => Dict(
            "u" => params["u"],
            "dtau" => params["dtau"],
            "beta" => params["beta"],
            "use_charge_hs" => params["use_charge_hs"],
            "sys_type" => params["sys_type"],
            "is_lowrank" => params["is_lowrank"],
        ),
    )
    open(joinpath(outdir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function write_summary(outdir, params, means, errs, nsamples, batches, time_slices)
    open(joinpath(outdir, "summary.tsv"), "w") do io
        println(io, "beta\tlambda_longitudinal_qmin0\tlambda_longitudinal_stderr\tlambda_transverse_0qmin\tlambda_transverse_stderr\trho_s_current\trho_s_current_stderr\ttime_slices\tnsamples\tbatches")
        println(io,
            string(
                params["beta"], '\t',
                means[1], '\t', errs[1], '\t',
                means[2], '\t', errs[2], '\t',
                means[3], '\t', errs[3], '\t',
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

    system = build_system(params)
    qmc = build_qmc(system, params)
    basis = gen_basis(system.V, system.N[1])
    occ_lists = [occupied_sites(state, system.V) for state in basis]
    qxmin = 2π / system.Ns[1]
    qymin = 2π / system.Ns[2]
    JqL = bilinear_sector_matrix(current_operator_x(system, qx=qxmin, qy=0.0), basis)
    JmL = bilinear_sector_matrix(current_operator_x(system, qx=-qxmin, qy=0.0), basis)
    JqT = bilinear_sector_matrix(current_operator_x(system, qx=0.0, qy=qymin), basis)
    JmT = bilinear_sector_matrix(current_operator_x(system, qx=0.0, qy=-qymin), basis)

    all = Matrix{Float64}(undef, 0, 3)
    for batch in 1:params["max_batches"]
        Random.seed!(params["seed"] + batch - 1)
        walker = Walker(system, qmc)
        sweep!(system, qmc, walker, loop_number=qmc.nwarmups)
        batch_data = zeros(Float64, qmc.nsamples, 3)
        for sample in 1:qmc.nsamples
            sweep!(system, qmc, walker, loop_number=qmc.measure_interval)
            λL = sample_current_response_exactsector(system, walker, basis, occ_lists, JqL, JmL)
            λT = sample_current_response_exactsector(system, walker, basis, occ_lists, JqT, JmT)
            ρs = 0.25 * (λL - λT)
            batch_data[sample, :] .= (λL, λT, ρs)
        end
        all = vcat(all, batch_data)
        means = vec(mean(all, dims=1))
        errs = vec(std(all, dims=1; corrected=true)) ./ sqrt(size(all, 1))
        write_summary(outdir, params, means, errs, size(all, 1), batch, system.L)
        println("batch=", batch, " lambdaL=", means[1], " lambdaT=", means[2], " rho_s=", means[3], " nsamples=", size(all, 1))
        errs[3] <= params["stderr_target"] && break
    end
end

main()
