#!/usr/bin/env julia

using LinearAlgebra
using Printf

function parse_args(args::Vector{String})
    params = Dict{String, Any}(
        "radius" => 5.0,
        "t_hop" => 1.0,
        "mu" => 0.0,
        "nup" => 5,
        "ndn" => 5,
        "temperatures" => [0.25, 0.5, 1.0, 2.0, 4.0],
        "outdir" => "",
        "write_matrices" => false,
    )

    for arg in args
        if startswith(arg, "--radius=")
            params["radius"] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--t=")
            params["t_hop"] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            params["mu"] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nup=")
            params["nup"] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--ndn=")
            params["ndn"] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--temperatures=")
            raw = split(split(arg, "=", limit = 2)[2], ",")
            params["temperatures"] = [parse(Float64, strip(value)) for value in raw if !isempty(strip(value))]
        elseif startswith(arg, "--outdir=")
            params["outdir"] = split(arg, "=", limit = 2)[2]
        elseif arg == "--write-matrices"
            params["write_matrices"] = true
        elseif arg in ("-h", "--help")
            println(
                """
                Exact canonical compressibility scan for a non-interacting 2D tight-binding disk.

                Options:
                  --radius=<R>            Disk radius in lattice units (default: 5.0)
                  --t=<t>                 Nearest-neighbor hopping amplitude (default: 1.0)
                  --mu=<mu>               Diagonal shift -mu (default: 0.0)
                  --nup=<Nup>             Fixed number of up-spin particles (default: 5)
                  --ndn=<Ndn>             Fixed number of down-spin particles (default: 5)
                  --temperatures=<list>   Comma-separated list, e.g. 0.25,0.5,1.0
                  --outdir=<path>         Output directory
                  --write-matrices        Also write the connected density-correlation matrix for each T

                Notes:
                  The calculation is exact for the non-interacting model and uses canonical
                  Fourier projection at fixed Nup and Ndn.
                  For fixed total particle number, the fully integrated compressibility
                  from the equal-time density-correlation sum is expected to vanish.
                """
            )
            exit(0)
        else
            error("Unrecognized argument: $arg")
        end
    end

    params["radius"] > 0 || error("radius must be positive")
    params["t_hop"] >= 0 || error("t must be non-negative")
    isempty(params["temperatures"]) && error("at least one temperature is required")
    minimum(params["temperatures"]) > 0 || error("temperatures must be positive")
    params["nup"] >= 0 || error("nup must be non-negative")
    params["ndn"] >= 0 || error("ndn must be non-negative")

    return params
end

function format_number_tag(value::Real)
    rounded = round(Float64(value); digits = 6)
    return replace(replace(string(rounded), "." => "p"), "-" => "m")
end

function disk_sites(radius::Float64)
    r2 = radius^2
    bound = floor(Int, radius)
    sites = Tuple{Int, Int}[]

    for y in -bound:bound
        for x in -bound:bound
            if x^2 + y^2 <= r2 + 1e-12
                push!(sites, (x, y))
            end
        end
    end

    sort!(sites; by = site -> (site[2], site[1]))
    return sites
end

function build_hamiltonian(sites::Vector{Tuple{Int, Int}}, t_hop::Float64, mu::Float64)
    nsites = length(sites)
    site_to_index = Dict(site => i for (i, site) in pairs(sites))
    hmat = zeros(Float64, nsites, nsites)

    for (i, (x, y)) in pairs(sites)
        hmat[i, i] = -mu
        for neighbor in ((x + 1, y), (x, y + 1))
            j = get(site_to_index, neighbor, 0)
            if j != 0
                hmat[i, j] = -t_hop
                hmat[j, i] = -t_hop
            end
        end
    end

    return hmat
end

function canonical_projection_weights(energies::Vector{Float64}, beta::Float64, nparticles::Int)
    nlevels = length(energies)
    nparticles == 0 && return ones(ComplexF64, nlevels + 1) / (nlevels + 1), zeros(ComplexF64, nlevels + 1), 1.0 + 0im
    nparticles == nlevels && return ones(ComplexF64, nlevels + 1) / (nlevels + 1), zeros(ComplexF64, nlevels + 1), 1.0 + 0im

    shifted = energies .- minimum(energies)
    λ = exp.(-beta .* shifted)

    μref = (energies[nparticles] + energies[nparticles + 1]) / 2
    expβμ = exp(-beta * (μref - minimum(energies)))

    nfourier = nlevels + 1
    φ = [2π * m / nfourier for m in 1:nfourier]
    expiφ = exp.(im .* φ)
    ξ = expiφ ./ expβμ

    logweights = ComplexF64[]
    sizehint!(logweights, nfourier)
    for m in eachindex(expiφ)
        push!(logweights, sum(log.(1 .+ ξ[m] .* λ)) + nparticles * log(expβμ) - nparticles * im * φ[m])
    end

    maxreal = maximum(real.(logweights))
    raw = exp.(logweights .- maxreal)
    normalized = raw ./ sum(raw)
    return normalized, ξ, λ
end

function canonical_spin_observables(
    eigenvectors::Matrix{Float64},
    energies::Vector{Float64},
    beta::Float64,
    nparticles::Int,
)
    nsites = length(energies)
    if nparticles == 0
        zero_mat = zeros(ComplexF64, nsites, nsites)
        zero_vec = zeros(ComplexF64, nsites)
        return zero_vec, zero_mat
    elseif nparticles == nsites
        ones_vec = ones(ComplexF64, nsites)
        return ones_vec, Matrix(Diagonal(ones_vec))
    end

    weights, ξ, λ = canonical_projection_weights(energies, beta, nparticles)

    one_body = zeros(ComplexF64, nsites, nsites)
    nn_same = zeros(ComplexF64, nsites, nsites)

    for m in eachindex(weights)
        orbital_occ = ξ[m] .* λ ./ (1 .+ ξ[m] .* λ)
        gmat = eigenvectors * Diagonal(oribital_to_complex(orbital_occ)) * transpose(eigenvectors)
        diag_g = diag(gmat)
        nn_m = diag_g * transpose(diag_g) .- gmat .* transpose(gmat)
        for i in 1:nsites
            nn_m[i, i] = diag_g[i]
        end
        one_body .+= weights[m] .* gmat
        nn_same .+= weights[m] .* nn_m
    end

    return diag(one_body), nn_same
end

function oribital_to_complex(values::AbstractVector)
    return ComplexF64.(values)
end

function row_abs_max(mat::AbstractMatrix{<:Real})
    maxval = 0.0
    for i in axes(mat, 1)
        maxval = max(maxval, abs(sum(@view mat[i, :])))
    end
    return maxval
end

function write_metadata(path::String, params::Dict, nsites::Int, energies::Vector{Float64})
    open(path, "w") do io
        println(io, "model = \"non-interacting tight-binding\"")
        println(io, "geometry = \"2D square-lattice disk\"")
        println(io, "boundary = \"open\"")
        println(io, "compressibility_definition = \"beta/V * sum_ij (<n_i n_j> - <n_i><n_j>)\"")
        println(io, "ensemble = \"canonical fixed Nup and Ndn\"")
        println(io, "radius = $(params["radius"])")
        println(io, "t = $(params["t_hop"])")
        println(io, "mu = $(params["mu"])")
        println(io, "nup = $(params["nup"])")
        println(io, "ndn = $(params["ndn"])")
        println(io, "nsites = $nsites")
        println(io, "lowest_single_particle_energy = $(minimum(energies))")
        println(io, "highest_single_particle_energy = $(maximum(energies))")
        println(io, "temperatures = \"$(join(string.(params["temperatures"]), ","))\"")
    end
end

function write_sites(path::String, sites::Vector{Tuple{Int, Int}})
    open(path, "w") do io
        println(io, "site_index\tx\ty")
        for (i, (x, y)) in pairs(sites)
            println(io, string(i, '\t', x, '\t', y))
        end
    end
end

function write_spectrum(path::String, energies::Vector{Float64})
    open(path, "w") do io
        println(io, "level_index\teigenvalue")
        for (i, energy) in pairs(energies)
            @printf(io, "%d\t%.12f\n", i, energy)
        end
    end
end

function write_scan_header(path::String)
    open(path, "w") do io
        println(
            io,
            "temperature\tbeta\tnup\tndn\tdensity\tglobal_compressibility\tmean_local_density_variance\tmax_abs_row_sum\ttotal_connected_sum"
        )
    end
end

function append_scan_row(
    path::String,
    temperature::Float64,
    beta::Float64,
    nup::Int,
    ndn::Int,
    density::Float64,
    global_kappa::Float64,
    mean_local_var::Float64,
    max_abs_row_sum::Float64,
    total_connected_sum::Float64,
)
    open(path, "a") do io
        @printf(
            io,
            "%.12f\t%.12f\t%d\t%d\t%.12f\t%.12e\t%.12f\t%.12e\t%.12e\n",
            temperature,
            beta,
            nup,
            ndn,
            density,
            global_kappa,
            mean_local_var,
            max_abs_row_sum,
            total_connected_sum,
        )
    end
end

function write_matrix(path::String, mat::AbstractMatrix{<:Real})
    open(path, "w") do io
        nrows, ncols = size(mat)
        header = ["i\\j"; string.(1:ncols)]
        println(io, join(header, '\t'))
        for i in 1:nrows
            row = [string(i); [@sprintf("%.12e", mat[i, j]) for j in 1:ncols]]
            println(io, join(row, '\t'))
        end
    end
end

function main()
    params = parse_args(ARGS)

    sites = disk_sites(params["radius"])
    nsites = length(sites)
    max_particles = nsites
    params["nup"] <= max_particles || error("nup exceeds the number of sites in the disk")
    params["ndn"] <= max_particles || error("ndn exceeds the number of sites in the disk")

    hmat = build_hamiltonian(sites, params["t_hop"], params["mu"])
    eig = eigen(Symmetric(hmat))
    energies = collect(eig.values)
    eigenvectors = collect(eig.vectors)

    radius_tag = format_number_tag(params["radius"])
    outdir = isempty(params["outdir"]) ?
        joinpath(
            @__DIR__,
            "..",
            "results",
            "non_interacting",
            "disk_canonical_kappa_radius_$(radius_tag)_nup_$(params["nup"])_ndn_$(params["ndn"])",
        ) :
        params["outdir"]
    mkpath(outdir)

    write_metadata(joinpath(outdir, "metadata.toml"), params, nsites, energies)
    write_sites(joinpath(outdir, "sites.tsv"), sites)
    write_spectrum(joinpath(outdir, "single_particle_energies.tsv"), energies)

    scan_path = joinpath(outdir, "compressibility_scan.tsv")
    write_scan_header(scan_path)

    density = (params["nup"] + params["ndn"]) / nsites
    for temperature in params["temperatures"]
        beta = 1 / temperature
        n_up, nn_up = canonical_spin_observables(eigenvectors, energies, beta, params["nup"])
        n_dn, nn_dn = canonical_spin_observables(eigenvectors, energies, beta, params["ndn"])

        total_density = real.(n_up .+ n_dn)
        connected = real.(nn_up .- n_up * transpose(n_up) .+ nn_dn .- n_dn * transpose(n_dn))

        total_connected_sum = sum(connected)
        global_kappa = beta / nsites * total_connected_sum
        mean_local_var = sum(diag(connected)) / length(diag(connected))
        max_abs_row = row_abs_max(connected)

        append_scan_row(
            scan_path,
            temperature,
            beta,
            params["nup"],
            params["ndn"],
            density,
            global_kappa,
            mean_local_var,
            max_abs_row,
            total_connected_sum,
        )

        if params["write_matrices"]
            temp_tag = format_number_tag(temperature)
            write_matrix(joinpath(outdir, "connected_density_T_$(temp_tag).tsv"), connected)
        end
    end

    println("Computed exact canonical density-correlation scan")
    println("  radius = $(params["radius"])")
    println("  nsites = $nsites")
    println("  nup = $(params["nup"])")
    println("  ndn = $(params["ndn"])")
    println("  temperatures = $(join(string.(params["temperatures"]), ", "))")
    println("  output directory = $outdir")
end

main()
