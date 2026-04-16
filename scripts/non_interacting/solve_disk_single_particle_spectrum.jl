#!/usr/bin/env julia

using LinearAlgebra
using Printf

function parse_args(args::Vector{String})
    params = Dict(
        "radius" => 5.0,
        "t" => 1.0,
        "mu" => 0.0,
        "outdir" => "",
    )

    for arg in args
        if startswith(arg, "--radius=")
            params["radius"] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--t=")
            params["t"] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            params["mu"] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--outdir=")
            params["outdir"] = split(arg, "=", limit = 2)[2]
        elseif arg in ("-h", "--help")
            println(
                """
                Build and diagonalize the single-particle tight-binding Hamiltonian
                on a 2D square-lattice disk with open boundaries.

                Options:
                  --radius=<R>   Disk radius in lattice units (default: 5.0)
                  --t=<t>        Nearest-neighbor hopping amplitude (default: 1.0)
                  --mu=<mu>      Chemical-potential shift added as -mu on the diagonal
                                 (default: 0.0)
                  --outdir=<p>   Output directory. If omitted, uses
                                 results/non_interacting/disk_spectrum_radius_<R>

                Geometry:
                  Sites are integer lattice points (x, y) satisfying x^2 + y^2 <= R^2.
                  Hopping connects nearest neighbors that both remain inside the disk.
                """
            )
            exit(0)
        else
            error("Unrecognized argument: $arg")
        end
    end

    params["radius"] > 0 || error("radius must be positive")
    params["t"] >= 0 || error("t must be non-negative")
    return params
end

function format_radius_tag(radius::Float64)
    rounded = round(radius; digits = 6)
    tag = replace(string(rounded), "." => "p")
    return replace(tag, "-" => "m")
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

    sort!(sites; by = s -> (s[2], s[1]))
    return sites
end

function build_hamiltonian(sites::Vector{Tuple{Int, Int}}, t::Float64, mu::Float64)
    nsites = length(sites)
    site_to_index = Dict(site => i for (i, site) in pairs(sites))
    hmat = zeros(Float64, nsites, nsites)

    for (i, (x, y)) in pairs(sites)
        hmat[i, i] = -mu

        for neighbor in ((x + 1, y), (x, y + 1))
            j = get(site_to_index, neighbor, 0)
            if j != 0
                hmat[i, j] = -t
                hmat[j, i] = -t
            end
        end
    end

    return hmat
end

function write_sites(path::String, sites::Vector{Tuple{Int, Int}})
    open(path, "w") do io
        println(io, "site_index\tx\ty")
        for (i, (x, y)) in pairs(sites)
            println(io, string(i, '\t', x, '\t', y))
        end
    end
end

function write_spectrum(path::String, eigenvalues::AbstractVector{<:Real})
    open(path, "w") do io
        println(io, "level_index\teigenvalue")
        for (i, value) in pairs(eigenvalues)
            @printf(io, "%d\t%.12f\n", i, value)
        end
    end
end

function write_metadata(path::String, params::Dict, sites::Vector{Tuple{Int, Int}})
    nsites = length(sites)
    xmin = minimum(first, sites)
    xmax = maximum(first, sites)
    ymin = minimum(last, sites)
    ymax = maximum(last, sites)

    open(path, "w") do io
        println(io, "model = \"non-interacting tight-binding\"")
        println(io, "geometry = \"2D square-lattice disk\"")
        println(io, "boundary = \"open\"")
        println(io, "radius = $(params["radius"])")
        println(io, "t = $(params["t"])")
        println(io, "mu = $(params["mu"])")
        println(io, "nsites = $nsites")
        println(io, "xmin = $xmin")
        println(io, "xmax = $xmax")
        println(io, "ymin = $ymin")
        println(io, "ymax = $ymax")
        println(io, "inclusion_rule = \"x^2 + y^2 <= R^2 with integer lattice sites\"")
    end
end

function main()
    params = parse_args(ARGS)
    radius_tag = format_radius_tag(params["radius"])
    outdir = isempty(params["outdir"]) ?
        joinpath(@__DIR__, "..", "results", "non_interacting", "disk_spectrum_radius_$radius_tag") :
        params["outdir"]
    mkpath(outdir)

    sites = disk_sites(params["radius"])
    isempty(sites) && error("No lattice sites lie inside the requested disk")

    hmat = build_hamiltonian(sites, params["t"], params["mu"])
    spectrum = eigvals(Symmetric(hmat))
    sort!(spectrum)

    write_sites(joinpath(outdir, "sites.tsv"), sites)
    write_spectrum(joinpath(outdir, "eigenvalues.tsv"), spectrum)
    write_metadata(joinpath(outdir, "metadata.toml"), params, sites)

    println("Computed disk spectrum")
    println("  radius = $(params["radius"])")
    println("  t = $(params["t"])")
    println("  mu = $(params["mu"])")
    println("  number of sites = $(length(sites))")
    println("  lowest eigenvalue = $(first(spectrum))")
    println("  highest eigenvalue = $(last(spectrum))")
    println("  output directory = $outdir")
end

main()
