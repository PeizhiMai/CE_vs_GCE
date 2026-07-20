module SquareLatticeGeometry

using LinearAlgebra
using TOML

export SquareGeometry,
       build_square_geometry,
       boundary_flags,
       site_index,
       density_matrix_from_smoqy_green,
       measure_equal_time_observables,
       geometry_metadata,
       validate_geometry

"""Physical square-lattice geometry shared by CE, GCE, and ED."""
struct SquareGeometry{T<:AbstractFloat}
    Lx::Int
    Ly::Int
    nsites::Int
    boundary::Symbol
    periodic::NTuple{2,Bool}
    coordinates::Vector{NTuple{2,Int}}
    nn_bonds::Vector{NTuple{2,Int}}
    nnn_bonds::Vector{NTuple{2,Int}}
    hopping::Matrix{T}
    t::T
    tprime::T
end

"""Return `(periodic_x, periodic_y)` for the public boundary setting."""
function boundary_flags(boundary)
    value = Symbol(lowercase(String(boundary)))
    value === :periodic && return (true, true)
    value === :open && return (false, false)
    throw(ArgumentError("boundary must be periodic or open; got $(boundary)"))
end

@inline site_index(x::Int, y::Int, Lx::Int) = x + 1 + y * Lx

function _add_physical_bond!(bonds::Set{NTuple{2,Int}}, i::Int, j::Int)
    i == j && return bonds
    push!(bonds, i < j ? (i, j) : (j, i))
    return bonds
end

"""
    build_square_geometry(Lx, Ly; boundary="periodic", t=1.0, tprime=0.0)

Build an x-fastest square lattice, its unique physical NN/NNN undirected bonds,
and the Hermitian one-body hopping matrix. `boundary="open"` is open in both
directions; mixed boundaries are deliberately not part of the v1 application API.
"""
function build_square_geometry(
    Lx::Int,
    Ly::Int;
    boundary="periodic",
    t::Real=1.0,
    tprime::Real=0.0,
)
    Lx > 0 && Ly > 0 || throw(ArgumentError("Lx and Ly must be positive"))
    periodic = boundary_flags(boundary)
    boundary_symbol = all(periodic) ? :periodic : :open
    T = promote_type(typeof(float(t)), typeof(float(tprime)))
    tT, tpT = T(t), T(tprime)
    nsites = Lx * Ly
    coordinates = [(x, y) for y in 0:(Ly - 1) for x in 0:(Lx - 1)]
    nn = Set{NTuple{2,Int}}()
    nnn = Set{NTuple{2,Int}}()

    for y in 0:(Ly - 1), x in 0:(Lx - 1)
        i = site_index(x, y, Lx)

        # Positive x and y generate every undirected nearest-neighbor edge once
        # for generic sizes. The set also removes small-lattice PBC duplicates.
        if x + 1 < Lx
            _add_physical_bond!(nn, i, site_index(x + 1, y, Lx))
        elseif periodic[1]
            _add_physical_bond!(nn, i, site_index(0, y, Lx))
        end
        if y + 1 < Ly
            _add_physical_bond!(nn, i, site_index(x, y + 1, Lx))
        elseif periodic[2]
            _add_physical_bond!(nn, i, site_index(x, 0, Lx))
        end

        # Positive x ± y diagonals generate the physical NNN edges.
        x2 = x + 1
        if x2 >= Lx
            periodic[1] || continue
            x2 = 0
        end
        for dy in (-1, 1)
            y2 = y + dy
            if !(0 <= y2 < Ly)
                periodic[2] || continue
                y2 = mod(y2, Ly)
            end
            _add_physical_bond!(nnn, i, site_index(x2, y2, Lx))
        end
    end

    nn_bonds = sort!(collect(nn))
    nnn_bonds = sort!(collect(nnn))
    hopping = zeros(T, nsites, nsites)
    for (i, j) in nn_bonds
        hopping[i, j] = hopping[j, i] = -tT
    end
    for (i, j) in nnn_bonds
        hopping[i, j] += -tpT
        hopping[j, i] += -tpT
    end

    geometry = SquareGeometry(
        Lx, Ly, nsites, boundary_symbol, periodic, coordinates,
        nn_bonds, nnn_bonds, hopping, tT, tpT,
    )
    validate_geometry(geometry)
    return geometry
end

"""Validate counts, site ordering, Hermiticity, and absence of OBC wrap edges."""
function validate_geometry(geometry::SquareGeometry)
    (; Lx, Ly, nsites, boundary, coordinates, nn_bonds, nnn_bonds, hopping) = geometry
    length(coordinates) == nsites || error("coordinate/site-count mismatch")
    hopping == adjoint(hopping) || error("hopping matrix is not Hermitian")
    all(1 <= i < j <= nsites for (i, j) in nn_bonds) || error("invalid NN bond")
    all(1 <= i < j <= nsites for (i, j) in nnn_bonds) || error("invalid NNN bond")
    length(unique(nn_bonds)) == length(nn_bonds) || error("duplicate NN bond")
    length(unique(nnn_bonds)) == length(nnn_bonds) || error("duplicate NNN bond")

    if boundary === :open
        expected_nn = (Lx - 1) * Ly + Lx * (Ly - 1)
        expected_nnn = 2 * (Lx - 1) * (Ly - 1)
        length(nn_bonds) == expected_nn || error("OBC NN count mismatch")
        length(nnn_bonds) == expected_nnn || error("OBC NNN count mismatch")
        for (i, j) in nn_bonds
            xi, yi = coordinates[i]
            xj, yj = coordinates[j]
            abs(xi - xj) + abs(yi - yj) == 1 || error("OBC NN wrap edge detected: $(i),$(j)")
        end
        for (i, j) in nnn_bonds
            xi, yi = coordinates[i]
            xj, yj = coordinates[j]
            (abs(xi - xj), abs(yi - yj)) == (1, 1) || error("OBC NNN wrap edge detected: $(i),$(j)")
        end
    end
    return geometry
end

"""
Convert SmoQyDQMC's equal-time Green matrix `G[i,j]=<c_i c_j^dagger>`
to the one-body density matrix `rho[i,j]=<c_i^dagger c_j>`.
"""
function density_matrix_from_smoqy_green(G::AbstractMatrix)
    size(G, 1) == size(G, 2) || throw(DimensionMismatch("G must be square"))
    rho = -Matrix(transpose(G))
    for i in axes(rho, 1)
        rho[i, i] += one(eltype(rho))
    end
    return rho
end

@inline function _same_spin_density_pair(rho, i::Int, j::Int)
    return rho[i, i] * rho[j, j] - rho[i, j] * rho[j, i]
end

function _bond_observables(rho_up, rho_dn, bonds)
    isempty(bonds) && return (spin=NaN, charge_raw=NaN)
    spin = zero(promote_type(eltype(rho_up), eltype(rho_dn)))
    charge = zero(spin)
    for (i, j) in bonds
        upup = _same_spin_density_pair(rho_up, i, j)
        dndn = _same_spin_density_pair(rho_dn, i, j)
        updn = rho_up[i, i] * rho_dn[j, j]
        dnup = rho_dn[i, i] * rho_up[j, j]
        spin += upup + dndn - updn - dnup
        charge += upup + dndn + updn + dnup
    end
    norm = inv(length(bonds))
    return (spin=real(spin * norm), charge_raw=real(charge * norm))
end

"""
    measure_equal_time_observables(geometry, rho_up, rho_dn; U=0)

Direct real-space equal-time estimators shared by CE, GCE, and ED. Spin uses
`(n_up-n_dn)_i (n_up-n_dn)_j` (no factor 1/4), matching the existing CE
thermometry convention. The returned charge values are raw pair averages;
`*_charge_connected` subtracts the product of ensemble site means when
`mean_density_up/dn` are supplied and otherwise subtracts the current density
matrix site means. Production pooling should supply the ensemble means.
"""
function measure_equal_time_observables(
    geometry::SquareGeometry,
    rho_up::AbstractMatrix,
    rho_dn::AbstractMatrix;
    U::Real=0.0,
    mean_density_up=nothing,
    mean_density_dn=nothing,
)
    n = geometry.nsites
    size(rho_up) == (n, n) || throw(DimensionMismatch("rho_up does not match geometry"))
    size(rho_dn) == (n, n) || throw(DimensionMismatch("rho_dn does not match geometry"))
    nup_site = real.(diag(rho_up))
    ndn_site = real.(diag(rho_dn))
    docc = sum(nup_site .* ndn_site) / n
    density = sum(nup_site .+ ndn_site) / n
    kinetic = real(sum(geometry.hopping .* (rho_up .+ rho_dn))) / n
    interaction = U * docc
    nn = _bond_observables(rho_up, rho_dn, geometry.nn_bonds)
    nnn = _bond_observables(rho_up, rho_dn, geometry.nnn_bonds)

    mean_up = isnothing(mean_density_up) ? nup_site : mean_density_up
    mean_dn = isnothing(mean_density_dn) ? ndn_site : mean_density_dn
    length(mean_up) == n && length(mean_dn) == n || throw(DimensionMismatch("site-density mean length mismatch"))
    mean_total = mean_up .+ mean_dn
    connected(raw, bonds) = isempty(bonds) ? NaN : raw - sum(mean_total[i] * mean_total[j] for (i, j) in bonds) / length(bonds)

    return (
        kinetic_per_site=kinetic,
        double_occupancy_per_site=docc,
        interaction_per_site=interaction,
        total_energy_per_site=kinetic + interaction,
        density=density,
        achieved_N=density * n,
        local_moment=density - 2docc,
        nn_spin=nn.spin,
        nn_charge_raw=nn.charge_raw,
        nn_charge_connected=connected(nn.charge_raw, geometry.nn_bonds),
        nnn_spin=nnn.spin,
        nnn_charge_raw=nnn.charge_raw,
        nnn_charge_connected=connected(nnn.charge_raw, geometry.nnn_bonds),
        density_up_site=nup_site,
        density_dn_site=ndn_site,
    )
end

"""Manifest-ready geometry and normalization metadata."""
function geometry_metadata(geometry::SquareGeometry)
    return Dict(
        "boundary" => String(geometry.boundary),
        "periodic_x" => geometry.periodic[1],
        "periodic_y" => geometry.periodic[2],
        "Lx" => geometry.Lx,
        "Ly" => geometry.Ly,
        "site_count" => geometry.nsites,
        "nn_bond_count" => length(geometry.nn_bonds),
        "nnn_bond_count" => length(geometry.nnn_bonds),
        "site_ordering" => "x-fastest: site=x+1+y*Lx",
        "kinetic_normalization" => "sum over physical hopping matrix / site_count",
        "double_occupancy_normalization" => "sum over sites / site_count",
        "nn_normalization" => "sum over existing undirected NN bonds / nn_bond_count",
        "nnn_normalization" => "sum over existing undirected NNN bonds / nnn_bond_count",
        "spin_definition" => "(n_up-n_dn)_i*(n_up-n_dn)_j",
        "connected_charge_definition" => "<n_i*n_j>-<n_i>*<n_j>, bond averaged",
    )
end

end # module
