#!/usr/bin/env julia

using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "square_lattice_geometry.jl"))
using .SquareLatticeGeometry

@testset "shared OBC square geometry" begin
    g2 = build_square_geometry(2, 2; boundary="open")
    g3 = build_square_geometry(3, 3; boundary="open")
    @test length(g2.nn_bonds) == 4
    @test length(g2.nnn_bonds) == 2
    @test length(g3.nn_bonds) == 12
    @test length(g3.nnn_bonds) == 8
    @test g2.hopping == g2.hopping'
    @test g3.hopping == g3.hopping'
    @test count(!iszero, triu(g2.hopping, 1)) == 4
    @test count(!iszero, triu(g3.hopping, 1)) == 12
    @test_throws ArgumentError build_square_geometry(3, 3; boundary="cylindrical")

    # The CE helper and shared Hamiltonian must be identical.
    using CanEnsAFQMC
    @test hopping_matrix_Hubbard_2d(2, 2, 1.0; isOBC=true) == g2.hopping
    @test hopping_matrix_Hubbard_2d(3, 3, 1.0; isOBC=true) == g3.hopping

    # The GCE SmoQyDQMC geometry must produce that same one-body Hamiltonian.
    using SmoQyDQMC
    import SmoQyDQMC.LatticeUtilities as lu
    function smoqy_obc_hopping(L)
        unit_cell = lu.UnitCell(
            lattice_vecs=[[1.0, 0.0], [0.0, 1.0]],
            basis_vecs=[[0.0, 0.0]],
        )
        lattice = lu.Lattice(L=[L, L], periodic=[false, false])
        model_geometry = ModelGeometry(unit_cell, lattice)
        model = TightBindingModel(
            model_geometry=model_geometry,
            t_bonds=[
                lu.Bond(orbitals=(1, 1), displacement=[1, 0]),
                lu.Bond(orbitals=(1, 1), displacement=[0, 1]),
            ],
            t_mean=[1.0, 1.0], t_std=[0.0, 0.0],
            ϵ_mean=[0.0], ϵ_std=[0.0], μ=0.0,
        )
        parameters = TightBindingParameters(
            tight_binding_model=model,
            model_geometry=model_geometry,
            rng=Xoshiro(1),
        )
        hopping = zeros(Float64, L^2, L^2)
        for edge in axes(parameters.neighbor_table, 2)
            i, j = parameters.neighbor_table[:, edge]
            hopping[i, j] -= parameters.t[edge]
            hopping[j, i] -= parameters.t[edge]
        end
        return hopping
    end
    @test smoqy_obc_hopping(2) == g2.hopping
    @test smoqy_obc_hopping(3) == g3.hopping

    # A diagonal density matrix provides a simple Wick-test oracle.
    rho_up = Diagonal(fill(0.4, 4)) |> Matrix
    rho_dn = Diagonal(fill(0.3, 4)) |> Matrix
    obs = measure_equal_time_observables(g2, rho_up, rho_dn; U=-3.0)
    @test obs.kinetic_per_site == 0
    @test obs.double_occupancy_per_site ≈ 0.12
    @test obs.density ≈ 0.7
    @test obs.local_moment ≈ 0.46
    @test obs.nn_spin ≈ (0.4^2 + 0.3^2 - 2 * 0.4 * 0.3)
    @test obs.nn_charge_connected ≈ 0.0 atol=1e-14

    # SmoQy Green conversion preserves diagonal occupations and orientation.
    rho = ComplexF64[0.2 0.1im; -0.1im 0.7]
    G = Matrix{ComplexF64}(I, 2, 2) - transpose(rho)
    @test density_matrix_from_smoqy_green(G) ≈ rho
end
