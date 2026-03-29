import Pkg

root = normpath(joinpath(@__DIR__, ".."))
env_dir = joinpath(root, "julia_env")
pkg_dir = joinpath(root, "external", "CanEnsAFQMC")

Pkg.activate(env_dir)
Pkg.develop(path=pkg_dir)
Pkg.instantiate()

println("Activated Julia environment at: ", env_dir)
println("Developed package path: ", pkg_dir)
