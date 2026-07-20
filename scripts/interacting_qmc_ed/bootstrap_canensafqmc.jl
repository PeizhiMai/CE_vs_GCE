import Pkg

root = normpath(joinpath(@__DIR__, "..", ".."))
env_dir = joinpath(root, "julia_env")
pkg_dir = joinpath(root, "external", "CanEnsAFQMC")
patch_dir = joinpath(root, "patches")

const CANENS_BASE_COMMIT = "21b4f6815d0b836973064ff8401fb2ba9c23b802"

function git_output(args...)
    return readchomp(Cmd(`git $(args)`; dir=pkg_dir))
end

function apply_patch_idempotently(path::AbstractString)
    isfile(path) || error("missing dependency patch: $(path)")
    # The OBC patch is deliberately generated with zero context so that the
    # patch artifact itself contains no whitespace-only context lines.  Git
    # requires --unidiff-zero to validate/apply that format; the option is also
    # harmless for the normal-context current-response patch.
    apply_cmd = Cmd(`git apply --unidiff-zero --check $path`; dir=pkg_dir)
    reverse_cmd = Cmd(`git apply --unidiff-zero --reverse --check $path`; dir=pkg_dir)
    if success(apply_cmd)
        run(Cmd(`git apply --unidiff-zero $path`; dir=pkg_dir))
        println("Applied dependency patch: ", basename(path))
    elseif success(reverse_cmd)
        println("Dependency patch already applied: ", basename(path))
    else
        error("$(basename(path)) neither applies cleanly nor is already applied")
    end
end

head = git_output("rev-parse", "HEAD")
head == CANENS_BASE_COMMIT || error(
    "CanEnsAFQMC must be checked out at $(CANENS_BASE_COMMIT); found $(head)",
)

for filename in (
    "CanEnsAFQMC-current-response.patch",
    "CanEnsAFQMC-obc.patch",
)
    apply_patch_idempotently(joinpath(patch_dir, filename))
end

Pkg.activate(env_dir)
cd(env_dir) do
    # Keep the manifest portable across local worktrees and CADES checkouts.
    Pkg.develop(path=joinpath("..", "external", "CanEnsAFQMC"))
end
Pkg.instantiate()

println("Activated Julia environment at: ", env_dir)
println("Developed package path: ", pkg_dir)
println("CanEnsAFQMC base commit: ", head)
