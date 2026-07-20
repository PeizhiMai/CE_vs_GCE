# Legacy SmoQyDQMC v2.0.11 environment

This immutable project/manifest pair records the package environment used by
pre-OBC production checkpoints. Resume those checkpoints only with this lock
and the original Julia/MPI preferences. New v2.0.12 or OBC runs must use fresh
run roots and must not load v2.0.11 JLD2 checkpoints.

The preserved manifest contains the original absolute `CanEnsAFQMC` development
path. On another machine, instantiate the project and then point that dependency
to CanEnsAFQMC commit `21b4f6815d0b836973064ff8401fb2ba9c23b802`
before attempting a legacy resume.
