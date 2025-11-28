# TPSRMP
TPSRPM.jl

[![Build Status](https://github.com/RainerHeintzmann/TPSRPM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/RainerHeintzmann/TPSRPM.jl/actions/workflows/CI.yml?query=branch%3Amain)

Robust non-rigid point set registration using Thin-Plate Spline Robust Point Matching (TPS-RPM) in Julia. Handles unequal set sizes, outliers on both source and destination, and provides diagnostics and visualization helpers.

<img src="https://github.com/RainerHeintzmann/TPSRPM.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status">

## Features
- Asymmetric TPS-RPM (maps X → Y; choose X as the smaller set)
- Soft correspondences with separate source/destination outlier penalties
- Annealing schedule (β) for stable convergence
- TPS bending + Tikhonov regularization (λ)
- Translation-invariant option via centering
- Diagnostics: energy breakdown and per-iteration logging
- Grid visualization helpers for the fitted warp
- Installation

## Quick start
- Visualize the warp with a grid
API overview
- vc2mat(vec): convert a vector of tuples/points to an N×2 matrix
- tps_rpm(X, Y; kwargs...): run TPS-RPM, returns warp params and diagnostics
- beta_sched: annealing schedule for inverse temperature β
λ: TPS regularization
cout_src, cout_dst: outlier penalties (source/destination)
iters_per_beta: EM iterations per β
center: make optimization path translation-invariant
verbose: per-iteration energy breakdown
tps_rpm_apply(X, params, Q): apply fitted warp to arbitrary points Q
make_grid_points(points; mystep): grid generator over point bounding box
show_pos(size, posmat): rasterize points as dots for quick visualization
infer_order(X, (H,W)): utility to infer coordinate order (if needed)

## Notes
Asymmetric design: The TPS is fit on sources X. Prefer X to be the smaller set.
Complexity: TPS solve is O(N^3); for very large sets, subsample control points (future option).
Translation robustness: set center=true or initialize affine offset to mean(Y)−mean(X).

## License
MIT

## Citation
If you use this in research, please cite this repository:
R. Heintzmann, “TPSRPM.jl: Thin-Plate Spline Robust Point Matching in Julia,” GitHub, 2025.