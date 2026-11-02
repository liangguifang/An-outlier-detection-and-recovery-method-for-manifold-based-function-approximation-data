# An outlier detection and recovery method for manifold-based function approximation data

MATLAB implementation of manifold moving least-squares approximation and sparse outlier detection.

## Contents

- `src/` Core MMLS, local-coordinate, polynomial, weight, and OMP routines.
- `examples/` Helix, Klein bottle, and car-image experiments (`LXX100.m`, `KB100.m`, `car100.m`).
- `data/` Optional datasets.

## Requirements

MATLAB R2019b or newer.

## Quick start

Run commands from the repository root:

```matlab
addpath(genpath('src'));
run('examples/LXX100.m');
```

`LXX100.m` runs the helix experiment. `KB100.m` runs the Klein bottle experiment. Both scripts build the MMLS matrix once and then call the outlier detector for each trial. The image experiment is in `car100.m`; set its `imageFolder` variable to the local `data/car/` directory before running it.

The core workflow is:

```matlab
A = MMLSforA(r, epsilon, d, m, h, delta);
[Ep, Index, f_approx] = MMLSforoutlier(A, f);
```

`r` is an ambient-dimension by sample-count matrix,  `d` is the intrinsic dimension, `m` is the polynomial degree, `epsilon` is the FLCS convergence tolerance, and `delta` is the support radius used by `MMLSforA`. `A` contains the MMLS reconstruction weights. `MMLSforoutlier` returns the estimated sparse error `Ep`, its nonzero locations `Index`, and the fitted values `f_approx`.

Supporting routines include `FLCS` for local coordinates, `PolyAlpha` and `PnByAlpha` for the polynomial basis, `LW` and `w_test` for compactly supported weights, `SbySVD` for the linear solves, and `OMPBYzspForL0` for sparse recovery.

The image experiment expects grayscale PNG files in `data/car/`. The selected images are flattened into columns of `r`.

## Weight convention

`MMLSforA` evaluates `LW` with the ambient-space distance `norm(r_i - q)` and the explicit radius `delta`. `FLCS` uses `w_test` with radius `4*h` (the coefficient is set in `src/FLCS.m`). Choose `delta` and `h` for the scale of each dataset; the example scripts use `20*h`, `4*h`, and `6*h` for the helix, Klein bottle, and image experiments respectively.

## Reproducibility

Large generated `.mat`, `.fig`, `.png`, `.pdf`, and `.csv` files should be stored separately or with Git LFS.

## License

Add the project license before publishing.


