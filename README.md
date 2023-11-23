# pyMMF

Simple module to find numerically the propagation modes and their corresponding propagation constants
of multimode fibers of arbitrary index profiles.

## What is it?

**pyMMF** is a simple module that allows finding the propagating modes of multimode fibers with arbitrary index profiles and simulates the transmission matrix for a given length.
The solver can also take into account the curvature of the fiber (experimental).
This code is not designed to compete with commercially available software in term of accuracy of the mode profiles/propagation constants or speed, it aims at quickly simulating realistic transmission matrices of short sections of fiber.

## Citing the code

If the code was helpful to your work, please consider citing it:

[![DOI](https://zenodo.org/badge/148702831.svg)](https://zenodo.org/badge/latestdoi/148702831)

## Online mode predictions

Test `pyMMF` on [Replicate](https://replicate.com/wavefrontshaping/pymmf)

## Installation

Download the file and execute the following command.

```shell
python setup.py install
```

## Contributions

This code is written and maintained by S. M. Popoff

I thank contributions from Pavel Gostev [vongostev/pyMMF](https://github.com/vongostev/pyMMF):

1. Semianalytical solver has parallelized by joblib, thanks to which its performance has increased dramatically on thick fibers.
2. Stability of fast radial solver increased, specifically on thick fibers and small wavelengths.

## How does it work?

`pyMMF` proposes different solvers to find the propagation constants and mode profiles of multimode optical fibers.
They solve the the transverse part of the **scalar** propagation equation.

### Semi-analytical solver for step-index

Ideal step-index fibers allow anlytical dispersion relations and mode profile expressions.
This solver numericaly solves this relation dispersion and compute the modes using the analytical formula of the modes.
It is only valid for ideal step-index fibers.

Use `solver.solve(mode = 'SI', ...)`

### Radial solver

Solver for fibers with an axisymmetric index profile defined by a radial function,
e.g. graded index fibers.
It solves the 1D problem using the finite difference recursive scheme for Riccati's equations.
It allows finding accurately and quickly the mode profiles and propagation constants for fibers
when the index profiles only depends on the radial coordinate.

More details here:

- [Fast numerical estimations of axisymmetric multimode fibers modes](https://www.wavefrontshaping.net/post/id/66)

Use `solver.solve(mode = 'radial', ...)`

### Eigenvalue solver

It finds the modes by numerically finding the eigenvalues of the transverse operator represented as a large but sparse matrix on a square mesh.
The eigenvectors represent the mode profiles and the eigenvalues give the corresponding propagation constants.
The solver needs to know how many modes you want to compute, if the number set is higher than the number of propagationg modes, it will only returns the propagating modes.
This solver is slower and requires finer discretisations compared to the radial solver, but it allows using arbitrary,
and in particular non-axisymmetric, index profiles.
It also allows introducing bending to the fiber and finding the modes of the perturbed fiber.

More detailed explanations can be found is this two-part tutorial:

- [Finding modes of straight fibers](https://www.wavefrontshaping.net/post/id/3)
- [Finding modes of bent fibers](https://www.wavefrontshaping.net/post/id/4)

Use `solver.solve(mode = 'eig', ...)`

### WKB solver

Find the propagation constants of parabolic GRIN multimode fibers under the WKB (Wentzel–Kramers–Brillouin) approximation [1]\_.
This approximation leads to inaccurate results for modes close to the cutoff,
which can be a significant proportion of the modes for typical fibers.
It is provided only for comparison.

## Examples

### Example 1: Finding the modes of a graded index fiber (GRIN)

#### Preambule

```python
import pyMMF
import numpy as np
import matplotlib.pyplot as plt
```

#### Parameters

We first set the parameters of the fiber we want to simulate.

```python
NA = 0.275
radius = 7 # in microns
areaSize = 2.5*radius # calculate the field on an area larger than the diameter of the fiber
npoints = 2**7 # resolution of the window
n1 = 1.45
wl = 0.6328 # wavelength in microns
curvature = None
```

#### Index profile

We first create the fiber object

```python
profile = pyMMF.IndexProfile(npoints = npoints, areaSize = areaSize)
```

We use the helper function that generates a parabolic index profile:

```python
profile.initParabolicGRIN(n1=n1, a=radius, NA=NA)
```

We then give the profile and the wavelength to the solver

```python
solver = pyMMF.propagationModeSolver()
solver.setIndexProfile(profile)
solver.setWL(wl)
```

#### Run the solver

The solver needs to know how many modes you want to compute.
We estimate the number of modes of a GRIN multimode fiber.

```python
NmodesMax = pyMMF.estimateNumModesGRIN(wl,radius,NA)
```

To be safe, we ask for a bit more than the estimated number of modes previously calculated.

##### 2d eigenvalue solver

```python
modes = solver.solve(nmodesMax=NmodesMax+10,
boundary = 'close',
mode = 'eig',
curvature = curvature)
```

##### Radial solver

```python
modes = solver.solve(mode = 'radial')
```

#### Results

Ask for the number of propagating modes found by the solver (other modes are discarded).

```python
Nmodes = modes.number
```

Display the profile of a mode

```python
m = 10
plt.figure()
plt.subplot(121)
plt.imshow(np.real(modes.profiles[m]).reshape([npoints]*2))
plt.subplot(122)
plt.imshow(np.imag(modes.profiles[m]).reshape([npoints]*2))
```

### Other examples

Other examples are provided as notebooks in the [example](example) folder.

## Release notes

### 0.6

#### Bug correction

- solve issue with optimized (scipy bisect) radial solver (see [PR #8](../../pull/8))

#### Changes

- switch radial solvers: `radial` corresponds now to the corrected optimized radial solver using scipy for bisect search, `radial_legacy` is the old one
- Store radial and azimuthal functions of the modes in the `radial` solver in `modes0.data[<ind_mode>]['radial_func']` and `modes0.data[<ind_mode>]['azimuthal_func']`, can be used to apply to your mesh, e.g.:

```python
modes = solver.solve(mode='radial_test', ...)
X, Y = np.meshgrid(...)
TH = np.arctan2(Y, X)
R = np.sqrt(X**2 + Y**2)
ind_mode = 0
psi_r = modes.data[ind_mode]['radial_func'](R)
psi_theta = modes.data[ind_mode]['azimuthal_func'](TH)
plt.figure()
plt.imshow(np.real(R*TH))
```

- in the radial solver, argument `min_radius_bc` is now in units of wavelength, defaults to 4.

### 0.5

#### Changes

- Radial solver performance improvements (Pavel Gostev)
- Semi-analytical solver performance improvements (Pavel Gostev)
- Improved documentation
- Add Jupyter notebook examples

### 0.4

- First public version
