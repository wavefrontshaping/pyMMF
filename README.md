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


## Installation

Download the file and execute the following command.

```shell
python setup.py install
```

## How does it work?

The sovler solve, for a given index profile, the transverse part of the **scalar** propagation equation.
It finds the modes by numerically finding the eigenvalues of the transverse operator represented as a large but sparse matrix on a square mesh.
The eigenvectors represent the mode profiles and the eigenvalues give the corresponding propagation constants.
The solver needs to know how many modes you want to compute, if the number set is higher than the number of propagationg modes, it will only returns the propagating modes.
More detailed explanations can be found is this two-part tutorial:
* [Finding modes of straight fibers](https://www.wavefrontshaping.net/post/id/3)
* [Finding modes of bent fibers](https://www.wavefrontshaping.net/post/id/4)
* [Fast numerical estimations of axisymmetric multimode fibers modes](https://www.wavefrontshaping.net/post/id/66)

## Examples

### Example 1: Finding the modes of a graded index fiber (GRIN)

#### Preambule

```python
import pyMMF
import numpy as np
import matplotlib as pyplot
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
profile.initParabolicGRIN(n1=n1,a=radius,NA=NA)
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
