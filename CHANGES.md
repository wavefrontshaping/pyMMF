# Release notes

## 0.7

### New features 

- Add documentation
- Add tests
- Add `save` and `load` methods for `IndexProfile` and `Mode` classes
- Add `beta_min` option for radial solver to return non-propagating or missing modes


### Changes
- `propagationModeSolver.solve` now take solver specific options as a `options` dictionnary 


## 0.6

### Bug correction

- solve issue with optimized (scipy bisect) radial solver (see [PR #8](../../pull/8))

### Changes

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

## 0.5

### Changes

- Radial solver performance improvements (Pavel Gostev)
- Semi-analytical solver performance improvements (Pavel Gostev)
- Improved documentation
- Add Jupyter notebook examples

## 0.1 

First public version
