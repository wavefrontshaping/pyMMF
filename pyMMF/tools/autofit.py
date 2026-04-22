import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from joblib import Parallel, delayed

try:
    from numba import njit
except Exception:

    def njit(*args, **kwargs):
        def deco(f):
            return f

        return deco


# ---------- Fast moments with threshold ----------
@njit(cache=True, fastmath=True)
def _centroid_and_rms_thresh_numba(img, frac):
    """
    Compute the intensity-weighted centroid and RMS radius of a 2D image,
    restricted to pixels above a fractional threshold of the image maximum.

    If the image has negative values, a global offset is added so that all
    contributing weights are non-negative. When the thresholded mass is
    zero, the routine falls back to using every pixel; if that mass is also
    zero, a null result is returned.

    Parameters
    ----------
    img : ndarray of shape (h, w)
        Real-valued 2D image (float64).
    frac : float
        Threshold as a fraction of the image maximum (e.g. 0.5 keeps pixels
        whose value is >= 0.5 * max).

    Returns
    -------
    (cy, cx) : tuple of float
        Row and column coordinates of the centroid, in pixel units.
    rms : float
        Square root of the sum of variances along y and x (spatial RMS size).
    """
    h, w = img.shape
    vmax = img[0, 0]
    vmin = img[0, 0]
    for y in range(h):
        for x in range(w):
            v = img[y, x]
            if v > vmax:
                vmax = v
            if v < vmin:
                vmin = v
    thr = frac * vmax
    offset = -vmin if vmin < 0.0 else 0.0

    m0 = 0.0
    for y in range(h):
        for x in range(w):
            v = img[y, x]
            if v >= thr:
                m0 += v + offset

    use_all = m0 <= 0.0
    if use_all:
        m0 = 0.0
        for y in range(h):
            for x in range(w):
                m0 += img[y, x] + offset
        if m0 <= 0.0:
            return (0.0, 0.0), 0.0
        mX = 0.0
        mY = 0.0
        for y in range(h):
            for x in range(w):
                vv = img[y, x] + offset
                mX += vv * x
                mY += vv * y
        cx = mX / m0
        cy = mY / m0
        varx = 0.0
        vary = 0.0
        for y in range(h):
            dy = y - cy
            for x in range(w):
                dx = x - cx
                vv = img[y, x] + offset
                varx += vv * dx * dx
                vary += vv * dy * dy
        varx /= m0
        vary /= m0
        return (cy, cx), (varx + vary) ** 0.5

    mX = 0.0
    mY = 0.0
    for y in range(h):
        for x in range(w):
            v = img[y, x]
            if v >= thr:
                vv = v + offset
                mX += vv * x
                mY += vv * y
    cx = mX / m0
    cy = mY / m0

    varx = 0.0
    vary = 0.0
    for y in range(h):
        dy = y - cy
        for x in range(w):
            v = img[y, x]
            if v >= thr:
                dx = x - cx
                vv = v + offset
                varx += vv * dx * dx
                vary += vv * dy * dy
    varx /= m0
    vary /= m0
    return (cy, cx), (varx + vary) ** 0.5


class Autofit:
    """
    - Centroids/sizes measured from TM-derived averages (thresholded).
    - TM is recentered on output and input (pure shifts, no zoom).
    - Modes M0 (high-res N_modes) are zoomed+cropped to output (N_data) and input (N_in) grids.
    - Now includes correlation-based fine-tuning of zoom on both sides.
    """

    def __init__(self, modes, order=3, mode="reflect", cval=0.0, prefilter=True):
        """
        Build an Autofit helper from a pyMMF modes object.

        The high-resolution mode matrix ``M0`` is extracted and the
        average intensity map ``mean_I_modes`` is precomputed and cached
        for later centroid / zoom estimation.

        Parameters
        ----------
        modes : pyMMF modes object
            Must expose ``indexProfile.npoints`` (grid size) and
            ``getModeMatrix()`` returning a ``(N_modes**2, K)`` matrix.
        order : int, default 3
            Spline interpolation order used by ``scipy.ndimage.affine_transform``.
        mode : str, default "reflect"
            Boundary handling mode for non-zero-pad transforms.
        cval : float, default 0.0
            Fill value used when ``mode='constant'``.
        prefilter : bool, default True
            Whether to apply spline prefiltering during affine transforms.
        """
        self.N_modes = int(modes.indexProfile.npoints)
        self.M0 = modes.getModeMatrix()  # (N_modes^2, K)
        self.mean_I_modes = (
            np.mean(np.abs(self.M0) ** 2, axis=1)
            .reshape((self.N_modes, self.N_modes))
            .astype(np.float64, copy=False)
        )

        self._order = order
        self._mode = mode
        self._cval = cval
        self._prefilter = prefilter

        # Output side
        self.n_out = None
        self._s_out = None
        self._c_modes_out = None
        self._c_data = None

        # Input side
        self.n_in = None
        self._s_in = None
        self._c_modes_in = None
        self._c_in_data = None

    # ---------- utilities ----------
    @staticmethod
    def _centroid_rms_thresh(img, threshold):
        """
        Thin wrapper around the numba kernel that casts ``img`` to float64.

        Parameters
        ----------
        img : ndarray
            2D real-valued image.
        threshold : float
            Fractional threshold of the image maximum.

        Returns
        -------
        (cy, cx), rms : tuple and float
            Centroid and RMS size as returned by the numba kernel.
        """
        return _centroid_and_rms_thresh_numba(img.astype(np.float64), float(threshold))

    @staticmethod
    def _affine_modes_to_target(s, c_in, c_out):
        """
        Build the ``(matrix, offset)`` pair that maps a high-resolution modes
        image onto a target grid with isotropic zoom ``s`` and center
        alignment from ``c_in`` (source) to ``c_out`` (target).

        The forward mapping is ``r_out = s * (r_in - c_in) + c_out``. Since
        ``scipy.ndimage.affine_transform`` expects output->input coordinates,
        this returns the inverse: ``A = (1/s) * I`` and
        ``b = c_in - c_out / s``.

        Parameters
        ----------
        s : float
            Isotropic zoom factor (must be > 0).
        c_in : array-like of shape (2,)
            Center coordinates on the source (modes) grid.
        c_out : array-like of shape (2,)
            Center coordinates on the target grid.

        Returns
        -------
        A : ndarray of shape (2, 2)
            Linear part passed as ``matrix`` to ``affine_transform``.
        b : ndarray of shape (2,)
            Translation passed as ``offset`` to ``affine_transform``.

        Raises
        ------
        ValueError
            If ``s`` is None or not strictly positive.
        """
        # forward: r_out = s*(r_in - c_in) + c_out  -> output->input: r_in = (1/s)*r_out + (c_in - c_out/s)
        if s is None or s <= 0:
            raise ValueError("Zoom `s` must be > 0.")
        A = np.array([[1.0 / s, 0.0], [0.0, 1.0 / s]], dtype=np.float64)
        c_in = np.array(c_in, dtype=np.float64)
        c_out = np.array(c_out, dtype=np.float64)
        b = c_in - (c_out / s)
        return A, b

    @staticmethod
    def _affine_pure_shift(shift):
        """
        Build the ``(matrix, offset)`` pair for a pure translation.

        Using identity linear part means there is no zoom or rotation; the
        output pixel ``r_out`` is sampled at ``r_in = r_out - shift``.

        Parameters
        ----------
        shift : array-like of shape (2,)
            Desired (dy, dx) shift in pixels applied to the image content.

        Returns
        -------
        A : ndarray of shape (2, 2)
            Identity matrix.
        b : ndarray of shape (2,)
            Offset equal to ``-shift`` to match ``affine_transform``'s
            output->input convention.
        """
        A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        b = np.array([-shift[0], -shift[1]], dtype=np.float64)
        return A, b

    def _apply_affine_single(self, img, A, B, out_shape, zero_pad=True):
        """
        Apply a single affine transform to a 2D image.

        Complex inputs are handled by transforming the real and imaginary
        parts independently and recombining them.

        Parameters
        ----------
        img : ndarray
            2D real or complex image to resample.
        A : ndarray of shape (2, 2)
            Linear part of the transform (output->input convention).
        B : ndarray of shape (2,)
            Translation part of the transform.
        out_shape : tuple of int
            Shape of the output grid.
        zero_pad : bool, default True
            If True, pad outside the source with zeros regardless of the
            instance's default boundary mode; otherwise use ``self._mode``
            and ``self._cval``.

        Returns
        -------
        ndarray
            Resampled image with shape ``out_shape`` and the same dtype
            family (real or complex) as ``img``.
        """
        mode = "constant" if zero_pad else self._mode
        cval = 0.0 if zero_pad else self._cval
        if np.iscomplexobj(img):
            real = ndimage.affine_transform(
                np.ascontiguousarray(img.real),
                matrix=A,
                offset=B,
                output_shape=out_shape,
                order=self._order,
                mode=mode,
                cval=cval,
                prefilter=self._prefilter,
            )
            imag = ndimage.affine_transform(
                np.ascontiguousarray(img.imag),
                matrix=A,
                offset=B,
                output_shape=out_shape,
                order=self._order,
                mode=mode,
                cval=cval,
                prefilter=self._prefilter,
            )
            return real + 1j * imag
        else:
            return ndimage.affine_transform(
                img,
                matrix=A,
                offset=B,
                output_shape=out_shape,
                order=self._order,
                mode=mode,
                cval=cval,
                prefilter=self._prefilter,
            )

    def _resample_modes_matrix(self, target_N, s, c_src, c_tgt, n_jobs=0):
        """
        Resample every column of the mode matrix ``M0`` onto a target grid.

        Each column is reshaped to the high-resolution ``(N_modes, N_modes)``
        image, affine-transformed with zoom ``s`` and center alignment from
        ``c_src`` to ``c_tgt``, then flattened back into a column.

        Parameters
        ----------
        target_N : int
            Side length of the target square grid.
        s : float
            Isotropic zoom factor applied to the modes.
        c_src : tuple of float
            Center of the source (modes) grid.
        c_tgt : tuple of float
            Center of the target grid.
        n_jobs : int, default 0
            If non-zero, columns are resampled in parallel using
            ``joblib.Parallel`` with a thread backend. ``0`` means serial.

        Returns
        -------
        ndarray of shape (target_N**2, K)
            Resampled mode matrix, in the same dtype as ``self.M0``.
        """
        A, b = self._affine_modes_to_target(s, c_src, c_tgt)
        out_shape = (target_N, target_N)
        K = self.M0.shape[1]

        def _work_col(colvec):
            img = colvec.reshape(self.N_modes, self.N_modes)
            out = self._apply_affine_single(img, A, b, out_shape, zero_pad=True)
            return out.reshape(target_N * target_N, 1)

        if n_jobs and n_jobs != 0:
            cols = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_work_col)(self.M0[:, j]) for j in range(K)
            )
            return np.hstack(cols).astype(self.M0.dtype, copy=False)
        else:
            out = np.empty((target_N * target_N, K), dtype=self.M0.dtype)
            for j in range(K):
                out[:, j] = _work_col(self.M0[:, j]).ravel()
            return out

    # ---------- fitting primitives ----------
    def _fit_output_side(self, mean_I_data, threshold):
        """
        Estimate output-side zoom and centers from the measured TM average.

        Compares the thresholded centroid and RMS size of the measured
        output intensity to those of the modes intensity, and stores the
        ratio as the initial output zoom ``self._s_out`` together with the
        two centroids.

        Parameters
        ----------
        mean_I_data : ndarray of shape (N_data, N_data)
            Mean output intensity over all input probes.
        threshold : float
            Fractional threshold used for the centroid/RMS estimation.
        """
        self.N_data = mean_I_data.shape[0]
        (cdy, cdx), r_data = self._centroid_rms_thresh(mean_I_data, threshold)
        (cmy, cmx), r_modes = self._centroid_rms_thresh(self.mean_I_modes, threshold)
        s_out = 1.0 if r_modes <= 1e-12 else (r_data / r_modes)
        self._s_out = float(s_out)
        self._c_modes_out = (float(cmy), float(cmx))
        self._c_data = (float(cdy), float(cdx))

    def _fit_input_side(self, TM, threshold):
        """
        Estimate input-side zoom and centers from the TM.

        Builds the mean input intensity by averaging ``|TM|**2`` over
        output pixels, then compares its thresholded centroid and RMS to
        the modes intensity to initialize ``self._s_in`` and the centers.

        Parameters
        ----------
        TM : ndarray of shape (n_out**2, n_in**2)
            Transmission matrix (typically already recentered on the
            output side).
        threshold : float
            Fractional threshold used for the centroid/RMS estimation.
        """
        mean_I_in_data = np.mean(np.abs(TM) ** 2, axis=0).reshape(
            (self.n_in, self.n_in)
        )

        (ciny, cinx), r_in = self._centroid_rms_thresh(mean_I_in_data, threshold)
        (cmy, cmx), r_mods = self._centroid_rms_thresh(self.mean_I_modes, threshold)
        s_in = 1.0 if r_mods <= 1e-12 else (r_in / r_mods)

        self._s_in = float(s_in)
        self._c_modes_in = (float(cmy), float(cmx))
        self._c_in_data = (float(ciny), float(cinx))

    # ---------- NEW: correlation-based fine-tuning of zoom ----------
    def fine_tune_zoom(
        self, target_map, side="out", init=None, widen=0.25, tol=1e-3, maxiter=50
    ):
        """
        Refine the zoom factor by maximizing the correlation between the
        zoomed modes intensity and a target intensity map.

        The zoom is centered: the modes center ``c_src`` is always mapped
        to the target grid center, so only the scale ``s`` is optimized
        (no drift). Optimization uses L-BFGS-B on ``1 - correlation``
        within ``[s0*(1-widen), s0*(1+widen)]``.

        Parameters
        ----------
        target_map : ndarray of shape (N_tgt, N_tgt)
            Measured mean intensity map (output or input side) already
            recentered on its grid center.
        side : {'out', 'in'}, default 'out'
            Which side to tune. ``'out'`` uses ``self._c_modes_out`` and
            target size ``self.N_data``; ``'in'`` uses
            ``self._c_modes_in`` and target size ``self.n_in``.
        init : float, optional
            Initial zoom. Defaults to the current ``self._s_out`` or
            ``self._s_in`` depending on ``side``.
        widen : float, default 0.25
            Half-width of the search interval around ``init``, expressed
            as a relative fraction.
        tol : float, default 1e-3
            L-BFGS-B ``ftol`` tolerance on the cost.
        maxiter : int, default 50
            Maximum number of L-BFGS-B iterations.

        Returns
        -------
        float
            Fine-tuned zoom factor. Falls back to the initial value if
            the target map has zero norm.

        Raises
        ------
        ValueError
            If ``side`` is not ``'out'`` or ``'in'``.
        """
        if side not in ("out", "in"):
            raise ValueError("side must be 'out' or 'in'.")

        if side == "out":
            N_tgt = self.N_data
            c_src = self._c_modes_out
            # After recentering TM, target center is the grid center:
            c_tgt = ((self.N_data - 1) / 2.0, (self.N_data - 1) / 2.0)
            s0 = self._s_out if init is None else float(init)
        else:
            N_tgt = self.n_in
            c_src = self._c_modes_in
            c_tgt = ((self.n_in - 1) / 2.0, (self.n_in - 1) / 2.0)
            s0 = self._s_in if init is None else float(init)

        target = target_map.astype(np.float64)
        # Normalize once to avoid repeated cost if zero-norm:
        t_norm = np.linalg.norm(target.ravel())
        if t_norm == 0:
            # Nothing to optimize against; keep initial scale
            return s0
        target /= t_norm

        def zoom_and_compare(z_scalar):
            z = float(z_scalar[0])
            if z <= 0:
                return 1.0  # invalid scale → worst cost
            A, b = self._affine_modes_to_target(z, c_src, c_tgt)
            zoomed = self._apply_affine_single(
                self.mean_I_modes, A, b, (N_tgt, N_tgt), zero_pad=True
            )
            zn = np.linalg.norm(zoomed.ravel())
            if zn == 0:
                return 1.0
            zoomed /= zn
            # 1 - correlation
            return 1.0 - float(np.dot(zoomed.ravel(), target.ravel()))

        bounds = [(s0 * (1.0 - widen), s0 * (1.0 + widen))]
        res = minimize(
            zoom_and_compare,
            x0=[s0],
            bounds=bounds,
            method="L-BFGS-B",
            options=dict(maxiter=maxiter, ftol=tol),
        )
        return float(res.x[0])

    # ---------- public: apply a manual transformation ----------
    @staticmethod
    def _parse_transform_params(params):
        """
        Normalize user-supplied transform parameters into a canonical tuple.

        Accepts either a list/tuple ``[zoom, shift_out, shift_in]`` or a
        dict with keys ``'zoom'`` (or ``'s'``), ``'shift_out'``,
        ``'shift_in'``. ``zoom`` may be a scalar (same on both sides) or
        a pair ``(s_out, s_in)``. Shifts default to ``(0.0, 0.0)`` when
        not provided; zoom defaults to ``1.0``.

        Returns
        -------
        s_out, s_in : float, float
        shift_out, shift_in : tuple of float
        """
        if isinstance(params, dict):
            s = params.get("zoom", params.get("s", 1.0))
            shift_out = params.get("shift_out", (0.0, 0.0))
            shift_in = params.get("shift_in", (0.0, 0.0))
        else:
            if len(params) != 3:
                raise ValueError(
                    "`params` list must be [zoom, shift_out, shift_in]."
                )
            s, shift_out, shift_in = params

        if np.isscalar(s):
            s_out = float(s)
            s_in = float(s)
        else:
            s_out, s_in = float(s[0]), float(s[1])

        shift_out = (float(shift_out[0]), float(shift_out[1]))
        shift_in = (float(shift_in[0]), float(shift_in[1]))
        return s_out, s_in, shift_out, shift_in

    def transform(self, TM, params, n_jobs=0):
        """
        Apply a forward affine transformation to a transmission matrix.

        The transform is composed of an isotropic zoom around the grid
        center followed by a translation, applied independently on the
        output side (columns reshaped as ``(n_out, n_out)`` images) and
        on the input side (rows reshaped as ``(n_in, n_in)`` images).
        This is the inverse operation of :meth:`realign_TM` and is
        useful for simulating a known misalignment of a clean TM.

        Parameters
        ----------
        TM : ndarray of shape (n_out**2, n_in**2)
            Transmission matrix to transform.
        params : list, tuple or dict
            Transformation parameters. Either
            ``[zoom, shift_out, shift_in]`` or
            ``{'zoom': ..., 'shift_out': ..., 'shift_in': ...}``.
            ``zoom`` may be a scalar applied to both sides or a pair
            ``(s_out, s_in)``. Shifts are ``(dy, dx)`` in pixels.
        n_jobs : int, default 0
            If non-zero, parallelize the output-column and input-row
            loops across threads via ``joblib``.

        Returns
        -------
        ndarray of shape (n_out**2, n_in**2)
            Transformed transmission matrix, same dtype as ``TM``.
        """
        s_out, s_in, shift_out, shift_in = self._parse_transform_params(params)

        N_out, N_in = TM.shape
        n_out = int(round(np.sqrt(N_out)))
        n_in = int(round(np.sqrt(N_in)))
        if n_out * n_out != N_out or n_in * n_in != N_in:
            raise ValueError("TM must be shaped (n_out**2, n_in**2).")

        c_out = ((n_out - 1) / 2.0, (n_out - 1) / 2.0)
        c_in = ((n_in - 1) / 2.0, (n_in - 1) / 2.0)

        # Forward: r_new = s * (r_old - c) + (c + shift)
        # _affine_modes_to_target uses c_src as the source center and c_tgt
        # as the placement center, which exactly encodes this mapping.
        A_out, b_out = self._affine_modes_to_target(
            s_out, c_out, (c_out[0] + shift_out[0], c_out[1] + shift_out[1])
        )
        A_in, b_in = self._affine_modes_to_target(
            s_in, c_in, (c_in[0] + shift_in[0], c_in[1] + shift_in[1])
        )

        def _work_col(colvec):
            img = colvec.reshape(n_out, n_out)
            out = self._apply_affine_single(
                img, A_out, b_out, (n_out, n_out), zero_pad=True
            )
            return out.reshape(n_out * n_out, 1)

        if n_jobs and n_jobs != 0:
            cols = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_work_col)(TM[:, j]) for j in range(N_in)
            )
            TM_out = np.hstack(cols).astype(TM.dtype, copy=False)
        else:
            TM_out = np.empty_like(TM)
            for j in range(N_in):
                TM_out[:, j] = _work_col(TM[:, j]).ravel()

        def _work_row(rowvec):
            img = rowvec.reshape(n_in, n_in)
            out = self._apply_affine_single(
                img, A_in, b_in, (n_in, n_in), zero_pad=True
            )
            return out.ravel()

        if n_jobs and n_jobs != 0:
            rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_work_row)(TM_out[i, :]) for i in range(N_out)
            )
            TM_final = np.vstack(rows).astype(TM.dtype, copy=False)
        else:
            TM_final = np.empty_like(TM_out)
            for i in range(N_out):
                TM_final[i, :] = _work_row(TM_out[i, :])

        return TM_final

    # ---------- public: full realign with fine-tuned zoom & normalized modes ----------
    def realign_TM(self, TM, params=None, n_jobs=0, do_fine_tune=True):
        """
        Recenter a measured transmission matrix and resample the modes
        onto the TM grids.

        Pipeline:
          1. Estimate output centroid, size and zoom from the mean of
             ``|TM|**2`` along inputs.
          2. Apply a pure shift to every TM column so the output
             intensity is centered on the output grid.
          3. Optionally refine the output zoom by correlating the zoomed
             modes intensity with the recentered output mean.
          4. Repeat steps 1-3 on the input side (rows of TM).
          5. Resample the high-resolution mode matrix ``M0`` onto the
             output and input TM grids using the final zooms, and
             normalize each resampled matrix by its global Frobenius
             norm.

        Parameters
        ----------
        TM : ndarray of shape (n_out**2 * pol_out, n_in**2 * pol_in)
            Measured transmission matrix. The spatial side lengths
            ``n_out`` and ``n_in`` are inferred from the shape and from
            ``params['polarizations']``.
        params : dict, optional
            Optional parameters:
              - ``threshold`` (float in (0, 1), default 0.5): fractional
                threshold used for centroid/RMS estimation.
              - ``polarizations`` (tuple of int, default (1, 1)):
                ``(pol_in, pol_out)`` polarization multiplicities packed
                into the TM shape.
        n_jobs : int, default 0
            If non-zero, parallelize shifts and mode resampling across
            threads using ``joblib``.
        do_fine_tune : bool, default True
            If True, refine both zooms via :meth:`fine_tune_zoom`.

        Returns
        -------
        TM_recentered : ndarray of shape (n_out**2, n_in**2)
            TM recentered on both sides using **pure translations only**.
            No zoom is ever applied to the TM: the zoom factors are
            estimated from the TM (from the RMS size of ``|TM|**2``
            compared to the modes intensity) but are applied only to the
            returned mode matrices below.
        new_modes_out : ndarray of shape (n_out**2, K)
            High-resolution modes zoomed by ``self._s_out`` and centered
            on the output grid, globally normalized by their Frobenius
            norm.
        new_modes_in : ndarray of shape (n_in**2, K)
            High-resolution modes zoomed by ``self._s_in`` and centered
            on the input grid, globally normalized by their Frobenius
            norm.

        Raises
        ------
        ValueError
            If ``threshold`` is outside ``(0, 1)`` or if the TM shape is
            inconsistent with the requested polarization multiplicities.
        """

        if params is None:
            params = {}
        threshold = params.get("threshold", 0.5)
        polarizations = params.get("polarizations", (1, 1))
        if not (0.0 < threshold < 1.0):
            raise ValueError("`threshold` must be in (0, 1).")

        N_out, N_in = TM.shape
        n_out = int(round(np.sqrt(N_out) / polarizations[1]))
        n_in = int(round(np.sqrt(N_in) / polarizations[0]))
        if n_out * n_out != N_out or n_in * n_in != N_in:
            raise ValueError(
                "TM must be shaped (n_out^2 * polarizations[1], n_in^2 * polarizations[0])."
            )
        self.n_out = n_out
        self.n_in = n_in

        # --- FIT OUTPUT SIDE ---
        mean_I_out = np.mean(np.abs(TM) ** 2, axis=1).reshape((n_out, n_out))
        self._fit_output_side(mean_I_out, threshold)

        # Recenter TM on output side (shift columns/images)
        p_data = ((self.N_data - 1) / 2.0, (self.N_data - 1) / 2.0)
        shift_out = (p_data[0] - self._c_data[0], p_data[1] - self._c_data[1])
        A_out, b_out = self._affine_pure_shift(shift_out)

        def _shift_output_col(colvec):
            img = colvec.reshape(self.n_out, self.n_out)
            out = self._apply_affine_single(
                img, A_out, b_out, (self.n_out, self.n_out), zero_pad=True
            )
            return out.reshape(self.n_out * self.n_out, 1)

        if n_jobs and n_jobs != 0:
            cols = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_shift_output_col)(TM[:, j]) for j in range(N_in)
            )
            TM_out = np.hstack(cols).astype(TM.dtype, copy=False)
        else:
            TM_out = np.empty_like(TM)
            for j in range(N_in):
                TM_out[:, j] = _shift_output_col(TM[:, j]).ravel()

        # Fine-tune output zoom against the recentered output mean
        if do_fine_tune:
            mean_I_out_centered = np.mean(np.abs(TM_out) ** 2, axis=1).reshape(
                (self.N_data, self.N_data)
            )
            self._s_out = self.fine_tune_zoom(mean_I_out_centered, side="out")

        # --- FIT INPUT SIDE ---
        self._fit_input_side(TM_out, threshold)

        # Recenter TM on input side (shift rows/vectors of length P=N_in^2)
        p_in = ((self.n_in - 1) / 2.0, (self.n_in - 1) / 2.0)
        shift_in = (p_in[0] - self._c_in_data[0], p_in[1] - self._c_in_data[1])
        A_in, b_in = self._affine_pure_shift(shift_in)

        def _shift_input_row(rowvec):
            img = rowvec.reshape(self.n_in, self.n_in)
            out = self._apply_affine_single(
                img, A_in, b_in, (self.n_in, self.n_in), zero_pad=True
            )
            return out.ravel()

        if n_jobs and n_jobs != 0:
            rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_shift_input_row)(TM_out[i, :]) for i in range(N_out)
            )
            TM_recentered = np.vstack(rows).astype(TM.dtype, copy=False)
        else:
            TM_recentered = np.empty_like(TM_out)
            for i in range(N_out):
                TM_recentered[i, :] = _shift_input_row(TM_out[i, :])

        # Fine-tune input zoom against the recentered input mean
        if do_fine_tune:
            mean_I_in_centered = np.mean(np.abs(TM_recentered) ** 2, axis=0).reshape(
                (self.n_in, self.n_in)
            )
            self._s_in = self.fine_tune_zoom(mean_I_in_centered, side="in")

        # --- RESAMPLE MODES WITH FINAL SCALES ---
        new_modes_out = self._resample_modes_matrix(
            target_N=self.n_out,
            s=self._s_out,
            c_src=self._c_modes_out,
            c_tgt=(
                (self.N_data - 1) / 2.0,
                (self.N_data - 1) / 2.0,
            ),  # centered after recentering TM
            n_jobs=n_jobs,
        )
        new_modes_in = self._resample_modes_matrix(
            target_N=self.n_in,
            s=self._s_in,
            c_src=self._c_modes_in,
            c_tgt=((self.n_in - 1) / 2.0, (self.n_in - 1) / 2.0),
            n_jobs=n_jobs,
        )

        # Normalize modes matrices (single scalar each)
        nout = np.linalg.norm(new_modes_out)
        if nout > 0:
            new_modes_out = new_modes_out / nout
        nin = np.linalg.norm(new_modes_in)
        if nin > 0:
            new_modes_in = new_modes_in / nin

        return TM_recentered, new_modes_out, new_modes_in
