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
        self.N_data = None
        self._s_out = None
        self._c_modes_out = None
        self._c_data = None

        # Input side
        self.N_in = None
        self._s_in = None
        self._c_modes_in = None
        self._c_in_data = None

    # ---------- utilities ----------
    @staticmethod
    def _centroid_rms_thresh(img, threshold):
        return _centroid_and_rms_thresh_numba(img.astype(np.float64), float(threshold))

    @staticmethod
    def _affine_modes_to_target(s, c_in, c_out):
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
        A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        b = np.array([-shift[0], -shift[1]], dtype=np.float64)
        return A, b

    def _apply_affine_single(self, img, A, B, out_shape, zero_pad=True):
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
        self.N_data = mean_I_data.shape[0]
        (cdy, cdx), r_data = self._centroid_rms_thresh(mean_I_data, threshold)
        (cmy, cmx), r_modes = self._centroid_rms_thresh(self.mean_I_modes, threshold)
        s_out = 1.0 if r_modes <= 1e-12 else (r_data / r_modes)
        self._s_out = float(s_out)
        self._c_modes_out = (float(cmy), float(cmx))
        self._c_data = (float(cdy), float(cdx))

    def _fit_input_side(self, TM, threshold):
        M, P = TM.shape
        N_in = int(round(np.sqrt(P)))
        if N_in * N_in != P:
            raise ValueError("P must be a perfect square (P=N_in^2).")
        mean_I_in_data = np.mean(np.abs(TM) ** 2, axis=0).reshape((N_in, N_in))

        (ciny, cinx), r_in = self._centroid_rms_thresh(mean_I_in_data, threshold)
        (cmy, cmx), r_mods = self._centroid_rms_thresh(self.mean_I_modes, threshold)
        s_in = 1.0 if r_mods <= 1e-12 else (r_in / r_mods)

        self.N_in = N_in
        self._s_in = float(s_in)
        self._c_modes_in = (float(cmy), float(cmx))
        self._c_in_data = (float(ciny), float(cinx))

    # ---------- NEW: correlation-based fine-tuning of zoom ----------
    def fine_tune_zoom(
        self, target_map, side="out", init=None, widen=0.25, tol=1e-3, maxiter=50
    ):
        """
        Fine-tune zoom by maximizing correlation between:
          zoomed(mean_I_modes)  vs  target_map
        with a CENTERED zoom (no drift).

        side: 'out' uses c_src=self._c_modes_out, target size N_data, center c_out=(N_data-1)/2 after TM recentering
              'in'  uses c_src=self._c_modes_in,  target size N_in,  center c_out=(N_in-1)/2 after TM recentering
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
            N_tgt = self.N_in
            c_src = self._c_modes_in
            c_tgt = ((self.N_in - 1) / 2.0, (self.N_in - 1) / 2.0)
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

    # ---------- public: full realign with fine-tuned zoom & normalized modes ----------
    def realign_TM(self, TM, params=None, n_jobs=0, do_fine_tune=True):
        """
        Returns
        -------
        TM_recentered : (N_data^2, P)
        new_modes_out : (N_data^2, K)   (normalized)
        new_modes_in  : (N_in^2,   K)   (normalized)
        """
        if params is None:
            params = {}
        threshold = params.get("threshold", 0.5)
        if not (0.0 < threshold < 1.0):
            raise ValueError("`threshold` must be in (0, 1).")

        M, P = TM.shape
        N_data = int(round(np.sqrt(M)))
        if N_data * N_data != M:
            raise ValueError("TM must be shaped (N_data^2, P).")

        # --- FIT OUTPUT SIDE ---
        mean_I_out = np.mean(np.abs(TM) ** 2, axis=1).reshape((N_data, N_data))
        self._fit_output_side(mean_I_out, threshold)

        # Recenter TM on output side (shift columns/images)
        p_data = ((self.N_data - 1) / 2.0, (self.N_data - 1) / 2.0)
        shift_out = (p_data[0] - self._c_data[0], p_data[1] - self._c_data[1])
        A_out, b_out = self._affine_pure_shift(shift_out)

        def _shift_output_col(colvec):
            img = colvec.reshape(self.N_data, self.N_data)
            out = self._apply_affine_single(
                img, A_out, b_out, (self.N_data, self.N_data), zero_pad=True
            )
            return out.reshape(self.N_data * self.N_data, 1)

        if n_jobs and n_jobs != 0:
            cols = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_shift_output_col)(TM[:, j]) for j in range(P)
            )
            TM_out = np.hstack(cols).astype(TM.dtype, copy=False)
        else:
            TM_out = np.empty_like(TM)
            for j in range(P):
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
        p_in = ((self.N_in - 1) / 2.0, (self.N_in - 1) / 2.0)
        shift_in = (p_in[0] - self._c_in_data[0], p_in[1] - self._c_in_data[1])
        A_in, b_in = self._affine_pure_shift(shift_in)

        def _shift_input_row(rowvec):
            img = rowvec.reshape(self.N_in, self.N_in)
            out = self._apply_affine_single(
                img, A_in, b_in, (self.N_in, self.N_in), zero_pad=True
            )
            return out.ravel()

        if n_jobs and n_jobs != 0:
            rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_shift_input_row)(TM_out[i, :]) for i in range(M)
            )
            TM_recentered = np.vstack(rows).astype(TM.dtype, copy=False)
        else:
            TM_recentered = np.empty_like(TM_out)
            for i in range(M):
                TM_recentered[i, :] = _shift_input_row(TM_out[i, :])

        # Fine-tune input zoom against the recentered input mean
        if do_fine_tune:
            mean_I_in_centered = np.mean(np.abs(TM_recentered) ** 2, axis=0).reshape(
                (self.N_in, self.N_in)
            )
            self._s_in = self.fine_tune_zoom(mean_I_in_centered, side="in")

        # --- RESAMPLE MODES WITH FINAL SCALES ---
        new_modes_out = self._resample_modes_matrix(
            target_N=self.N_data,
            s=self._s_out,
            c_src=self._c_modes_out,
            c_tgt=(
                (self.N_data - 1) / 2.0,
                (self.N_data - 1) / 2.0,
            ),  # centered after recentering TM
            n_jobs=n_jobs,
        )
        new_modes_in = self._resample_modes_matrix(
            target_N=self.N_in,
            s=self._s_in,
            c_src=self._c_modes_in,
            c_tgt=((self.N_in - 1) / 2.0, (self.N_in - 1) / 2.0),
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
