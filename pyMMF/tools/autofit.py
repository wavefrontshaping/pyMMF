import numpy as np
from scipy import ndimage
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

    # mass on thresholded set
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

    # centroid on thresholded set
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

    # second moments on thresholded set
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


# ==================== autofit ====================
class autofit:
    """
    Single image: (N, N)
    Batch:        (N*N, P)   # each column is one image

    - fit_to_data(img, threshold=0.5): thresholded moments on both images.
    - transform(...): one-pass affine with centered zoom; zero-padded by default.
    - External params supported as [s, pre_shift, post_shift].
    """

    def __init__(self, modes, order=3, mode="reflect", cval=0.0, prefilter=True):
        self.N = int(modes.indexProfile.npoints)
        M0 = modes.getModeMatrix()
        self.mean_I = (
            np.mean(np.abs(M0) ** 2, axis=1)
            .reshape((self.N, self.N))
            .astype(np.float64, copy=False)
        )

        self._order = order
        self._mode = mode
        self._cval = cval
        self._prefilter = prefilter

        self._s = None
        self._pre = None
        self._post = None

    # ---- parameter estimation with threshold; pre/post relative to centers ----
    @staticmethod
    def _estimate_zoom_and_shifts(img_src, img_ref, threshold):
        (c1y, c1x), r1 = _centroid_and_rms_thresh_numba(
            img_src.astype(np.float64), float(threshold)
        )
        (c2y, c2x), r2 = _centroid_and_rms_thresh_numba(
            img_ref.astype(np.float64), float(threshold)
        )
        s = 1.0 if r1 <= 1e-12 else (r2 / r1)
        return (
            s,
            (c1y, c1x),
            (c2y, c2x),
        )  # return centers; pre/post are formed with pivots later

    def fit_to_data(self, img, threshold=0.5):
        if img.ndim != 2 or img.shape != (self.N, self.N):
            raise ValueError(
                f"fit_to_data expects a single image of shape ({self.N}, {self.N})."
            )

        # Get scale and centroids
        s, c1, c2 = self._estimate_zoom_and_shifts(img, self.mean_I, threshold)

        # Define pre/post RELATIVE TO IMAGE CENTERS to be consistent with centered-zoom:
        pin = ((self.N - 1) / 2.0, (self.N - 1) / 2.0)
        pout = pin  # same shape
        pre = (pin[0] - c1[0], pin[1] - c1[1])  # pre = p_in - c1
        post = (c2[0] - pout[0], c2[1] - pout[1])  # post = c2 - p_out

        self._s = float(s)
        self._pre = (float(pre[0]), float(pre[1]))
        self._post = (float(post[0]), float(post[1]))
        return self.get_params()

    def get_params(self):
        return self._s, self._pre, self._post

    # ---- centered-zoom affine (unchanged) ----
    @staticmethod
    def _affine_from_shift_zoom(s, pre_shift, post_shift, in_shape, out_shape):
        """
        affine_transform inverse mapping r_in = A r_out + b with centered zoom:
          r_out = s*(r_in + pre - p_in) + p_out + post
          => r_in = (1/s) r_out + [p_in - pre - (p_out + post)/s]
        """
        if s is None or s <= 0:
            raise ValueError("Zoom `s` must be > 0.")
        A = np.array([[1.0 / s, 0.0], [0.0, 1.0 / s]], dtype=np.float64)
        pre = np.array(pre_shift, dtype=np.float64)
        post = np.array(post_shift, dtype=np.float64)
        p_in = np.array(
            ((in_shape[0] - 1) / 2.0, (in_shape[1] - 1) / 2.0), dtype=np.float64
        )
        p_out = np.array(
            ((out_shape[0] - 1) / 2.0, (out_shape[1] - 1) / 2.0), dtype=np.float64
        )
        b = p_in - pre - (p_out + post) / s
        return A, b

    def _apply_affine_single(self, img, A, b, out_shape, zero_pad):
        mode = "constant" if zero_pad else self._mode
        cval = 0.0 if zero_pad else self._cval
        return ndimage.affine_transform(
            img,
            matrix=A,
            offset=b,
            output_shape=out_shape,
            order=self._order,
            mode=mode,
            cval=cval,
            prefilter=self._prefilter,
        )

    @staticmethod
    def _parse_params(params):
        if params is None:
            return None
        if not hasattr(params, "__len__") or len(params) != 3:
            raise ValueError("`params` must be [s, (dy,dx), (dy,dx)].")
        s = float(params[0])
        pre = tuple(params[1])
        post = tuple(params[2])
        if len(pre) != 2 or len(post) != 2:
            raise ValueError("pre_shift and post_shift must be 2-tuples (dy, dx).")
        return s, pre, post

    def transform(self, data, n_jobs=0, output_shape=None, params=None, zero_pad=True):
        """
        Apply with external `params=[s, pre, post]` (center-relative) or stored params.
        Shapes: single (N,N) or batch (N*N, P). Out-of-bounds -> 0 by default.
        """
        parsed = self._parse_params(params)
        if parsed is not None:
            s, pre, post = parsed
        else:
            if self._s is None or self._pre is None or self._post is None:
                raise RuntimeError(
                    "No transform parameters available. Provide `params` or call fit_to_data() first."
                )
            s, pre, post = self._s, self._pre, self._post

        # Single image
        if data.ndim == 2 and data.shape == (self.N, self.N):
            out_shape = output_shape or (self.N, self.N)
            A, b = self._affine_from_shift_zoom(
                s, pre, post, in_shape=data.shape, out_shape=out_shape
            )
            return self._apply_affine_single(data, A, b, out_shape, zero_pad)

        # Batch: (N*N, P)
        if data.ndim == 2 and data.shape[0] == self.N * self.N:
            P = data.shape[1]
            out_shape = output_shape or (self.N, self.N)
            A, b = self._affine_from_shift_zoom(
                s, pre, post, in_shape=(self.N, self.N), out_shape=out_shape
            )

            def _work_col(colvec):
                img = colvec.reshape(self.N, self.N)
                out = self._apply_affine_single(img, A, b, out_shape, zero_pad)
                return out.reshape(out_shape[0] * out_shape[1], 1)

            if n_jobs and n_jobs != 0:
                cols = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_work_col)(data[:, j]) for j in range(P)
                )
                return np.hstack(cols).astype(data.dtype, copy=False)
            else:
                out = np.empty((out_shape[0] * out_shape[1], P), dtype=data.dtype)
                for j in range(P):
                    out[:, j] = _work_col(data[:, j]).ravel()
                return out

        raise ValueError(
            f"Unsupported input shape {data.shape}. Expected (N,N) or (N*N, P) with N={self.N}."
        )
