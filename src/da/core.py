"""
src/da/core.py
==============
Single source of truth for all DA methods used in the WS experiments.

Public API
----------
  tempering_schedule(ntemp, alpha_s)             -> ndarray (ntemp,)
  compute_hxf(xf, ox, oy, oz, var_idx)           -> ndarray (nobs, Ne)
  aoei(yo, hxf, R0)                              -> ndarray (nobs,)
  letkf_update(xf, yo, R0, ox, oy, oz, L, vi)   -> dict
  tenkf_update(..., ntemp, alpha_s)              -> dict
  aoei_update(...)                               -> dict
  atenkf_update(..., ntemp, alpha_s)             -> dict

Conventions
-----------
- obs_error / R0 : always obs error VARIANCE (sigma^2), NOT std.
- sigma_dbz in configs is std; square it before passing here.
- All heavy arrays are float32.
- The Fortran backend is loaded lazily so unit tests run without it.
"""

import os, sys
import numpy as np

# ── Lazy Fortran loader ────────────────────────────────────────────────────

_cda = None

def _get_cda():
    global _cda
    if _cda is not None:
        return _cda
    for attempt in range(2):
        try:
            from cletkf_wloc import common_da as cda
            _cda = cda
            return _cda
        except ImportError:
            if attempt == 0:
                here = os.path.dirname(os.path.abspath(__file__))
                fort_dir = os.path.normpath(os.path.join(here, "..", "fortran"))
                if fort_dir not in sys.path:
                    sys.path.insert(0, fort_dir)
    raise RuntimeError(
        "Fortran backend (cletkf_wloc) not found. "
        "Run src/build_fortran.sh from the repo root first."
    )


# ── Tempering schedule ─────────────────────────────────────────────────────

def tempering_schedule(ntemp: int, alpha_s: float) -> np.ndarray:
    """
    Back-loaded exponential weights (Eq. 12).

        alpha_i = exp(-(Nt+1)*alpha_s / i) / sum_j exp(-(Nt+1)*alpha_s / j)

    for i = 1 ... Ntemp.  Weights sum to 1.
    Larger alpha_s -> more back-loading (stronger inflation in early steps).
    alpha_s = 0  -> equal weights = 1/Ntemp for all steps.

    At step i the obs error is inflated to R / alpha_i.
    """
    if ntemp == 1:
        return np.array([1.0], dtype=np.float32)
    i = np.arange(1, ntemp + 1, dtype=np.float64)
    w = np.exp(-(ntemp + 1) * float(alpha_s) / i)
    w /= w.sum()
    return w.astype(np.float32)


# ── Observation operator ───────────────────────────────────────────────────

def compute_hxf(xf_grid: np.ndarray,
                ox: np.ndarray,
                oy: np.ndarray,
                oz: np.ndarray,
                var_idx: dict) -> np.ndarray:
    """
    Apply the nonlinear reflectivity operator H to every ensemble member
    at every observation location.

    Parameters
    ----------
    xf_grid : (nx, ny, nz, Ne, nvar)
    ox, oy, oz : (nobs,) integer arrays, 0-based
    var_idx    : dict  {"qg":0, "qr":1, "qs":2, "T":3, "P":4, ...}

    Returns
    -------
    hxf : (nobs, Ne), float32
    """
    cda  = _get_cda()
    nobs = len(ox)
    Ne   = xf_grid.shape[3]
    hxf  = np.empty((nobs, Ne), dtype=np.float32)
    for ii in range(nobs):
        i, j, k = int(ox[ii]), int(oy[ii]), int(oz[ii])
        for m in range(Ne):
            hxf[ii, m] = cda.calc_ref(
                xf_grid[i, j, k, m, var_idx["qr"]],
                xf_grid[i, j, k, m, var_idx["qs"]],
                xf_grid[i, j, k, m, var_idx["qg"]],
                xf_grid[i, j, k, m, var_idx["T"]],
                xf_grid[i, j, k, m, var_idx["P"]],
            )
    return hxf


# ── AOEI ───────────────────────────────────────────────────────────────────

def aoei(yo: np.ndarray,
         hxf: np.ndarray,
         R0: np.ndarray) -> np.ndarray:
    """
    Adaptive Observation Error Inflation (Minamide & Zhang 2017, Eq. 4).

        R_tilde_j = max( R0_j,  d_j^2 - sigma2_f_j )

    where d_j = yo_j - mean(hxf_j) and sigma2_f_j = var(hxf_j, ddof=1).

    Inflation activates when d^2 > R0 + sigma2_f, i.e. the squared
    innovation exceeds the combined obs + background variance.

    Parameters
    ----------
    yo  : (nobs,)      observations
    hxf : (nobs, Ne)   ensemble in obs space
    R0  : (nobs,)      nominal obs error VARIANCE (floor)

    Returns
    -------
    R_tilde : (nobs,), float32, always >= R0
    """
    yo_  = np.asarray(yo,  np.float64)
    hxf_ = np.asarray(hxf, np.float64)
    R0_  = np.asarray(R0,  np.float64)
    d        = yo_ - hxf_.mean(axis=1)
    sigma2_f = hxf_.var(axis=1, ddof=1)
    return np.maximum(R0_, d**2 - sigma2_f).astype(np.float32)


# ── Single LETKF step ──────────────────────────────────────────────────────

def _letkf_step(xf_grid, hxf, yo, obs_error_var, ox, oy, oz, loc_scales):
    """
    One LETKF analysis via Fortran.  obs_error_var is R (variance).
    Returns xa : (nx, ny, nz, Ne, nvar), float32.
    """
    cda = _get_cda()
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)

    ox_f = (np.asarray(ox, np.int64) + 1).astype(np.float32)   # 1-based
    oy_f = (np.asarray(oy, np.int64) + 1).astype(np.float32)
    oz_f = (np.asarray(oz, np.int64) + 1).astype(np.float32)

    dep    = (yo - hxf.mean(axis=1)).astype(np.float32)
    oerr_f = np.asarray(obs_error_var, np.float32)
    locs_f = np.asarray(loc_scales,    np.float32)

    xa = cda.simple_letkf_wloc(
        nx=nx, ny=ny, nz=nz,
        nbv=Ne, nvar=nvar, nobs=nobs,
        hxf=np.asfortranarray(hxf.astype(np.float32)),
        xf=np.asfortranarray(xf_grid.astype(np.float32)),
        dep=dep,
        ox=ox_f, oy=oy_f, oz=oz_f,
        locs=locs_f, oerr=oerr_f,
    )
    return xa.astype(np.float32)


# ── High-level DA methods ──────────────────────────────────────────────────

def letkf_update(xf_grid, yo, obs_error_var, ox, oy, oz, loc_scales, var_idx):
    """
    Standard LETKF: single step, no AOEI, no tempering.

    Returns
    -------
    dict: xa, hxf, dep, obs_error
    """
    R0  = np.asarray(obs_error_var, np.float32)
    hxf = compute_hxf(xf_grid, ox, oy, oz, var_idx)
    xa  = _letkf_step(xf_grid, hxf, yo, R0, ox, oy, oz, loc_scales)
    return dict(
        xa=xa,
        hxf=hxf,
        dep=(yo - hxf.mean(axis=1)).astype(np.float32),
        obs_error=R0,
    )


def tenkf_update(xf_grid, yo, obs_error_var, ox, oy, oz, loc_scales, var_idx,
                 ntemp, alpha_s):
    """
    TEnKF (LETKF-T): Ntemp sequential steps with back-loaded inflation.

    At each step i: recompute H(x), inflate R -> R/alpha_i, run LETKF.

    Returns
    -------
    dict: xa, xatemp, hxfs, deps, alpha_weights, obs_error
    """
    steps = tempering_schedule(ntemp, alpha_s)
    R0    = np.asarray(obs_error_var, np.float32)
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)
    Nt   = len(steps)

    xatemp = np.empty((nx, ny, nz, Ne, nvar, Nt + 1), dtype=np.float32, order="F")
    xatemp[..., 0] = xf_grid.astype(np.float32)
    hxfs = np.empty((Nt, nobs, Ne), dtype=np.float32)
    deps = np.empty((Nt, nobs),     dtype=np.float32)

    for it in range(Nt):
        hxf = compute_hxf(xatemp[..., it], ox, oy, oz, var_idx)
        dep = (yo - hxf.mean(axis=1)).astype(np.float32)
        hxfs[it] = hxf
        deps[it] = dep
        oerr = R0 / steps[it]
        print(f"  [TEnKF]  step {it+1}/{Nt}  "
              f"alpha={steps[it]:.4f}  R/alpha={oerr.mean():.2f}  "
              f"|dep|={np.abs(dep).mean():.3f}")
        xatemp[..., it + 1] = _letkf_step(
            xatemp[..., it], hxf, yo, oerr, ox, oy, oz, loc_scales)

    return dict(xa=xatemp[..., -1], xatemp=xatemp, hxfs=hxfs, deps=deps,
                alpha_weights=steps, obs_error=R0)


def aoei_update(xf_grid, yo, obs_error_var, ox, oy, oz, loc_scales, var_idx):
    """
    LETKF + AOEI: inflate once from the prior, then one LETKF step.

    Returns
    -------
    dict: xa, hxf, dep, obs_error_raw, obs_error
    """
    R0  = np.asarray(obs_error_var, np.float32)
    hxf = compute_hxf(xf_grid, ox, oy, oz, var_idx)
    R_t = aoei(yo, hxf, R0)
    xa  = _letkf_step(xf_grid, hxf, yo, R_t, ox, oy, oz, loc_scales)
    n_inf = int((R_t > R0).sum())
    print(f"  [AOEI]  inflated {n_inf}/{len(yo)} obs  "
          f"R_tilde={R_t.mean():.2f}  R0={R0.mean():.2f}")
    return dict(
        xa=xa,
        hxf=hxf,
        dep=(yo - hxf.mean(axis=1)).astype(np.float32),
        obs_error_raw=R0,
        obs_error=R_t,
    )


def atenkf_update(xf_grid, yo, obs_error_var, ox, oy, oz, loc_scales, var_idx,
                  ntemp, alpha_s):
    """
    ATEnKF: TEnKF with AOEI recomputed at every tempering step.

    At each step i: recompute H(x), apply AOEI -> R_tilde,
    then inflate R_tilde / alpha_i, run LETKF.

    Returns
    -------
    dict: xa, xatemp, hxfs, deps, obs_error_aoei, obs_error_eff,
          alpha_weights, obs_error_raw
    """
    steps = tempering_schedule(ntemp, alpha_s)
    R0    = np.asarray(obs_error_var, np.float32)
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)
    Nt   = len(steps)

    xatemp         = np.empty((nx, ny, nz, Ne, nvar, Nt + 1), dtype=np.float32, order="F")
    xatemp[..., 0] = xf_grid.astype(np.float32)
    hxfs           = np.empty((Nt, nobs, Ne), dtype=np.float32)
    deps           = np.empty((Nt, nobs),     dtype=np.float32)
    obs_error_aoei = np.empty((Nt, nobs),     dtype=np.float32)
    obs_error_eff  = np.empty((Nt, nobs),     dtype=np.float32)

    for it in range(Nt):
        hxf  = compute_hxf(xatemp[..., it], ox, oy, oz, var_idx)
        dep  = (yo - hxf.mean(axis=1)).astype(np.float32)
        R_t  = aoei(yo, hxf, R0)
        oerr = (R_t / steps[it]).astype(np.float32)
        hxfs[it] = hxf;  deps[it] = dep
        obs_error_aoei[it] = R_t;  obs_error_eff[it] = oerr
        n_inf = int((R_t > R0).sum())
        print(f"  [ATEnKF] step {it+1}/{Nt}  "
              f"alpha={steps[it]:.4f}  R_eff={oerr.mean():.2f}  "
              f"AOEI={n_inf}/{nobs}  |dep|={np.abs(dep).mean():.3f}")
        xatemp[..., it + 1] = _letkf_step(
            xatemp[..., it], hxf, yo, oerr, ox, oy, oz, loc_scales)

    return dict(xa=xatemp[..., -1], xatemp=xatemp, hxfs=hxfs, deps=deps,
                obs_error_aoei=obs_error_aoei, obs_error_eff=obs_error_eff,
                alpha_weights=steps, obs_error_raw=R0)
