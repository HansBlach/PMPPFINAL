"""LLR_monthly_jacobi.py"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isfile(
        os.path.join(_root, "kalman_common.py")):
    _root = os.path.dirname(_root)
for _p in (_here, _root,
           os.path.join(_root, "plots_tables_code"),
           os.path.join(_root, "plots_tables_code", "BIC"),
           os.path.join(_root, "plots_tables_code", "Simulations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import GetData as gd                                # noqa: F401  (parity)
import kalman_filter_jacobi as ld
import BIC_monthly_jacobi    as drv

from simulate_paths_monthly_jacobi import (
    filter_to_end_jacobi,
    simulate_state_paths_jacobi,
    compute_observations_jacobi,
    load_params_vec,            # mode-aware loader with c_tilde migration
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

M_FACTORS = 1

# Corridor pinning — match BIC_monthly_jacobi.py.

CORRIDOR_PINNED = True

RUN = {
    "baseline_llr":     True,
    "cdf_figure":       True,
    "slice_figure":     True,
    "rmse_insample":    True,   # In-sample EKF residual RMSE / diagnostics
}

SLICE_DEGS        = (1, 2, 3, 4, 5)
SLICE_GRID_PTS    = 41        # odd → MLE itself sits on the grid
SLICE_REL_RANGE   = 0.25
SLICE_MIN_HALF    = 0.05
SLICE_PINNED_HALF = 0.5


SLICE_LOGL_WINDOW = 500.0

SLICE_XFIT_TO_WINDOW = True

# Per-parameter slice-window overrides. Key is the label as rendered by
# `_greek` (e.g. r"$\lambda_{1}$"); value is a dict with optional "lo"
# and/or "hi" keys. Missing keys keep the default grid edge.
SLICE_RANGE_OVERRIDES = {
    r"$\lambda_{1}$":  {"lo": -0.35},
    r"$\theta_{1}$":   {"lo":  0.55},
    r"$\alpha_{1,0}$": {"hi":  3.0},
    r"$\beta_{1,1}$":  {"lo":  0.8},
}


SLICE_PTS_OVERRIDES = {
    r"$\alpha": 200,
}

N_PATHS_CDF   = 200
CDF_DEGS      = (1, 2, 3, 4, 5)
CDF_OBS_NOISE = True

CDF_XLIM_MODE   = "simulated_pct"
CDF_XLIM_PCT    = (1.0, 98.0)
CDF_XLIM_BUFFER = 0.05

SEED = 42


# ---------------------------------------------------------------
# Inherit panel / flag config from BIC_monthly_jacobi
# ---------------------------------------------------------------

OUT_DIR    = drv.OUT_DIR
PERIOD_TAG = drv.PERIOD_TAG
FIG_DIR    = os.path.join(OUT_DIR,
                            f"figures_llr_jacobi_{PERIOD_TAG}")
THESIS_DIR = os.path.join(OUT_DIR, f"figures_thesis_{PERIOD_TAG}")
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(THESIS_DIR, exist_ok=True)

TAU_REF = drv.TAU_REF
USE_WEEKLY_SAMPLING = drv.USE_WEEKLY_SAMPLING
DT      = drv.DT_EKF

CONTRACT_LABELS = list(drv.SUBSET_LABELS)


def params_filename(m, N_poly, per_factor_c=None):
    """Match BIC_monthly_jacobi.run_ekf_grid's saved-filename convention."""
    if per_factor_c is None:
        per_factor_c = drv.PER_FACTOR_C
    tag = "perfac" if per_factor_c else "global"
    return f"params_{PERIOD_TAG}_jacobi_m{m}_N{N_poly}_{tag}_lamratio.npy"


# ---------------------------------------------------------------
# Stage A / Stage B data load
# ---------------------------------------------------------------

def load_panels_and_residual():
    """Load Stage A + Stage B panels, fit Stage A seasonality, return the
    Stage-B residual + everything the simulators / EKF need."""
    print("Loading Stage A + Stage B panels via BIC_monthly_jacobi loaders ...")
    y_stagea, mat_stagea, del_stagea, tra_stagea = drv.load_stage_a_data()
    y_matrix, maturity, delivery, trading        = drv.load_stage_b_data()
    print(f"  Stage A: {y_stagea.shape[0]} days x {y_stagea.shape[1]} contracts "
          f"({tuple(drv.STAGE_A_LABELS)})")
    print(f"  Stage B: {y_matrix.shape[0]} days x {y_matrix.shape[1]} contracts "
          f"({tuple(CONTRACT_LABELS)})")

    if drv.START_DATE is not None:
        y_stagea, mat_stagea, del_stagea, tra_stagea, idx_a = \
            drv.slice_panel_after_date(drv.START_DATE, y_stagea, mat_stagea,
                                        del_stagea, tra_stagea)
        y_matrix, maturity, delivery, trading, idx_b = \
            drv.slice_panel_after_date(drv.START_DATE, y_matrix, maturity,
                                        delivery, trading)
        print(f"  Restricting to dates >= {drv.START_DATE}: "
              f"Stage A {idx_a} dropped, "
              f"Stage B {idx_b} dropped, "
              f"new Stage B size = {y_matrix.shape[0]}")

    price_scale  = float(y_stagea.mean())
    y_stagea_n   = y_stagea / price_scale
    y_norm       = y_matrix / price_scale
    print(f"  price_scale = {price_scale:.4f} EUR/MWh  "
          f"(computed on {y_stagea.shape[0]} pre-thin Stage A rows)")

    print("Fitting Stage A seasonality grid (on full post-cutoff data) ...")
    best = None
    for ah in drv.ANNUAL_GRID:
        info = ld.seasonality_bic(tra_stagea[:, 0],
                                   mat_stagea, del_stagea,
                                   y_stagea_n, ah)
        if best is None or info["BIC"] < best["BIC"]:
            best = info
    seas_beta = best["beta"]
    annual_h  = int(best["annual_h"])
    print(f"  best seasonality (a={annual_h})  BIC={best['BIC']:.1f}")

    n_t, n_c = maturity.shape
    _, S_hist, _ = ld.build_seasonality_matrix(
        trading[:, 0], maturity, delivery, y_norm,
        annual_h=annual_h,
    )
    g_bar   = (S_hist @ seas_beta).reshape(n_t, n_c)
    y_resid = y_norm - g_bar
    print(f"  Stage B residual (pre-thin): mean={y_resid.mean():+.5f}  "
          f"std={y_resid.std():.5f}")

    # Data already at the calibration cadence (weekly when
    # drv.USE_WEEKLY_SAMPLING, daily otherwise).
    print(f"  Calibration cadence: "
          f"{'weekly ISO-Mon' if drv.USE_WEEKLY_SAMPLING else 'daily'}; "
          f"Stage B size = {y_matrix.shape[0]}, DT = {DT:.6f}")

    return dict(
        y_matrix=y_matrix, maturity=maturity, delivery=delivery, trading=trading,
        y_norm=y_norm, y_resid=y_resid, g_bar=g_bar,
        price_scale=price_scale,
        seas_beta=seas_beta, annual_h=annual_h,
    )


# ---------------------------------------------------------------
# Re-evaluate helpers
# ---------------------------------------------------------------

def evaluate_logL(params_vec, y_resid, maturity, delivery, dt, m, N_poly,
                   per_factor_c=None):
    if per_factor_c is None:
        per_factor_c = drv.PER_FACTOR_C
    nll = ld.EKF_MLE(
        params_vec, y_resid, maturity, delivery, dt, N_poly, m,
        tau_ref=TAU_REF, per_factor_c=per_factor_c,
    )
    return -float(nll)


# ---------------------------------------------------------------
# Parameter report
# ---------------------------------------------------------------

def params_to_row(params_vec, m, N_poly, prefix="", per_factor_c=None):
    """Unpack a Jacobi `params_vec` and return a flat dict of all parameters."""
    if params_vec is None:
        return {}
    if per_factor_c is None:
        per_factor_c = drv.PER_FACTOR_C
    try:
        p = ld.unpack_Jacobi(params_vec, m, N_poly=N_poly,
                              per_factor_c=per_factor_c)
    except Exception:
        return {}
    row = {}
    def put(k, v): row[f"{prefix}{k}"] = float(v)

    for i, k in enumerate(p.kappa):
        put(f"kappa_{i}", k)
    for i, t in enumerate(p.theta):
        put(f"theta_{i}", t)
    for i, l in enumerate(p.lam):
        put(f"lam_{i}", l)
    for i, s in enumerate(p.sigma):
        put(f"sigma_{i}", s)
    put("p_delta", p.p_delta)
    # c_tilde / c amplitudes — always present in the new layout. In
    # per-factor mode there is one per factor; in global mode the
    # broadcast value is reported under c[0..m].
    c_vec = p.c                          # shape (m,) after broadcast
    if p.c_tilde is not None:
        ct = np.asarray(p.c_tilde).reshape(-1)
        for j, v in enumerate(ct):
            put(f"c_tilde_{j}", v)
    for i in range(m):
        put(f"c_{i}", c_vec[i])
    if p.k > 0:
        alpha_m = p.alpha          # (m, k)
        beta_m  = p.beta           # (m, k)
        for i in range(m):
            for j in range(p.k):
                put(f"alpha_tilde_{i}_{j}", p.alpha_tilde[i, j])
                put(f"beta_tilde_{i}_{j}",  p.beta_tilde [i, j])
                put(f"alpha_{i}_{j}",       alpha_m[i, j])
                put(f"beta_{i}_{j}",        beta_m [i, j])
    put("p_e", p.p_e)
    return row


def n_active_poly_params(N_poly):
    """Effective free polynomial-map parameters per degree (Jacobi sum mode)."""
    k      = ld.k_from_N(N_poly)
    k_free = ld.k_free_from_N(N_poly)
    return (k_free + k) * M_FACTORS


def llr_test(logL_restricted, logL_full, df):
    LR = 2.0 * (logL_full - logL_restricted)
    if not np.isfinite(LR):
        return float("nan"), float("nan"), df
    if LR < 0:
        return LR, 1.0, df
    p = float(chi2.sf(LR, df))
    return LR, p, df


# ---------------------------------------------------------------
# Block 1 — baseline LLR
# ---------------------------------------------------------------

def run_baseline_llr(data):
    print("\n=== Block 1: baseline LLR () ===")
    y_resid     = data["y_resid"]
    maturity    = data["maturity"]
    delivery    = data["delivery"]
    price_scale = data["price_scale"]

    # BIC penalty sample size — matches BIC_monthly_jacobi.run_ekf_grid.
    n_obs_bic = int(y_resid.shape[0] * y_resid.shape[1]
                    - np.isnan(y_resid).sum())

    rows  = []
    cells = {}
    pf_c  = drv.PER_FACTOR_C
    for N in (1, 2, 3, 4, 5):
        pf_path, v = load_params_vec(M_FACTORS, N, per_factor_c=pf_c)
        if v is None:
            new_name = params_filename(M_FACTORS, N, per_factor_c=pf_c)
            print(f"  [skip] no params file for m={M_FACTORS}, N={N}. "
                  f"Looked for {new_name}.")
            continue

        # The migration logic in `load_params_vec` already guarantees
        # v.size == num_params_ld(...) with c_tilde padded to zero for
        # legacy files, so the old-format size mismatch can no longer
        # be a silent failure mode.
        expected_k = ld.num_params_ld(M_FACTORS, N, per_factor_c=pf_c)
        if v.shape[0] != expected_k:
            print(f"  [skip] {os.path.basename(pf_path)}: post-migration "
                  f"length {v.shape[0]} != num_params_ld(m={M_FACTORS}, "
                  f"N={N}, per_factor_c={pf_c}) = {expected_k}. "
                  f"The file may correspond to a different mode; "
                  f"re-run BIC_monthly_jacobi to regenerate.")
            continue

        logL = evaluate_logL(v, y_resid, maturity, delivery,
                              DT, M_FACTORS, N, per_factor_c=pf_c)
        n_eff = n_active_poly_params(N)

        k_full = expected_k
        if CORRIDOR_PINNED:
            _c_size  = M_FACTORS if pf_c else 1
            n_pinned = 1 + _c_size                # p_delta + c_tilde slots
        else:
            n_pinned = 0
        k_eff = k_full - n_pinned
        bic   = k_eff * np.log(n_obs_bic) - 2.0 * logL

        # Summed (pooled across maturities) in-sample RMSE in EUR/MWh via
        # the same EKF diagnostics pass run_in_sample_rmse uses.
        rmse_eur = float("nan")
        try:
            params = ld.unpack_Jacobi(v, M_FACTORS, N_poly=N,
                                       per_factor_c=pf_c)
            theta_P = params.theta + params.lam / params.kappa
            a_jac = 2.0 * params.kappa * theta_P         / params.sigma ** 2
            b_jac = 2.0 * params.kappa * (1.0 - theta_P) / params.sigma ** 2
            x0 = theta_P.reshape(-1, 1)
            P0 = np.diag(theta_P * (1.0 - theta_P) / (a_jac + b_jac + 1.0))
            diag = _ekf_run_with_diagnostics_jacobi(
                params, x0, P0,
                y_resid, maturity, delivery, DT, N_pricing=N,
            )
            rmse_eur = float(_safe_rmse(diag["resid"]) * price_scale)
        except Exception as exc:
            print(f"  [m={M_FACTORS} N={N}] RMSE pass failed ({exc})")

        row = {"m": M_FACTORS, "N_poly": N,
               "n_active_poly_params": n_eff,
               "k_total":   k_full,
               "k_pinned":  n_pinned,
               "k":         k_eff,
               "n_obs":     n_obs_bic,
               "logL":      logL,
               "BIC":       bic,
               "rmse_eur":  rmse_eur,
               "per_factor_c": pf_c,
               "params_file": os.path.basename(pf_path)}
        row.update(params_to_row(v, M_FACTORS, N, per_factor_c=pf_c))
        rows.append(row)
        cells[N] = dict(params_vec=v, logL=logL, n_eff=n_eff,
                         BIC=bic, rmse_eur=rmse_eur,
                         k_total=k_full, k_pinned=n_pinned, k=k_eff)
        if n_pinned > 0:
            print(f"  m={M_FACTORS}  N={N}  logL={logL:.4f}  "
                  f"BIC={bic:.2f}  RMSE={rmse_eur:.3f} EUR/MWh  "
                  f"(n_active_poly_params={n_eff}, "
                  f"k_total={k_full}, k_pinned={n_pinned}, k={k_eff})")
        else:
            print(f"  m={M_FACTORS}  N={N}  logL={logL:.4f}  "
                  f"BIC={bic:.2f}  RMSE={rmse_eur:.3f} EUR/MWh  "
                  f"(n_active_poly_params={n_eff}, k_total={k_full})")

    pair_rows = []
    nested_pairs = [
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 5),
    ]
    for (Nr, Nf) in nested_pairs:
        if Nr in cells and Nf in cells:
            df  = cells[Nf]["n_eff"] - cells[Nr]["n_eff"]
            LR, p, _ = llr_test(cells[Nr]["logL"], cells[Nf]["logL"], df)
            pair_rows.append({"restricted": Nr, "full": Nf,
                              "df": df,
                              "logL_R": cells[Nr]["logL"],
                              "logL_F": cells[Nf]["logL"],
                              "LR": LR, "p_value": p})
            print(f"  LLR  N={Nr} (restricted)  vs  N={Nf} (full):  "
                  f"LR={LR:.4f}  df={df}  p={p:.3e}")

    base_csv = os.path.join(OUT_DIR,
                             f"llr_jacobi_baseline_m{M_FACTORS}_{PERIOD_TAG}.csv")
    out = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    with open(base_csv, "w") as f:
        f.write("# Per-cell log-likelihoods\n")
        out.to_csv(f, index=False)
        f.write("\n# Pairwise LLR\n")
        pair_df.to_csv(f, index=False)
    print(f"  saved -> {base_csv}")
    return cells


# ---------------------------------------------------------------
# Block X — 1-D logL slice figure
# ---------------------------------------------------------------

def _greek(name, i=None, j=None):
    """LaTeX-rendered parameter label for matplotlib."""
    if i is None and j is None:
        return f"${name}$"
    if j is None:
        return f"${name}_{{{i}}}$"
    return f"${name}_{{{i},{j}}}$"


SLICE_SKIP_PATTERNS = ()


def param_labels(m, N_poly, per_factor_c=None):
    """Return ordered Greek-letter labels matching the current pack/unpack layout."""
    if per_factor_c is None:
        per_factor_c = drv.PER_FACTOR_C
    k      = ld.k_from_N(N_poly)
    k_free = ld.k_free_from_N(N_poly)
    labels = []
    for i in range(m):
        labels.append(_greek(r"\kappa",  i if m > 1 else None))
    for i in range(m):
        labels.append(_greek(r"\theta",  i if m > 1 else None))
    for i in range(m):
        if m > 1:
            labels.append(rf"$\lambda_{{r,{i}}}$")
        else:
            labels.append(r"$\lambda_r$")
    for i in range(m):
        labels.append(_greek(r"\sigma",  i if m > 1 else None))
    labels.append(_greek(r"\delta"))
    c_size = m if per_factor_c else 1
    if c_size == 1:
        labels.append(_greek("c"))
    else:
        for i in range(c_size):
            labels.append(_greek("c", i))
    if k > 0:
        for i in range(m):
            for j in range(k_free):
                if m > 1 or k_free > 1:
                    labels.append(_greek(r"\alpha", i, j))
                else:
                    labels.append(_greek(r"\alpha"))
        for i in range(m):
            for j in range(k):
                if m > 1 or k > 1:
                    labels.append(_greek(r"\beta", i, j))
                else:
                    labels.append(_greek(r"\beta"))
    labels.append(r"$p_e$")
    return labels


def _is_skipped(label):
    return any(label == p or label.startswith(p + "[")
                for p in SLICE_SKIP_PATTERNS)


def _slice_n_pts(label):
    """Grid-point count for a slice panel, honouring SLICE_PTS_OVERRIDES."""
    for prefix, n in SLICE_PTS_OVERRIDES.items():
        if label.startswith(prefix):
            return int(n)
    return SLICE_GRID_PTS


def make_slice_grid(lo, hi, mle, n_pts=SLICE_GRID_PTS,
                     rel_range=SLICE_REL_RANGE,
                     min_half_frac=SLICE_MIN_HALF):
    if abs(hi - lo) < 1e-12:
        return None
    half = max(rel_range * abs(mle), min_half_frac * (hi - lo))
    a = max(lo, mle - half)
    b = min(hi, mle + half)
    if b - a < 1e-10:
        return None
    return np.linspace(a, b, n_pts)


def run_slice_figure(data, baseline_cells):
    print("\n=== Block: 1-D logL slice figure (Jacobi) ===")
    degs = [d for d in SLICE_DEGS if d in baseline_cells]
    if not degs:
        print("  ! No baseline cells available for slice figure. Skipping.")
        return None

    y_resid  = data["y_resid"]
    maturity = data["maturity"]
    delivery = data["delivery"]

    out_paths = []
    pf_c = drv.PER_FACTOR_C
    for N in degs:
        v_mle  = np.asarray(baseline_cells[N]["params_vec"], dtype=float).copy()
        bounds = ld.make_bounds(M_FACTORS, N, per_factor_c=pf_c)
        labels = param_labels(M_FACTORS, N, per_factor_c=pf_c)
        if not (len(v_mle) == len(bounds) == len(labels)):
            print(f"  [skip N={N}] vector/bound/label length mismatch "
                  f"(v={len(v_mle)}, b={len(bounds)}, l={len(labels)})")
            continue

        logL_mle = evaluate_logL(v_mle, y_resid, maturity, delivery,
                                  DT, M_FACTORS, N, per_factor_c=pf_c)
        print(f"\n  m={M_FACTORS}  N={N}   logL(MLE) = {logL_mle:.4f}   "
              f"({len(v_mle)} params, "
              f"{SLICE_GRID_PTS} pts each → "
              f"{len(v_mle) * SLICE_GRID_PTS} EKF evaluations)")

        slices = []
        t0 = time.time()
        for i in range(len(v_mle)):
            if _is_skipped(labels[i]):
                continue
            lo, hi = bounds[i]
            mle_val = v_mle[i]
            n_pts   = _slice_n_pts(labels[i])
            grid    = make_slice_grid(lo, hi, mle_val, n_pts=n_pts)
            if grid is None:
                # Pinned parameter (lo == hi) — skip the panel entirely.
                continue
            override = SLICE_RANGE_OVERRIDES.get(labels[i])
            if override is not None:
                lo_use = float(override.get("lo", grid[0]))
                hi_use = float(override.get("hi", grid[-1]))
                grid   = np.linspace(lo_use, hi_use, n_pts)
            status  = ""
            width = hi - lo
            if width > 0:
                if mle_val - lo < 0.01 * width:
                    status = "lower"
                elif hi - mle_val < 0.01 * width:
                    status = "upper"
            ll = np.full_like(grid, np.nan, dtype=float)
            for j, val in enumerate(grid):
                x = v_mle.copy()
                x[i] = float(val)
                ll[j] = evaluate_logL(x, y_resid, maturity, delivery,
                                       DT, M_FACTORS, N, per_factor_c=pf_c)

            ll[ll < -1e6] = np.nan
            slices.append((labels[i], grid, ll, mle_val, status))
        print(f"    ... swept in {time.time() - t0:.1f}s")

        n = len(slices)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                   figsize=(3.4 * ncols, 2.5 * nrows),
                                   squeeze=False)
        for idx, (label, grid, ll, mle_val, status) in enumerate(slices):
            ax = axes[idx // ncols][idx % ncols]
            if status == "pinned":
                colour = "#888888"
            elif status in ("lower", "upper"):
                colour = "#D62728"
            else:
                colour = "#1F77B4"
            ax.plot(grid, ll, "-", color=colour, lw=1.2)
            ax.axvline(mle_val, color="#D62728", ls="--", lw=0.8)
            j_mle = int(np.argmin(np.abs(grid - mle_val)))
            ax.scatter([grid[j_mle]], [ll[j_mle]],
                        color="#D62728", s=18, zorder=5)
            tag = ""
            if status == "pinned":
                tag = f"  [pinned at {mle_val:.4g}]"
            elif status in ("lower", "upper"):
                tag = f"  [at {status} bound]"

            if SLICE_LOGL_WINDOW is not None:
                finite = ll[np.isfinite(ll)]
                if finite.size:
                    y_hi   = float(np.nanmax(finite))
                    y_lo   = max(float(np.nanmin(finite)),
                                 y_hi - float(SLICE_LOGL_WINDOW))
                    pad    = 0.05 * max(y_hi - y_lo, 1e-9)
                    ax.set_ylim(y_lo - pad, y_hi + pad)

                    if SLICE_XFIT_TO_WINDOW:
                        inside = np.where(ll >= y_lo)[0]
                        if inside.size:
                            i0 = max(int(inside.min()) - 1, 0)
                            i1 = min(int(inside.max()) + 1, len(grid) - 1)
                            if grid[i1] > grid[i0]:
                                ax.set_xlim(grid[i0], grid[i1])
            ax.set_title(label + tag, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(len(slices), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.tight_layout()
        out_path = os.path.join(
            FIG_DIR,
            f"slice_jacobi_m{M_FACTORS}_N{N}_{PERIOD_TAG}.png",
        )
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    saved -> {out_path}")
        out_paths.append(out_path)

    return out_paths


# ---------------------------------------------------------------
# Block 4 — CDF figure
# ---------------------------------------------------------------

def _simulate_inhistory_prices_jacobi(params, data, N_poly, n_paths, rng):
    """500 in-history paths starting from EKF posterior X[0], folded into
    prices via compute_observations_jacobi + Stage A seasonality."""
    y_resid     = data["y_resid"]
    maturity    = data["maturity"]
    delivery    = data["delivery"]
    trading     = data["trading"]
    seas_beta   = data["seas_beta"]
    annual_h    = data["annual_h"]
    price_scale = data["price_scale"]

    n_days, n_c = maturity.shape

    theta_P = params.theta + params.lam / params.kappa
    a_jac = 2.0 * params.kappa * theta_P          / params.sigma ** 2
    b_jac = 2.0 * params.kappa * (1.0 - theta_P)  / params.sigma ** 2
    x0    = theta_P.reshape(-1, 1)
    P0    = np.diag(theta_P * (1.0 - theta_P) / (a_jac + b_jac + 1.0))
    _, _, state_filt, state_cov_d, *_unused = filter_to_end_jacobi(
        params, x0, P0, y_resid, maturity, delivery,
        DT, N_pricing=N_poly,
    )
    x_start = np.asarray(state_filt[0]).reshape(-1)
    P_start = np.diag(np.maximum(state_cov_d[0], 0.0))

    state_paths = simulate_state_paths_jacobi(
        params, x_start, P_start, DT, n_steps=n_days,
        n_paths=n_paths, rng=rng, sample_init=True,
    )
    state_paths_obs = state_paths[:, 1:, :]

    y_norm_sim = compute_observations_jacobi(
        params, state_paths_obs,
        T_step=maturity, delta_step=delivery, N_pricing=N_poly,
    )

    if CDF_OBS_NOISE:
        R_diag = ld.precompute_R(
            maturity, params.p_e, tau_ref=ld.TAU_REF_DEFAULT,
        )
        y_norm_sim = (y_norm_sim
                      + rng.standard_normal((n_paths, n_days, n_c))
                        * np.sqrt(R_diag)[None, :, :])

    _, S_hist, _ = ld.build_seasonality_matrix(
        np.asarray(trading[:, 0]), maturity, delivery,
        np.zeros((n_days, n_c)),
        annual_h=annual_h,
    )
    g_bar = (S_hist @ seas_beta).reshape(n_days, n_c)
    sim_prices_eur = price_scale * (g_bar[None, :, :] + y_norm_sim)
    return sim_prices_eur


def _compute_xlim(observed, simulated):
    if CDF_XLIM_MODE == "none":
        return None
    if CDF_XLIM_MODE == "simulated_pct":
        sample = np.asarray(simulated, dtype=float).ravel()
    elif CDF_XLIM_MODE == "observed_pct":
        sample = np.asarray(observed, dtype=float).ravel()
    elif CDF_XLIM_MODE == "pooled_pct":
        sample = np.concatenate([
            np.asarray(observed,  dtype=float).ravel(),
            np.asarray(simulated, dtype=float).ravel(),
        ])
    else:
        return None
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return None
    lo, hi = np.percentile(sample, CDF_XLIM_PCT)
    width  = max(hi - lo, 1e-12)
    buf    = float(CDF_XLIM_BUFFER) * width
    return (float(lo - buf), float(hi + buf))


def _ecdf_pair(observed, simulated):
    obs = np.asarray(observed,  dtype=float).ravel()
    sim = np.asarray(simulated, dtype=float).ravel()
    obs = obs[np.isfinite(obs)]
    sim = sim[np.isfinite(sim)]
    grid = np.unique(np.concatenate([obs, sim]))
    obs_s = np.sort(obs)
    sim_s = np.sort(sim)
    F_obs = np.searchsorted(obs_s, grid, side="right") / len(obs_s)
    F_sim = np.searchsorted(sim_s, grid, side="right") / len(sim_s)
    ks    = float(np.max(np.abs(F_sim - F_obs))) if len(grid) else float("nan")
    return grid, F_obs, F_sim, ks


def _save_thesis_histograms(degs, sim_by_deg, y_matrix, n_bins=50,
                              layout=None, suffix=None):
    """Plot pooled price-density histograms for `degs`."""
    if layout is None:
        layout = (1, len(degs))
    nrows, ncols = layout
    if nrows * ncols != len(degs):
        raise ValueError(
            f"_save_thesis_histograms: layout {layout} can fit "
            f"{nrows * ncols} panels but received {len(degs)} degrees.")

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.6 * ncols, 3.6 * nrows),
                              sharey=False)
    axes_flat = np.atleast_1d(axes).reshape(-1)

    pooled_ks = {}
    for idx, N in enumerate(degs):
        ax  = axes_flat[idx]
        sim = sim_by_deg[N].ravel()
        obs = y_matrix.ravel()
        obs = obs[np.isfinite(obs)]
        sim = sim[np.isfinite(sim)]

        _, _, _, ks = _ecdf_pair(obs, sim)
        pooled_ks[N] = ks

        xlim = _compute_xlim(obs, sim)
        if xlim is None:
            lo = float(min(obs.min(), sim.min()))
            hi = float(max(obs.max(), sim.max()))
        else:
            lo, hi = xlim
        bins = np.linspace(lo, hi, n_bins + 1)

        ax.hist(obs, bins=bins, density=True, alpha=0.45,
                color="#1F77B4", label="observed",
                edgecolor="white", linewidth=0.3)
        ax.hist(sim, bins=bins, density=True, alpha=0.45,
                color="#D62728", label=f"simulated (n={N_PATHS_CDF})",
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"deg {N}")
        ax.set_xlabel("EUR / MWh")
        ax.set_ylabel("density")
        ax.set_xlim(lo, hi)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Pooled price density ()  -  "
        f"m={M_FACTORS}  -  {N_PATHS_CDF} paths",
        fontsize=11, y=1.02)
    fig.tight_layout()
    suffix_str = f"_{suffix}" if suffix else ""
    out_path = os.path.join(THESIS_DIR,
                             f"histogram_jacobi_m{M_FACTORS}_{PERIOD_TAG}"
                             f"{suffix_str}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return out_path, pooled_ks


def _save_thesis_diff(degs, sim_by_deg, y_matrix):
    fig, axes = plt.subplots(1, len(degs),
                              figsize=(4.6 * len(degs), 2.8),
                              sharey=False)
    if len(degs) == 1:
        axes = [axes]

    for ci, N in enumerate(degs):
        ax  = axes[ci]
        obs = y_matrix.ravel()
        sim = sim_by_deg[N].ravel()
        grid, F_obs, F_sim, ks = _ecdf_pair(obs, sim)
        xlim = _compute_xlim(obs, sim)

        ax.fill_between(grid, 0.0, np.abs(F_sim - F_obs),
                        color="#7F7F7F", alpha=0.55, step="post")
        ax.axhline(ks, color="#D62728", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(f"deg {N}")
        ax.set_xlabel("EUR / MWh")
        ax.set_ylabel("|F_sim - F_obs|")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Pooled CDF difference ()  -  m={M_FACTORS}",
        fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(THESIS_DIR,
                             f"cdf_diff_jacobi_m{M_FACTORS}_{PERIOD_TAG}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return out_path


def run_cdf_figure(data, baseline_cells):
    print("\n=== Block 4: CDF overlay figure () ===")
    degs = [d for d in CDF_DEGS if d in baseline_cells]
    if not degs:
        print("  ! No baseline cells available for CDF figure. Skipping.")
        return None

    y_matrix = data["y_matrix"]
    n_c      = y_matrix.shape[1]

    rng = np.random.default_rng(SEED)
    sim_by_deg = {}
    pf_c = drv.PER_FACTOR_C
    for N in degs:
        v = baseline_cells[N]["params_vec"]
        params = ld.unpack_Jacobi(v, M_FACTORS, N_poly=N, per_factor_c=pf_c)
        print(f"  simulating {N_PATHS_CDF} in-history paths for N={N} ...")
        sim_by_deg[N] = _simulate_inhistory_prices_jacobi(
            params, data, N_poly=N, n_paths=N_PATHS_CDF,
            rng=np.random.default_rng(SEED + N),
        )
        print(f"    sim shape = {sim_by_deg[N].shape}")

    fig = plt.figure(figsize=(4.2 * len(degs), 3.4 * n_c))
    height_ratios = []
    for _ in range(n_c):
        height_ratios.extend([3.0, 1.0])
    gs = fig.add_gridspec(2 * n_c, len(degs),
                           height_ratios=height_ratios, hspace=0.07,
                           wspace=0.18)

    ks_summary = {}
    for ci, N in enumerate(degs):
        sim_eur = sim_by_deg[N]
        for ri in range(n_c):
            cname = CONTRACT_LABELS[ri]
            obs   = y_matrix[:, ri]
            sim   = sim_eur[:, :, ri].ravel()
            grid, F_obs, F_sim, ks = _ecdf_pair(obs, sim)
            ks_summary[(N, cname)] = ks
            xlim = _compute_xlim(obs, sim)

            ax_cdf = fig.add_subplot(gs[2 * ri, ci])
            ax_d   = fig.add_subplot(gs[2 * ri + 1, ci], sharex=ax_cdf)
            ax_cdf.tick_params(labelbottom=False)

            ax_cdf.step(grid, F_obs, where="post",
                         color="#1F77B4", lw=1.2, label="observed")
            ax_cdf.step(grid, F_sim, where="post",
                         color="#D62728", lw=1.2,
                         label=f"sim (n={N_PATHS_CDF})")
            ax_cdf.set_ylim(0, 1.05)
            ax_cdf.set_ylabel(f"{cname}  CDF")
            ax_cdf.set_title(f"deg {N}  |  {cname}  |  KS = {ks:.3f}")
            ax_cdf.legend(loc="lower right", frameon=False, fontsize=8)
            ax_cdf.grid(True, alpha=0.3)

            ax_d.fill_between(grid, 0.0, np.abs(F_sim - F_obs),
                              color="#7F7F7F", alpha=0.45, step="post")
            ax_d.axhline(ks, color="#D62728", ls="--", lw=0.8,
                         alpha=0.6, label=f"KS = {ks:.3f}")
            ax_d.set_ylabel(f"{cname}  |dF|")
            ax_d.set_xlabel("EUR / MWh")
            ax_d.legend(loc="upper right", frameon=False, fontsize=8)
            ax_d.grid(True, alpha=0.3)

            if xlim is not None:
                ax_cdf.set_xlim(*xlim)
                ax_d.set_xlim(*xlim)
                sim_max = float(np.nanmax(sim)) if sim.size else float("nan")
                if np.isfinite(sim_max) and sim_max > xlim[1]:
                    ax_cdf.text(
                        0.98, 0.55,
                        f"sim max = {sim_max:.0f}  (>{xlim[1]:.0f})",
                        transform=ax_cdf.transAxes,
                        ha="right", va="top", fontsize=7, color="#D62728",
                        alpha=0.85,
                    )

    fig.suptitle(
        f"Empirical vs simulated CDFs ()  -  m={M_FACTORS}  "
        f"({', '.join(CONTRACT_LABELS)})  -  {N_PATHS_CDF} paths",
        fontsize=11, y=1.0)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR,
                             f"cdf_overlay_jacobi_m{M_FACTORS}_{PERIOD_TAG}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")

    print("\n  KS distance per maturity (sim vs observed CDF):")
    header = "    " + " " * 8 + "  ".join(f"{c:>10s}" for c in CONTRACT_LABELS)
    print(header)
    for N in degs:
        row = f"    deg {N}  " + "  ".join(
            f"{ks_summary[(N, c)]:>10.4f}" for c in CONTRACT_LABELS)
        print(row)

    # Histograms: produce two 2×2 panels 
    pooled_ks = {}
    canonical_groups = [
        ([1, 2, 3, 4], "deg1_4"),
        ([2, 3, 4, 5], "deg2_5"),
    ]
    full_groups = [(g, suf) for (g, suf) in canonical_groups
                    if all(d in degs for d in g)]
    if full_groups:
        for group, suf in full_groups:
            _, ks_sub = _save_thesis_histograms(
                group, sim_by_deg, y_matrix,
                layout=(2, 2), suffix=suf)
            pooled_ks.update(ks_sub)

        missing_in_groups = [d for d in degs if d not in pooled_ks]
        if missing_in_groups:
            _, ks_extra = _save_thesis_histograms(
                missing_in_groups, sim_by_deg, y_matrix,
                suffix="extra")
            pooled_ks.update(ks_extra)
    else:
        _, pooled_ks = _save_thesis_histograms(degs, sim_by_deg, y_matrix)

    _save_thesis_diff(degs, sim_by_deg, y_matrix)
    print("\n  Pooled KS (all maturities, sim vs observed CDF):")
    for N in degs:
        print(f"    deg {N}  KS = {pooled_ks[N]:.4f}")
    return out_path


# ---------------------------------------------------------------
# In-sample RMSE / diagnostics
# ---------------------------------------------------------------

RMSE_DEGS    = (1, 2, 3, 4, 5)
RMSE_NPZ_DIR = os.path.join(OUT_DIR,
                              f"predictions_jacobi_in_sample_{PERIOD_TAG}")


def _ekf_run_with_diagnostics_jacobi(params, x0, P0,
                                      y_obs, T, delta, dt, N_pricing,
                                      tau_ref=ld.TAU_REF_DEFAULT):
    """Drive ld.EKF_step over the historical horizon AND collect diagnostics:"""
    n_steps, n_c = y_obs.shape
    m = len(params.kappa)

    theta_P = params.theta + params.lam / params.kappa
    a = 2.0 * params.kappa * theta_P          / params.sigma ** 2
    b = 2.0 * params.kappa * (1.0 - theta_P)  / params.sigma ** 2

    p_T = ld.build_poly_nd(params, m, N_pricing)
    G   = ld.infinitesimal_generator_jacobi(
        -params.kappa, params.kappa * params.theta,
        params.sigma, N_pricing,
    )
    Mp_all, _ = ld._precompute_Mp(G, p_T, T, delta)
    R_all     = ld.precompute_R(T, params.p_e, tau_ref=tau_ref)

    resid       = np.full((n_steps, n_c), np.nan)
    y_pred      = np.full((n_steps, n_c), np.nan)
    state_filt  = np.full((n_steps, m), np.nan)
    state_cov_d = np.full((n_steps, m), np.nan)
    S_diag      = np.full((n_steps, n_c), np.nan)

    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    P = np.atleast_2d(P0)
    log_lik = 0.0

    for t in range(n_steps):
        try:
            x_prior, Q = ld.f_Jacobi(params, x, dt)
            A_jac      = ld.A_Jacobi(params, x, dt)
            P_prior    = A_jac @ P @ A_jac.T + Q

            x_vec  = np.asarray(x_prior).flatten()
            H_x    = ld.build_H(x_vec,  N_pricing)
            dH_x   = ld.build_dH(x_vec, N_pricing)
            h_vals = np.array([H_x  @ Mp for Mp in Mp_all[t]])
            H_jac  = np.array([dH_x @ Mp for Mp in Mp_all[t]])
            if H_jac.ndim == 1:
                H_jac = H_jac.reshape(-1, 1)

            R_mat = np.diag(R_all[t].astype(float))
            S     = H_jac @ P_prior @ H_jac.T + R_mat
            if not np.all(np.isfinite(S)):
                raise ValueError("Non-finite S")

            K           = P_prior @ H_jac.T @ np.linalg.inv(S)
            resid_full  = (y_obs[t] - h_vals).flatten()
            x_post      = x_vec + K @ resid_full
            eps         = 1e-4
            x_post      = np.clip(x_post, eps, 1.0 - eps)
            I_KH        = np.eye(m) - K @ H_jac
            P_post      = I_KH @ P_prior @ I_KH.T + K @ R_mat @ K.T
            for i in range(m):
                P_post[i, i] = min(P_post[i, i], ld.P_POST_DIAG_CAP)
            x = x_post.reshape(-1, 1)
            P = P_post

            sign, log_det = np.linalg.slogdet(S)
            if sign <= 0 or not np.isfinite(log_det):
                raise ValueError("Non-PSD S")
            n_obs_t  = len(resid_full)
            log_lik += -0.5 * (n_obs_t * np.log(2 * np.pi) + log_det
                               + resid_full @ np.linalg.solve(S, resid_full))
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(f"     EKF step failed at t={t}: {exc}")
            break

        resid [t] = resid_full
        y_pred[t] = h_vals
        S_diag[t] = np.diag(S)
        state_filt [t] = x.flatten()
        state_cov_d[t] = np.diag(P)

    return dict(
        resid=resid, y_pred=y_pred,
        state_filt=state_filt, state_cov_diag=state_cov_d,
        S_diag=S_diag, log_lik=log_lik,
    )


def _safe_rmse(arr):
    sq = np.asarray(arr, dtype=float) ** 2
    if np.all(np.isnan(sq)):
        return float("nan")
    return float(np.sqrt(np.nanmean(sq)))


def _summarize_rmse(resid, price_scale):
    n_c = resid.shape[1]
    rmse_norm = _safe_rmse(resid)
    return dict(
        rmse_norm=rmse_norm,
        rmse_eur =rmse_norm * price_scale,
        per_contract_norm=np.array([_safe_rmse(resid[:, c]) for c in range(n_c)]),
        per_contract_eur =np.array([_safe_rmse(resid[:, c]) for c in range(n_c)])
                          * price_scale,
    )


def _coverage_rate(resid, S_diag):
    sd = np.sqrt(S_diag)
    inside = np.abs(resid) < 2.0 * sd
    valid  = ~np.isnan(resid) & ~np.isnan(S_diag)
    if valid.sum() == 0:
        return float("nan")
    return float(inside[valid].mean())


def _sigma_vs_rmse(resid, S_diag):
    pred_sd = np.sqrt(np.nanmean(S_diag))
    real_sd = _safe_rmse(resid)
    if real_sd == 0 or np.isnan(real_sd):
        ratio = float("nan")
    else:
        ratio = pred_sd / real_sd
    return dict(predicted_sd=float(pred_sd),
                realized_rmse=float(real_sd),
                ratio=float(ratio))


def run_in_sample_rmse(data, baseline_cells=None):
    """In-sample RMSE / bias / coverage / sigma diagnostics. Walks RMSE_DEGS,"""
    print("\n=== In-sample RMSE / EKF diagnostics (sum) ===")
    os.makedirs(RMSE_NPZ_DIR, exist_ok=True)

    y_matrix    = data["y_matrix"]
    maturity    = data["maturity"]
    delivery    = data["delivery"]
    trading     = data["trading"]
    y_resid     = data["y_resid"]
    g_bar       = data["g_bar"]
    price_scale = data["price_scale"]

    n_c = maturity.shape[1]

    rmse_rows  = []
    bias_rows  = []
    sigma_rows = []
    cov_rows   = []
    pf_c = drv.PER_FACTOR_C

    for N in RMSE_DEGS:
        pf_path, v = load_params_vec(M_FACTORS, N, per_factor_c=pf_c)
        if v is None:
            print(f"  [m={M_FACTORS} N={N}] no params file (new or legacy), "
                  f"skipping")
            continue
        try:
            params = ld.unpack_Jacobi(v, M_FACTORS, N_poly=N,
                                       per_factor_c=pf_c)
        except Exception as exc:
            print(f"  [m={M_FACTORS} N={N}] unpack failed ({exc}), skipping")
            continue
        theta_P = params.theta + params.lam / params.kappa
        a = 2.0 * params.kappa * theta_P          / params.sigma ** 2
        b = 2.0 * params.kappa * (1.0 - theta_P)  / params.sigma ** 2
        x0 = theta_P.reshape(-1, 1)
        P0 = np.diag(theta_P * (1.0 - theta_P) / (a + b + 1.0))

        t0 = time.time()
        print(f"  [m={M_FACTORS} N={N}]  filtering ...", flush=True)
        out = _ekf_run_with_diagnostics_jacobi(
            params, x0, P0,
            y_resid, maturity, delivery, DT, N_pricing=N,
        )
        elapsed = time.time() - t0

        r          = out["resid"]
        y_pred_eur = price_scale * (g_bar + out["y_pred"])

        npz_path = os.path.join(
            RMSE_NPZ_DIR,
            f"predictions_m{M_FACTORS}_N{N}_in_sample.npz")
        np.savez(
            npz_path,
            y_obs=y_matrix, y_pred=y_pred_eur,
            resid=r, resid_eur=r * price_scale,
            state_filt=out["state_filt"],
            state_cov_diag=out["state_cov_diag"],
            S_diag=out["S_diag"],
            trading=trading,
            maturity=maturity, delivery=delivery,
            g_bar=g_bar, price_scale=price_scale,
            log_lik=out["log_lik"],
        )

        rs = _summarize_rmse(r, price_scale)
        row = {"m": M_FACTORS, "N_poly": N, "split": "in_sample",
               "logL": out["log_lik"],
               "rmse_norm": rs["rmse_norm"],
               "rmse_eur":  rs["rmse_eur"]}
        for c in range(n_c):
            row[f"rmse_{CONTRACT_LABELS[c]}_eur"] = rs["per_contract_eur"][c]
        rmse_rows.append(row)
        print(f"     RMSE (EUR/MWh) = {rs['rmse_eur']:.3f}  "
              f"per-contract = "
              f"{['%.3f' % v for v in rs['per_contract_eur']]}  "
              f"({elapsed:.1f}s)")

        bias_norm = np.nanmean(r, axis=0)
        for c in range(n_c):
            bias_rows.append({"m": M_FACTORS, "N_poly": N,
                              "contract": CONTRACT_LABELS[c],
                              "bias_norm": float(bias_norm[c]),
                              "bias_eur":  float(bias_norm[c] * price_scale)})

        sigma_rows.append({"m": M_FACTORS, "N_poly": N,
                            **_sigma_vs_rmse(r, out["S_diag"])})

        cov_rows.append({"m": M_FACTORS, "N_poly": N,
                          "coverage_2sigma":
                              _coverage_rate(r, out["S_diag"])})

    summary_csv  = os.path.join(OUT_DIR,
        f"rmse_insample_jacobi_m{M_FACTORS}_{PERIOD_TAG}.csv")
    bias_csv     = os.path.join(OUT_DIR,
        f"rmse_insample_jacobi_bias_m{M_FACTORS}_{PERIOD_TAG}.csv")
    sigma_csv    = os.path.join(OUT_DIR,
        f"rmse_insample_jacobi_sigma_m{M_FACTORS}_{PERIOD_TAG}.csv")
    coverage_csv = os.path.join(OUT_DIR,
        f"rmse_insample_jacobi_coverage_m{M_FACTORS}_{PERIOD_TAG}.csv")
    pd.DataFrame(rmse_rows).to_csv(summary_csv,  index=False)
    pd.DataFrame(bias_rows).to_csv(bias_csv,     index=False)
    pd.DataFrame(sigma_rows).to_csv(sigma_csv,   index=False)
    pd.DataFrame(cov_rows).to_csv(coverage_csv,  index=False)

    print(f"\n  Saved per-cell predictions to {RMSE_NPZ_DIR}")
    print(f"  Saved {summary_csv}")
    print(f"  Saved {bias_csv}")
    print(f"  Saved {sigma_csv}")
    print(f"  Saved {coverage_csv}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="LLR test framework for Jacobi PMPP polynomial degrees "
                    "(.")
    p.add_argument("--baseline", action="store_true",
                    help="Run the baseline LLR block.")
    p.add_argument("--cdf", action="store_true",
                    help="Run the CDF overlay figure block.")
    p.add_argument("--slice", action="store_true",
                    help="Run the 1-D logL slice figure.")
    p.add_argument("--rmse", action="store_true",
                    help="Run the in-sample RMSE / EKF diagnostics block.")
    p.add_argument("--all", action="store_true",
                    help="Run every block.")
    p.add_argument("--m", type=int, choices=(1, 2), default=None,
                    help="Override M_FACTORS for this run only.")
    return p.parse_args()


def _resolve_run_flags(args):
    if args.all:
        return {"baseline_llr": True, "cdf_figure": True,
                "slice_figure": True, "rmse_insample": True}
    any_block = (args.baseline or args.cdf or args.slice or args.rmse)
    if any_block:
        return {"baseline_llr":  bool(args.baseline),
                "cdf_figure":    bool(args.cdf),
                "slice_figure":  bool(args.slice),
                "rmse_insample": bool(args.rmse)}
    return dict(RUN)


def main():
    args = _parse_args()
    global M_FACTORS
    if args.m is not None:
        M_FACTORS = int(args.m)
    run_flags = _resolve_run_flags(args)

    print(f"=== LLR_monthly_jacobi () :  "
          f"M_FACTORS={M_FACTORS} ===")
    print(f"OUT_DIR = {OUT_DIR}")
    print(f"FIG_DIR = {FIG_DIR}")
    print(f"RUN     = {run_flags}")
    data = load_panels_and_residual()

    baseline_cells = {}
    if run_flags["baseline_llr"]:
        baseline_cells = run_baseline_llr(data)

    if run_flags["cdf_figure"]:
        if not baseline_cells:
            baseline_cells = run_baseline_llr(data)
        run_cdf_figure(data, baseline_cells)

    if run_flags.get("slice_figure", False):
        if not baseline_cells:
            baseline_cells = run_baseline_llr(data)
        run_slice_figure(data, baseline_cells)

    if run_flags.get("rmse_insample", False):
        # rmse block doesn't need baseline_cells — it loads params from disk
        run_in_sample_rmse(data)

    print("\nAll requested LLR_monthly_jacobi blocks finished.")


if __name__ == "__main__":
    main()
