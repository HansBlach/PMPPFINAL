"""LLR_monthly_twomarkets.py"""
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
import Kalman_filter_TwoMarket as tm
import BIC_monthly_twomarkets   as drv


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

M_FACTORS = 1

RUN = {
    "baseline_llr":  True,
    "slice_figure":  True,
    "rmse_insample": True,
}

# Supported degrees for the LLR baseline sweep.
BASELINE_DEGS = (1, 3, 5)

NESTED_PAIRS = [(1, 3), (1, 5), (3, 5)]

# ---- 1-D logL slice figure config (ported from LLR_monthly_jacobi) ----

SLICE_DEGS      = (1, 3, 5)   # two-market supports odd-only Sun degrees
SLICE_GRID_PTS  = 41          # odd → MLE itself sits on the grid
SLICE_REL_RANGE = 0.25
SLICE_MIN_HALF  = 0.05

# Per-parameter slice-window overrides. Key is the label as rendered by
# `param_labels` (e.g. r"$\beta_{1}$"); value is a dict with optional "lo"
# and/or "hi" keys. Missing keys keep the default grid edge.
SLICE_RANGE_OVERRIDES = {}

SLICE_PTS_OVERRIDES = {
    r"$\beta":  121,
    r"$\gamma": 121,
}

SLICE_LOGL_WINDOW = 500.0

SLICE_XFIT_TO_WINDOW = True


# ---------------------------------------------------------------
# Inherit panel / flag config from BIC_monthly_twomarkets
# ---------------------------------------------------------------

OUT_DIR     = drv.OUT_DIR
PERIOD_TAG  = drv.PERIOD_TAG
FIG_DIR     = os.path.join(OUT_DIR, f"figures_llr_twomarket_{PERIOD_TAG}")
THESIS_DIR  = os.path.join(OUT_DIR, "figures_thesis")
RMSE_NPZ_DIR = os.path.join(OUT_DIR,
                              f"predictions_twomarket_in_sample_llr_{PERIOD_TAG}")
os.makedirs(FIG_DIR,      exist_ok=True)
os.makedirs(THESIS_DIR,   exist_ok=True)
os.makedirs(RMSE_NPZ_DIR, exist_ok=True)

TAU_REF  = drv.TAU_REF

USE_WEEKLY_SAMPLING = drv.USE_WEEKLY_SAMPLING
DT       = drv.DT_EKF

CONTRACT_LABELS = list(drv.SUBSET_LABELS)


def params_filename(m_per_market, N_poly):
    """Match BIC_monthly_twomarkets.run_ekf_grid's saved-filename convention."""
    return f"params_{PERIOD_TAG}_twomarket_m{m_per_market}_N{N_poly}.npy"


# ---------------------------------------------------------------
# Stage A / Stage B data load + seasonality fits
# ---------------------------------------------------------------

def load_panels_and_residuals():
    """Load joint DE + FR Stage A and Stage B panels, fit per-market Stage A"""
    print("Loading two-market Stage A panels (DE + FR) via "
          "BIC_monthly_twomarkets.load_stage_a_data ...")
    (y_a_DE, mat_a_DE, del_a_DE, tra_a_DE), \
    (y_a_FR, mat_a_FR, del_a_FR, tra_a_FR) = drv.load_stage_a_data()
    print(f"  Stage A DE: {y_a_DE.shape[0]} days × {y_a_DE.shape[1]} contracts "
          f"({tuple(drv.STAGE_A_LABELS)})")
    print(f"  Stage A FR: {y_a_FR.shape[0]} days × {y_a_FR.shape[1]} contracts")

    print("\nLoading joint two-market Stage B panel ...")
    y1, y2, mat_1, mat_2, del_1, del_2, trading = drv.load_stage_b_data()
    print(f"  Stage B (joined): {y1.shape[0]} days × {y1.shape[1]} contracts "
          f"per market ({tuple(CONTRACT_LABELS)})")

    if drv.START_DATE is not None:
        y_a_DE, mat_a_DE, del_a_DE, tra_a_DE, idx_a_DE = \
            drv.slice_panel_after_date(drv.START_DATE, y_a_DE, mat_a_DE,
                                         del_a_DE, tra_a_DE)
        y_a_FR, mat_a_FR, del_a_FR, tra_a_FR, idx_a_FR = \
            drv.slice_panel_after_date(drv.START_DATE, y_a_FR, mat_a_FR,
                                         del_a_FR, tra_a_FR)
        y1, y2, mat_1, mat_2, del_1, del_2, trading, idx_b = \
            drv.slice_panel_after_date(drv.START_DATE, y1, y2,
                                         mat_1, mat_2, del_1, del_2, trading)
        print(f"  Restricting to dates >= {drv.START_DATE}: "
              f"Stage A DE -{idx_a_DE}, Stage A FR -{idx_a_FR}, "
              f"Stage B -{idx_b}; new Stage B size = {y1.shape[0]}")

    price_scale_1 = float(y_a_DE.mean())
    price_scale_2 = float(y_a_FR.mean())
    y1_a_norm     = y_a_DE / price_scale_1
    y2_a_norm     = y_a_FR / price_scale_2
    y1_norm       = y1     / price_scale_1
    y2_norm       = y2     / price_scale_2
    print(f"  price_scale DE = {price_scale_1:.4f} EUR/MWh")
    print(f"  price_scale FR = {price_scale_2:.4f} EUR/MWh")

    # Per-market Stage A seasonality grid (same code path as the BIC driver).
    print("\nFitting Stage A seasonality grid (DE) ...")
    best_1 = None
    for ah in drv.ANNUAL_GRID:
        info = tm.seasonality_bic(tra_a_DE[:, 0], mat_a_DE, del_a_DE,
                                   y1_a_norm, ah)
        if best_1 is None or info["BIC"] < best_1["BIC"]:
            best_1 = info
    seas_beta_1 = best_1["beta"]
    print(f"  DE best seasonality: a={int(best_1['annual_h'])} "
          f"BIC={best_1['BIC']:.1f}")

    print("\nFitting Stage A seasonality grid (FR) ...")
    best_2 = None
    for ah in drv.ANNUAL_GRID:
        info = tm.seasonality_bic(tra_a_FR[:, 0], mat_a_FR, del_a_FR,
                                   y2_a_norm, ah)
        if best_2 is None or info["BIC"] < best_2["BIC"]:
            best_2 = info
    seas_beta_2 = best_2["beta"]
    print(f"  FR best seasonality: a={int(best_2['annual_h'])} "
          f"BIC={best_2['BIC']:.1f}")

    # Rebuild design on the Stage B panel for each market.
    n_t, n_c = mat_1.shape
    _, S_1, _ = tm.build_seasonality_matrix(
        trading[:, 0], mat_1, del_1, y1_norm,
        annual_h=int(best_1["annual_h"]),
    )
    _, S_2, _ = tm.build_seasonality_matrix(
        trading[:, 0], mat_2, del_2, y2_norm,
        annual_h=int(best_2["annual_h"]),
    )
    g_bar_1   = (S_1 @ seas_beta_1).reshape(n_t, n_c)
    g_bar_2   = (S_2 @ seas_beta_2).reshape(n_t, n_c)
    y_resid_1 = y1_norm - g_bar_1
    y_resid_2 = y2_norm - g_bar_2
    print(f"\n  Stage B residual DE (pre-thin): mean={y_resid_1.mean():+.5f}  "
          f"std={y_resid_1.std():.5f}")
    print(f"  Stage B residual FR (pre-thin): mean={y_resid_2.mean():+.5f}  "
          f"std={y_resid_2.std():.5f}")

    # Data already at calibration cadence (weekly when USE_WEEKLY_SAMPLING).
    print(f"  Calibration cadence: "
          f"{'weekly ISO-Mon' if drv.USE_WEEKLY_SAMPLING else 'daily'}; "
          f"Stage B size = {y1.shape[0]}, DT = {DT:.6f}")

    return dict(
        y1=y1, y2=y2, mat_1=mat_1, mat_2=mat_2, del_1=del_1, del_2=del_2,
        trading=trading,
        y_resid_1=y_resid_1, y_resid_2=y_resid_2,
        g_bar_1=g_bar_1,   g_bar_2=g_bar_2,
        price_scale_1=price_scale_1, price_scale_2=price_scale_2,
        seas_beta_1=seas_beta_1, seas_beta_2=seas_beta_2,
        annual_h_1=int(best_1["annual_h"]),
        annual_h_2=int(best_2["annual_h"]),
    )



def evaluate_logL(params_vec, data, m_per_market, N_poly):
    """Wrapper around tm.EKF_MLE that returns +logL on the joint residual."""
    nll = tm.EKF_MLE(
        params_vec,
        data["y_resid_1"], data["y_resid_2"],
        data["mat_1"], data["del_1"],
        data["mat_2"], data["del_2"],
        DT, N_poly, m_per_market,
        TAU_REF,
    )
    return -float(nll)


# ---------------------------------------------------------------
# Parameter accounting + LR test
# ---------------------------------------------------------------

def n_active_poly_params(N_poly):
    """Number of *active* polynomial-map parameters at degree N_poly,"""
    if N_poly == 1:
        return 2          # p_delta per market only
    if N_poly == 3:
        return 4          # adds p_beta per market
    if N_poly == 5:
        return 8          # adds p_gamma and p_K per market
    raise ValueError(f"Unsupported N_poly={N_poly}")


def llr_test(logL_restricted, logL_full, df):
    LR = 2.0 * (logL_full - logL_restricted)
    if not np.isfinite(LR):
        return float("nan"), float("nan"), df
    if LR < 0:
        return LR, 1.0, df
    p = float(chi2.sf(LR, df))
    return LR, p, df


# ---------------------------------------------------------------
# Parameter report
# ---------------------------------------------------------------

def params_to_row(params_vec, m_per_market, N_poly):
    """Unpack a two-market `params_vec` and return a flat dict of all parameters."""
    if params_vec is None:
        return {}
    try:
        p = tm.unpack(params_vec, m_per_market=m_per_market, N_poly=N_poly)
    except Exception:
        return {}
    row = {}
    def put(k, v): row[k] = float(v)
    for i, k in enumerate(p.kappa_Z): put(f"kappa_Z_{i}", k)
    for i, k in enumerate(p.sigma_Z): put(f"sigma_Z_{i}", k)
    for i, k in enumerate(p.lam_Z):   put(f"lam_Z_{i}",   k)
    for i, k in enumerate(p.kappa_Y): put(f"kappa_Y_{i}", k)
    for i, k in enumerate(p.sigma_Y): put(f"sigma_Y_{i}", k)
    for i, k in enumerate(p.lam_Y):   put(f"lam_Y_{i}",   k)
    put("kappa_R", p.kappa_R)
    put("theta_R", p.theta_R)
    put("sigma_R", p.sigma_R)
    put("lam_R",   p.lam_R)
    put("p_delta_1", p.p_delta_1); put("p_beta_1", p.p_beta_1)
    put("p_delta_2", p.p_delta_2); put("p_beta_2", p.p_beta_2)
    if N_poly >= 5:
        put("p_gamma_1", p.p_gamma_1); put("p_K_1", p.p_K_1)
        put("p_gamma_2", p.p_gamma_2); put("p_K_2", p.p_K_2)
    put("p_e_1", p.p_e_1)
    put("p_e_2", p.p_e_2)
    return row


# ---------------------------------------------------------------
# Block 1 — baseline LLR
# ---------------------------------------------------------------

def run_baseline_llr(data):
    print("\n=== Block 1: baseline LLR (two-market, Sun polynomial) ===")
    # BIC penalty sample size — matches BIC_monthly_twomarkets.run_ekf_grid
    # (joint count across both markets, with NaN-masking).
    n_obs_bic = int(data["y_resid_1"].size + data["y_resid_2"].size
                    - np.isnan(data["y_resid_1"]).sum()
                    - np.isnan(data["y_resid_2"]).sum())

    rows  = []
    cells = {}
    for N in BASELINE_DEGS:
        pf_path = os.path.join(OUT_DIR, params_filename(M_FACTORS, N))
        if not os.path.exists(pf_path):
            print(f"  [skip] missing {pf_path}  -- run "
                  f"BIC_monthly_twomarkets.py first")
            continue
        v = np.load(pf_path)
        expected_k = tm.num_params(M_FACTORS, N)
        if v.shape[0] != expected_k:
            print(f"  [skip] {os.path.basename(pf_path)}: saved length "
                  f"{v.shape[0]} != num_params(m={M_FACTORS}, N={N}) "
                  f"= {expected_k}. Refit the cell with the current "
                  f"BIC_monthly_twomarkets before re-running the LLR.")
            continue

        logL  = evaluate_logL(v, data, M_FACTORS, N)
        n_eff = n_active_poly_params(N)
        k_full = expected_k
        bic = k_full * np.log(n_obs_bic) - 2.0 * logL

        row = {"m_per_market": M_FACTORS, "N_poly": N,
               "n_active_poly_params": n_eff,
               "k_total":  k_full,
               "n_obs":    n_obs_bic,
               "logL":     logL,
               "BIC":      bic,
               "params_file": os.path.basename(pf_path)}
        row.update(params_to_row(v, M_FACTORS, N))
        rows.append(row)
        cells[N] = dict(params_vec=v, logL=logL, n_eff=n_eff,
                         BIC=bic, k_total=k_full)
        print(f"  m_per_market={M_FACTORS}  N={N}  logL={logL:.4f}  "
              f"BIC={bic:.2f}  "
              f"(n_active_poly_params={n_eff}, k_total={k_full})")

    pair_rows = []
    for (Nr, Nf) in NESTED_PAIRS:
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
                             f"llr_twomarket_baseline_m{M_FACTORS}_{PERIOD_TAG}.csv")
    out = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    with open(base_csv, "w") as f:
        f.write("# Per-cell log-likelihoods (joint DE + FR)\n")
        out.to_csv(f, index=False)
        f.write("\n# Pairwise LLR\n")
        pair_df.to_csv(f, index=False)
    print(f"  saved -> {base_csv}")
    return cells


# ---------------------------------------------------------------
# Block X — 1-D logL slice figure
# ---------------------------------------------------------------

def _lab(sym, fid, idx=None, m=1):
    """LaTeX label in the thesis SDE notation. `sym` is the bare symbol"""
    if idx is not None and m > 1:
        return f"${sym}_{{{fid},{idx}}}$"
    return f"${sym}_{{{fid}}}$"


def param_labels(m_per_market, N_poly):
    """Ordered labels matching tm.pack / tm.make_bounds, rendered in the"""
    m = m_per_market
    L = []
    # X^(1) leader (Z): speed θ_1, mean μ_1, vol σ_1, risk premium λ_1
    for i in range(m): L.append(_lab(r"\theta",  1, i, m))   # kappa_Z (speed)
    for i in range(m): L.append(_lab(r"\mu",     1, i, m))   # theta_Z (Q-mean)
    for i in range(m): L.append(_lab(r"\sigma",  1, i, m))   # sigma_Z
    for i in range(m): L.append(_lab(r"\lambda", 1, i, m))   # lam_Z   (premium)
    # X^(2) follower (Y): speed θ_2 (reverts to X^(1), no constant mean)
    for i in range(m): L.append(_lab(r"\theta",  2, i, m))   # kappa_Y (speed)
    for i in range(m): L.append(_lab(r"\sigma",  2, i, m))   # sigma_Y
    for i in range(m): L.append(_lab(r"\lambda", 2, i, m))   # lam_Y   (premium)
    # X^(3) correlation (R): speed θ_3, mean μ_3, vol σ_3, premium λ_3
    L.append(_lab(r"\theta",  3))                            # kappa_R (speed)
    L.append(_lab(r"\mu",     3))                            # theta_R (Q-mean)
    L.append(_lab(r"\sigma",  3))                            # sigma_R
    L.append(_lab(r"\lambda", 3))                            # lam_R    (premium)
    # Observation map (Sun polynomial) — not part of the state SDEs.
    L.append(r"$\delta_{1}$")
    L.append(r"$\beta_{1}$")
    if N_poly >= 5:
        L.append(r"$\gamma_{1}$")
        L.append(r"$K_{1}$")
    L.append(r"$\delta_{2}$")
    L.append(r"$\beta_{2}$")
    if N_poly >= 5:
        L.append(r"$\gamma_{2}$")
        L.append(r"$K_{2}$")
    L.append(r"$p_{e,1}$")
    L.append(r"$p_{e,2}$")
    return L


def _slice_n_pts(label):
    """Grid-point count for a slice panel, honouring SLICE_PTS_OVERRIDES
    (keyed by a label prefix). Falls back to SLICE_GRID_PTS."""
    for prefix, n in SLICE_PTS_OVERRIDES.items():
        if label.startswith(prefix):
            return int(n)
    return SLICE_GRID_PTS


def make_slice_grid(lo, hi, mle, n_pts=SLICE_GRID_PTS,
                     rel_range=SLICE_REL_RANGE,
                     min_half_frac=SLICE_MIN_HALF):
    if abs(hi - lo) < 1e-12:
        return None                       # pinned parameter (lo == hi)
    half = max(rel_range * abs(mle), min_half_frac * (hi - lo))
    a = max(lo, mle - half)
    b = min(hi, mle + half)
    if b - a < 1e-10:
        return None
    return np.linspace(a, b, n_pts)


def run_slice_figure(data, baseline_cells):
    print("\n=== Block: 1-D logL slice figure (two-market) ===")
    degs = [d for d in SLICE_DEGS if d in baseline_cells]
    if not degs:
        print("  ! No baseline cells available for slice figure. Skipping.")
        return None

    out_paths = []
    for N in degs:
        v_mle  = np.asarray(baseline_cells[N]["params_vec"], dtype=float).copy()
        bounds = tm.make_bounds(M_FACTORS, N)
        labels = param_labels(M_FACTORS, N)
        if not (len(v_mle) == len(bounds) == len(labels)):
            print(f"  [skip N={N}] vector/bound/label length mismatch "
                  f"(v={len(v_mle)}, b={len(bounds)}, l={len(labels)})")
            continue

        logL_mle = evaluate_logL(v_mle, data, M_FACTORS, N)
        print(f"\n  m={M_FACTORS}  N={N}   logL(MLE) = {logL_mle:.4f}   "
              f"({len(v_mle)} params)")

        slices = []
        t0 = time.time()
        for i in range(len(v_mle)):
            lo, hi  = bounds[i]
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
            status = ""
            width  = hi - lo
            if width > 0:
                if mle_val - lo < 0.01 * width:
                    status = "lower"
                elif hi - mle_val < 0.01 * width:
                    status = "upper"
            ll = np.full_like(grid, np.nan, dtype=float)
            for j, val in enumerate(grid):
                x = v_mle.copy()
                x[i] = float(val)
                ll[j] = evaluate_logL(x, data, M_FACTORS, N)
            # Mask only EKF-failure sentinels / log_det blow-ups so they
            # don't drag the per-panel y-axis autoscale.
            ll[ll < -1e6] = np.nan
            slices.append((labels[i], grid, ll, mle_val, status))
        print(f"    ... swept in {time.time() - t0:.1f}s")

        n = len(slices)
        if n == 0:
            print(f"  [skip N={N}] no free parameters to slice.")
            continue
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(3.4 * ncols, 2.5 * nrows),
                                  squeeze=False)
        for idx, (label, grid, ll, mle_val, status) in enumerate(slices):
            ax = axes[idx // ncols][idx % ncols]
            colour = "#D62728" if status in ("lower", "upper") else "#1F77B4"
            ax.plot(grid, ll, "-", color=colour, lw=1.2)
            ax.axvline(mle_val, color="#D62728", ls="--", lw=0.8)
            j_mle = int(np.argmin(np.abs(grid - mle_val)))
            ax.scatter([grid[j_mle]], [ll[j_mle]],
                       color="#D62728", s=18, zorder=5)
            tag = f"  [at {status} bound]" if status in ("lower", "upper") else ""
            # Clamp the y-axis to a fixed window below the panel max so
            # deep tails don't compress the well around the local max.
            if SLICE_LOGL_WINDOW is not None:
                finite = ll[np.isfinite(ll)]
                if finite.size:
                    y_hi = float(np.nanmax(finite))
                    y_lo = max(float(np.nanmin(finite)),
                               y_hi - float(SLICE_LOGL_WINDOW))
                    pad  = 0.05 * max(y_hi - y_lo, 1e-9)
                    ax.set_ylim(y_lo - pad, y_hi + pad)
                    # Trim the x-axis to where the curve is inside the
                    # window, extended one grid step each side so the line
                    # reaches the bottom corners.
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
            f"slice_twomarket_m{M_FACTORS}_N{N}_{PERIOD_TAG}.png",
        )
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    saved -> {out_path}")
        out_paths.append(out_path)

    return out_paths


# ---------------------------------------------------------------
# Block 2 — in-sample RMSE / EKF diagnostics
# ---------------------------------------------------------------

def _ekf_run_with_diagnostics_twomarket(params, x0, P0, data, N_pricing):
    """Mirror `tm.EKF_run_two_market` but capture per-step residuals,"""
    y_obs_1 = data["y_resid_1"]
    y_obs_2 = data["y_resid_2"]
    T1, del1 = data["mat_1"], data["del_1"]
    T2, del2 = data["mat_2"], data["del_2"]
    n_steps, n_c = y_obs_1.shape

    # ---- build the per-step pricing operators (same as EKF_run_two_market) ----
    helper = tm._PredictHelper(params, DT, N_pred=2)
    G_Q    = tm.infinitesimal_generator_two_market(params, N=N_pricing,
                                                    use_P=False)
    p_T_1  = tm.build_poly_market(params, market=1, N=N_pricing)
    p_T_2  = tm.build_poly_market(params, market=2, N=N_pricing)
    Mp_1_all, expm_cache = tm._precompute_Mp(G_Q, p_T_1, T1, del1)
    Mp_2_all, _          = tm._precompute_Mp(G_Q, p_T_2, T2, del2,
                                              expm_cache=expm_cache)
    R_all_1 = tm.precompute_R(T1, params.p_e_1, tau_ref=TAU_REF)
    R_all_2 = tm.precompute_R(T2, params.p_e_2, tau_ref=TAU_REF)

    # ---- output buffers ----
    resid_1   = np.full((n_steps, n_c), np.nan)
    resid_2   = np.full((n_steps, n_c), np.nan)
    y_pred_1  = np.full((n_steps, n_c), np.nan)
    y_pred_2  = np.full((n_steps, n_c), np.nan)
    S_diag_1  = np.full((n_steps, n_c), np.nan)
    S_diag_2  = np.full((n_steps, n_c), np.nan)
    n_state   = params.n_state
    state_filt  = np.full((n_steps, n_state), np.nan)
    state_cov_d = np.full((n_steps, n_state), np.nan)

    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    P = np.atleast_2d(P0)
    log_lik = 0.0

    for t in range(n_steps):
        y_t      = np.concatenate([np.asarray(y_obs_1[t]).flatten(),
                                    np.asarray(y_obs_2[t]).flatten()])
        R_diag_t = np.concatenate([R_all_1[t], R_all_2[t]])
        try:
            x, P, resid, S = tm.EKF_step_two_market(
                params, x, P, helper,
                Mp_1_all[t], Mp_2_all[t],
                y_t, R_diag_t, N_pricing,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(f"     EKF step failed at t={t}: {exc}")
            break
        sign, log_det = np.linalg.slogdet(S)
        if sign <= 0 or not np.isfinite(log_det):
            print(f"     non-PSD S at t={t}; aborting filter")
            break

        # Split the stacked (n_c1 + n_c2) innovation back into two markets.
        n_c1     = n_c
        resid_1[t]   = resid[:n_c1]
        resid_2[t]   = resid[n_c1:]
        # y_pred is the *predicted observation*; recover it from resid =
        # y - y_pred → y_pred = y - resid (per-market subset).
        y_pred_1[t]  = y_obs_1[t] - resid_1[t]
        y_pred_2[t]  = y_obs_2[t] - resid_2[t]
        S_diag       = np.diag(S)
        S_diag_1[t]  = S_diag[:n_c1]
        S_diag_2[t]  = S_diag[n_c1:]
        state_filt[t]  = np.asarray(x).flatten()
        state_cov_d[t] = np.diag(P)

        n_obs_t  = len(resid)
        log_lik += -0.5 * (n_obs_t * np.log(2 * np.pi) + log_det
                           + resid @ np.linalg.solve(S, resid))

    return dict(
        resid_1=resid_1, resid_2=resid_2,
        y_pred_1=y_pred_1, y_pred_2=y_pred_2,
        S_diag_1=S_diag_1, S_diag_2=S_diag_2,
        state_filt=state_filt, state_cov_diag=state_cov_d,
        log_lik=log_lik,
    )


def _safe_rmse(arr):
    sq = np.asarray(arr, dtype=float) ** 2
    if np.all(np.isnan(sq)):
        return float("nan")
    return float(np.sqrt(np.nanmean(sq)))


def _coverage_rate(resid, S_diag):
    sd = np.sqrt(S_diag)
    inside = np.abs(resid) < 2.0 * sd
    valid  = ~np.isnan(resid) & ~np.isnan(S_diag)
    if valid.sum() == 0:
        return float("nan")
    return float(inside[valid].mean())


def run_in_sample_rmse(data):
    """Per-cell EKF filter through the joint Stage B residual; saves"""
    print("\n=== Block 2: in-sample RMSE / EKF diagnostics (two-market) ===")

    rmse_rows  = []
    bias_rows  = []
    cov_rows   = []

    n_c = data["mat_1"].shape[1]

    for N in BASELINE_DEGS:
        pf_path = os.path.join(OUT_DIR, params_filename(M_FACTORS, N))
        if not os.path.exists(pf_path):
            print(f"  [m={M_FACTORS} N={N}] missing {pf_path}, skipping")
            continue
        v = np.load(pf_path)
        try:
            params = tm.unpack(v, m_per_market=M_FACTORS, N_poly=N)
        except Exception as exc:
            print(f"  [m={M_FACTORS} N={N}] unpack failed ({exc}), skipping")
            continue

        # Use the model's own initial-state helper if available, fall back
        # to a zero-state warm start otherwise.
        try:
            x0, P0 = tm._initial_state(params)
        except AttributeError:
            x0 = np.zeros((params.n_state, 1))
            P0 = np.eye(params.n_state) * 0.1

        t0 = time.time()
        print(f"  [m={M_FACTORS} N={N}]  filtering ...", flush=True)
        try:
            out = _ekf_run_with_diagnostics_twomarket(
                params, x0, P0, data, N_pricing=N,
            )
        except Exception as exc:
            print(f"     diagnostics pass not supported by current "
                  f"Kalman_filter_TwoMarket build ({exc}); skipping this cell")
            continue
        elapsed = time.time() - t0

        # ---- RMSE per market and per contract ----
        rmse_norm_1 = _safe_rmse(out["resid_1"])
        rmse_norm_2 = _safe_rmse(out["resid_2"])
        rmse_eur_1  = rmse_norm_1 * data["price_scale_1"]
        rmse_eur_2  = rmse_norm_2 * data["price_scale_2"]
        per_c_eur_1 = np.array([_safe_rmse(out["resid_1"][:, c])
                                 for c in range(n_c)]) * data["price_scale_1"]
        per_c_eur_2 = np.array([_safe_rmse(out["resid_2"][:, c])
                                 for c in range(n_c)]) * data["price_scale_2"]
        print(f"     RMSE (DE, EUR/MWh) = {rmse_eur_1:.3f}  "
              f"per-contract = {['%.3f' % v for v in per_c_eur_1]}")
        print(f"     RMSE (FR, EUR/MWh) = {rmse_eur_2:.3f}  "
              f"per-contract = {['%.3f' % v for v in per_c_eur_2]}  "
              f"({elapsed:.1f}s)")

        row = {"m_per_market": M_FACTORS, "N_poly": N, "split": "in_sample",
               "logL": out["log_lik"],
               "rmse_norm_DE": rmse_norm_1, "rmse_eur_DE": rmse_eur_1,
               "rmse_norm_FR": rmse_norm_2, "rmse_eur_FR": rmse_eur_2}
        for c in range(n_c):
            row[f"rmse_{CONTRACT_LABELS[c]}_DE_eur"] = per_c_eur_1[c]
            row[f"rmse_{CONTRACT_LABELS[c]}_FR_eur"] = per_c_eur_2[c]
        rmse_rows.append(row)

        # ---- Bias per contract per market ----
        bias_norm_1 = np.nanmean(out["resid_1"], axis=0)
        bias_norm_2 = np.nanmean(out["resid_2"], axis=0)
        for c in range(n_c):
            bias_rows.append({"m_per_market": M_FACTORS, "N_poly": N,
                              "market": "DE",
                              "contract": CONTRACT_LABELS[c],
                              "bias_norm": float(bias_norm_1[c]),
                              "bias_eur":  float(bias_norm_1[c]
                                                 * data["price_scale_1"])})
            bias_rows.append({"m_per_market": M_FACTORS, "N_poly": N,
                              "market": "FR",
                              "contract": CONTRACT_LABELS[c],
                              "bias_norm": float(bias_norm_2[c]),
                              "bias_eur":  float(bias_norm_2[c]
                                                 * data["price_scale_2"])})

        # ---- Coverage at 2-sigma per market ----
        cov_rows.append({"m_per_market": M_FACTORS, "N_poly": N,
                          "market": "DE",
                          "coverage_2sigma":
                              _coverage_rate(out["resid_1"], out["S_diag_1"])})
        cov_rows.append({"m_per_market": M_FACTORS, "N_poly": N,
                          "market": "FR",
                          "coverage_2sigma":
                              _coverage_rate(out["resid_2"], out["S_diag_2"])})

        # ---- Save per-cell predictions ----
        npz_path = os.path.join(
            RMSE_NPZ_DIR,
            f"predictions_m{M_FACTORS}_N{N}_in_sample.npz")
        np.savez(
            npz_path,
            y_pred_eur_DE = data["price_scale_1"]
                             * (data["g_bar_1"] + out["y_pred_1"]),
            y_pred_eur_FR = data["price_scale_2"]
                             * (data["g_bar_2"] + out["y_pred_2"]),
            resid_DE     = out["resid_1"],
            resid_FR     = out["resid_2"],
            resid_eur_DE = out["resid_1"] * data["price_scale_1"],
            resid_eur_FR = out["resid_2"] * data["price_scale_2"],
            state_filt   = out["state_filt"],
            state_cov_diag = out["state_cov_diag"],
            S_diag_DE    = out["S_diag_1"],
            S_diag_FR    = out["S_diag_2"],
            trading      = data["trading"],
            log_lik      = out["log_lik"],
            price_scale_DE = data["price_scale_1"],
            price_scale_FR = data["price_scale_2"],
        )

    summary_csv  = os.path.join(OUT_DIR,
        f"rmse_insample_twomarket_m{M_FACTORS}_{PERIOD_TAG}.csv")
    bias_csv     = os.path.join(OUT_DIR,
        f"rmse_insample_twomarket_bias_m{M_FACTORS}.csv")
    coverage_csv = os.path.join(OUT_DIR,
        f"rmse_insample_twomarket_coverage_m{M_FACTORS}.csv")
    pd.DataFrame(rmse_rows).to_csv(summary_csv,  index=False)
    pd.DataFrame(bias_rows).to_csv(bias_csv,     index=False)
    pd.DataFrame(cov_rows).to_csv(coverage_csv,  index=False)
    print(f"\n  Saved per-cell predictions to {RMSE_NPZ_DIR}")
    print(f"  Saved {summary_csv}")
    print(f"  Saved {bias_csv}")
    print(f"  Saved {coverage_csv}")


# ---------------------------------------------------------------
# Main + CLI
# ---------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="LLR / RMSE framework for the two-market Kalman model.")
    p.add_argument("--baseline", action="store_true",
                    help="Run the baseline LLR block.")
    p.add_argument("--slice",    action="store_true",
                    help="Run the 1-D logL slice figure.")
    p.add_argument("--rmse",     action="store_true",
                    help="Run the in-sample RMSE / EKF diagnostics block.")
    p.add_argument("--all",      action="store_true",
                    help="Run every block.")
    p.add_argument("--m",        type=int, choices=(1, 2), default=None,
                    help="Override M_FACTORS (m_per_market) for this run.")
    return p.parse_args()


def _resolve_run_flags(args):
    if args.all:
        return {"baseline_llr": True, "slice_figure": True,
                "rmse_insample": True}
    any_block = (args.baseline or args.slice or args.rmse)
    if any_block:
        return {"baseline_llr":  bool(args.baseline),
                "slice_figure":  bool(args.slice),
                "rmse_insample": bool(args.rmse)}
    return dict(RUN)


def main():
    args = _parse_args()
    global M_FACTORS
    if args.m is not None:
        M_FACTORS = int(args.m)
    run_flags = _resolve_run_flags(args)

    print(f"=== LLR_monthly_twomarkets :  M_FACTORS (m_per_market) "
          f"= {M_FACTORS} ===")
    print(f"OUT_DIR = {OUT_DIR}")
    print(f"FIG_DIR = {FIG_DIR}")
    print(f"RUN     = {run_flags}")
    data = load_panels_and_residuals()

    baseline_cells = {}
    if run_flags["baseline_llr"]:
        baseline_cells = run_baseline_llr(data)

    if run_flags.get("slice_figure", False):
        if not baseline_cells:
            baseline_cells = run_baseline_llr(data)
        run_slice_figure(data, baseline_cells)

    if run_flags["rmse_insample"]:
        run_in_sample_rmse(data)

    print("\nAll requested LLR_monthly_twomarkets blocks finished.")


if __name__ == "__main__":
    main()
