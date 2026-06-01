"""LLR_monthly_OU.py"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
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

import GetData as gd                              # noqa: F401  (kept for parity)
import Kalman_filter_LD as ld
import BIC_monthly_OU   as drv

# Re-use simulation primitives — these have no weekly/monthly assumptions
# baked in; they're pure functions of (params, state, maturity, ...). They
# live in simulate_paths_monthly so the legacy simulate_paths.py can be
# deleted without breaking this file.
from simulate_paths_monthly import (
    filter_to_end,
    simulate_state_paths,
    compute_observations,
)


# ---------------------------------------------------------------
# Configuration  (all four blocks are gated by the RUN dict below)
# ---------------------------------------------------------------

# Number of latent factors. Must match the saved params filename.
M_FACTORS = 1

# Switch which blocks run on this invocation. Each block is independent
# except that the CDF block reuses the baseline-loaded params.
RUN = {
    "baseline_llr":     True,
    "deg3_beta_sweep":  True,
    "deg5_joint_sweep": True,
    "cdf_figure":       True,
    "slice_figure":     True,    
    "rmse_insample":    True,    # In-sample EKF residual RMSE / diagnostics
}

# ---- Slice figure ----
# For each parameter at its fitted MLE, evaluate the EKF logL on a grid
# around the MLE (other params held fixed) and plot logL(param). Confirms
# each slot is at a local optimum (parabolic) — and makes it visually
# obvious which slot, if any, is monotonic into a bound (e.g., p_gamma).
SLICE_DEGS        = (3, 5)
SLICE_GRID_PTS    = 21        # odd → MLE itself sits on the grid
SLICE_REL_RANGE   = 0.5       # +/- 50% around the MLE value
SLICE_MIN_HALF    = 0.05      # minimum half-width as fraction of bound width
# For PINNED slots (lo == hi) we still sweep this absolute half-width
# around the pinned value, so the figure can confirm the pin sits at a
# local optimum (or on a flat region — both are useful).
SLICE_PINNED_HALF = 0.5

# Upper-bound ladder for p_beta (lower bound stays at 0).
BETA_UPPERS  = (2.0, 1.5, 1.0, 0.5, 0.3)
# Upper-bound ladder for p_gamma (lower bound stays at the existing 0.01
# floor used in make_bounds_shared so the EKF can build phi'(x) > 0).
GAMMA_UPPERS = (2.0, 1.5, 1.0, 0.5, 0.3)

# Refit controls. Smaller DE budgets than the BIC driver because we run
# many cells; tune up for production runs if needed.
DE_MAXITER_DEG3 = 60
DE_MAXITER_DEG5 = 40
DE_POPSIZE      = 10
LBFGS_MAXITER   = 500

# CDF figure
N_PATHS_CDF   = 500
CDF_DEGS      = (1, 3, 5)        # baseline degrees to plot
CDF_OBS_NOISE = True             # add per-bucket measurement noise to sim prices

# X-axis cap so a few extreme simulated prices don't smear the CDF.
# The KS distance is ALWAYS computed on the full pooled sample (no clipping),
# so the test result is unaffected by what we display.
#   "simulated_pct" — clip to a percentile range of the SIMULATED prices
#                     (default: show out to the 98th percentile of sim prices)
#   "observed_pct"  — clip to a percentile range of the OBSERVED prices only
#   "pooled_pct"    — clip to a percentile range of (observed + simulated)
#   "none"          — show everything (matplotlib autoscale)
CDF_XLIM_MODE   = "simulated_pct"
CDF_XLIM_PCT    = (1.0, 98.0)    # percentile lo/hi when mode != "none"
CDF_XLIM_BUFFER = 0.05           # 5% padding on each side of the percentile range

SEED = 42


# ---------------------------------------------------------------
# Re-use BIC_monthly_OU's panel / flag config so this file can never
# disagree with whatever you ran the calibration on.
# ---------------------------------------------------------------

OUT_DIR    = drv.OUT_DIR
PERIOD_TAG = drv.PERIOD_TAG
FIG_DIR    = os.path.join(OUT_DIR, f"figures_llr_ou_{PERIOD_TAG}")
THESIS_DIR = os.path.join(OUT_DIR, f"figures_thesis_{PERIOD_TAG}")
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(THESIS_DIR, exist_ok=True)

INDEPENDENT_POLY = drv.INDEPENDENT_POLY
FIT_LAM          = drv.FIT_LAM
FIT_D            = drv.FIT_D
TAU_REF          = drv.TAU_REF
# Match calibration cadence — drv.DT_EKF is 7/365 in weekly mode, 1/252 daily.
USE_WEEKLY_SAMPLING = drv.USE_WEEKLY_SAMPLING
DT               = drv.DT_EKF

CONTRACT_LABELS  = list(drv.SUBSET_LABELS)

INDEP_TAG = "_indep" if INDEPENDENT_POLY else ""
LAM_TAG   = "_lam"   if FIT_LAM         else ""
FLAG_TAG  = INDEP_TAG + LAM_TAG


def params_filename(m, N_poly):
    """Match BIC_monthly_OU.run_ekf_grid's saved-filename convention."""
    return f"params_{PERIOD_TAG}_ou_m{m}_N{N_poly}{FLAG_TAG}.npy"


# ---------------------------------------------------------------
# Stage A / Stage B data load — mirrors simulate_paths_monthly.main()
# Computes the residual y_resid that EKF_MLE expects.
# ---------------------------------------------------------------

def load_panels_and_residual():
    """Load Stage A + Stage B panels, fit Stage A seasonality, return the
    Stage-B residual + everything the simulators / EKF need."""
    print("Loading Stage A + Stage B panels via BIC_monthly_OU loaders ...")
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
              f"Stage A {idx_a} rows dropped, "
              f"Stage B {idx_b} rows dropped, "
              f"new Stage B size = {y_matrix.shape[0]}")

    # Data is already at the calibration cadence (weekly when
    # drv.USE_WEEKLY_SAMPLING, daily otherwise) — see drv._load_panel.
    print(f"  Calibration cadence: "
          f"{'weekly ISO-Mon' if drv.USE_WEEKLY_SAMPLING else 'daily'}; "
          f"Stage B size = {y_matrix.shape[0]}, DT = {DT:.6f}")

    # Shared price scale (Stage A mean) -- consistent with BIC + simulate.
    price_scale  = float(y_stagea.mean())
    y_stagea_n   = y_stagea / price_scale
    y_norm       = y_matrix / price_scale
    print(f"  price_scale = {price_scale:.4f} EUR/MWh")

    # Stage A seasonality grid -> best annual_h and beta.
    print("Fitting Stage A seasonality grid ...")
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

    # Stage B residual (what EKF_MLE consumes).
    n_t, n_c = maturity.shape
    _, S_hist, _ = ld.build_seasonality_matrix(
        trading[:, 0], maturity, delivery, y_norm,
        annual_h=annual_h,
    )
    g_bar   = (S_hist @ seas_beta).reshape(n_t, n_c)
    y_resid = y_norm - g_bar
    print(f"  Stage B residual: mean={y_resid.mean():+.5f}  std={y_resid.std():.5f}")

    return dict(
        y_matrix=y_matrix, maturity=maturity, delivery=delivery, trading=trading,
        y_norm=y_norm, y_resid=y_resid, g_bar=g_bar,
        price_scale=price_scale,
        seas_beta=seas_beta, annual_h=annual_h,
    )


# ---------------------------------------------------------------
# Bounds helpers — wraps make_bounds_shared / make_bounds_independent
# but allows overriding p_beta and p_gamma upper bounds.
# ---------------------------------------------------------------

def make_bounds_capped(m, N_poly, p_beta_ub=None, p_gamma_ub=None):
    """Build the parameter bounds vector for shared OR independent polynomial"""
    head, tail = ld._make_bounds_dynamics_block(m, FIT_D)

    if N_poly == 1:
        # Pin the polynomial slots to 0 — see Kalman_filter_LD docstring.
        if INDEPENDENT_POLY:
            poly = [(0.0, 0.0)] + [(0.0, 0.0)] * m
        else:
            poly = [(0.0, 0.0), (0.0, 0.0)]
    elif INDEPENDENT_POLY:
        # Independent mode (per-factor polynomials).
        beta_lb,  beta_ub_default  = 0.001, 0.4
        gamma_lb, gamma_ub_default = 0.001, 0.4
        b_ub = beta_ub_default  if p_beta_ub  is None else float(p_beta_ub)
        g_ub = gamma_ub_default if p_gamma_ub is None else float(p_gamma_ub)
        poly  = [(-2.0, 2.0)]                   # p_delta
        poly += [(beta_lb, b_ub)] * m           # p_beta_arr
        if N_poly >= 5:
            poly += [(gamma_lb, g_ub)] * m      # p_gamma_arr
            poly += [(-1.0, 1.0)] * m           # p_K_arr
    else:
        # Shared mode.
        beta_ub_default  = 1.0
        gamma_lb, gamma_ub_default = 0.01, 1.0
        b_ub = beta_ub_default  if p_beta_ub  is None else float(p_beta_ub)
        g_ub = gamma_ub_default if p_gamma_ub is None else float(p_gamma_ub)
        poly  = [(-2.0, 2.0)]                   # p_delta
        poly += [(0.0, b_ub)]                   # p_beta
        if N_poly >= 5:
            poly += [(gamma_lb, g_ub)]          # p_gamma
            poly += [(-4.0, 4.0)]               # p_K

    bounds   = head + poly + tail
    expected = ld.num_params_ld(m, N_poly,
                                 fit_d=FIT_D,
                                 independent_poly=INDEPENDENT_POLY)
    assert len(bounds) == expected, (
        f"make_bounds_capped produced {len(bounds)} bounds but num_params_ld "
        f"expects {expected} for m={m}, N_poly={N_poly}.")
    return bounds


# ---------------------------------------------------------------
# Refit one cell with custom bounds. Mirror BIC_monthly_OU.fit_ekf_model
# closely so the optimisation pipeline is identical.
# ---------------------------------------------------------------

def fit_with_bounds(y_resid, maturity, delivery, dt, m, N_poly, bounds,
                     de_maxiter=60, seed=SEED):
    # Data is already at the calibration cadence; `dt` matches it.
    extra = (TAU_REF, FIT_D, INDEPENDENT_POLY)
    args  = (y_resid, maturity, delivery, dt, N_poly, m) + extra

    de = differential_evolution(
        ld.EKF_MLE, bounds=bounds, args=args,
        seed=seed, maxiter=de_maxiter, tol=1e-3,
        popsize=DE_POPSIZE, mutation=(0.5, 1), recombination=0.7,
        workers=1, polish=False,
    )
    lb = minimize(
        fun=ld.EKF_MLE, x0=de.x, args=args,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": LBFGS_MAXITER, "ftol": 1e-12, "gtol": 1e-8},
    )
    return lb


def evaluate_logL(params_vec, y_resid, maturity, delivery, dt, m, N_poly):
    """Re-evaluate the EKF log-likelihood for a previously-saved param"""
    nll = ld.EKF_MLE(
        params_vec, y_resid, maturity, delivery, dt, N_poly, m,
        tau_ref=TAU_REF, fit_d=FIT_D,
        independent_poly=INDEPENDENT_POLY,
    )
    return -float(nll)


# ---------------------------------------------------------------
# Parameter report — flatten an (m, N_poly) packed vector into a dict
# of stable keys so the LLR CSVs can carry the full parameter set per
# row. Useful when reading the CSV in Excel / pandas after the fact.
# ---------------------------------------------------------------

def params_to_row(params_vec, m, N_poly, prefix=""):
    """Unpack `params_vec` and return a flat dict of all model parameters."""
    if params_vec is None:
        return {}
    try:
        p = ld.unpack_ld(
            params_vec, m, N_poly=N_poly, fit_d=FIT_D,
            independent_poly=INDEPENDENT_POLY,
        )
    except Exception:
        return {}
    row = {}
    def put(k, v): row[f"{prefix}{k}"] = float(v)

    for i, t in enumerate(p.theta):
        put(f"theta_{i}", t)
    put("mu_0", p.mu[0])
    if FIT_LAM:
        for i, l in enumerate(p.lam):
            put(f"lam_{i}", l)
    for i, c in enumerate(p.c):
        put(f"c_{i}", c)
    if FIT_D:
        for i, dd in enumerate(p.d):
            put(f"d_{i}", dd)
    put("p_delta", p.p_delta)
    if INDEPENDENT_POLY:
        if p.p_beta_arr is not None:
            for i, b in enumerate(p.p_beta_arr):
                put(f"p_beta_{i}", b)
        if N_poly >= 5:
            if p.p_gamma_arr is not None:
                for i, g in enumerate(p.p_gamma_arr):
                    put(f"p_gamma_{i}", g)
            if p.p_K_arr is not None:
                for i, K in enumerate(p.p_K_arr):
                    put(f"p_K_{i}", K)
    else:
        put("p_beta", p.p_beta)
        if N_poly >= 5:
            put("p_gamma", p.p_gamma)
            put("p_K",     p.p_K)
    put("p_e", p.p_e)
    return row


# ---------------------------------------------------------------
# Effective free polynomial-map parameters per degree.
# In shared mode at deg 1, p_beta is in the bounds list but does NOT
# enter phi(x) = p_delta + x — `num_params_ld` overcounts it. The LLR
# df between models is computed from this effective count, NOT k.
# ---------------------------------------------------------------

def n_active_poly_params(N_poly):
    """How many polynomial-map coefficients actually enter phi(x) AND are"""
    if INDEPENDENT_POLY:
        if N_poly == 1: return 0                       # p_delta + p_beta_arr pinned
        if N_poly == 3: return 1 + M_FACTORS           # p_delta + p_beta_arr
        if N_poly == 5: return 1 + 3 * M_FACTORS       # + p_gamma_arr + p_K_arr
    else:
        if N_poly == 1: return 0                       # p_delta + p_beta pinned
        if N_poly == 3: return 2                       # p_delta + p_beta
        if N_poly == 5: return 4                       # + p_gamma + p_K
    raise ValueError(f"Unsupported N_poly={N_poly}")


def llr_test(logL_restricted, logL_full, df):
    """Returns (LR_stat, p_value, df). LR_stat is 2*(L_full - L_restricted)."""
    LR = 2.0 * (logL_full - logL_restricted)
    if not np.isfinite(LR):
        return float("nan"), float("nan"), df
    if LR < 0:
        # Numerical: with the same data and a strictly nested model, the
        # full likelihood should dominate. Negative usually means the
        # bigger-model optimisation didn't reach the same optimum.
        return LR, 1.0, df
    p = float(chi2.sf(LR, df))
    return LR, p, df


# ---------------------------------------------------------------
# Block 1 — baseline LLR (deg 1 / 3 / 5 unconstrained from saved params)
# ---------------------------------------------------------------

def run_baseline_llr(data):
    print("\n=== Block 1: baseline LLR (deg 1 / 3 / 5 unconstrained) ===")
    y_resid     = data["y_resid"]
    maturity    = data["maturity"]
    delivery    = data["delivery"]
    price_scale = data["price_scale"]

    # BIC penalty sample size. Mirrors BIC_monthly_OU.run_ekf_grid: count
    # the number of valid observation cells in y_resid (already thinned by
    # load_panels_and_residual when USE_THIN_FIT=True).
    n_obs_bic = int(y_resid.shape[0] * y_resid.shape[1]
                    - np.isnan(y_resid).sum())

    rows  = []
    cells = {}
    for N in (1, 3, 5):
        pf_path = os.path.join(OUT_DIR, params_filename(M_FACTORS, N))
        if not os.path.exists(pf_path):
            print(f"  [skip] missing {pf_path}  -- run BIC_monthly_OU first")
            continue
        v = np.load(pf_path)

        expected_k = ld.num_params_ld(
            M_FACTORS, N,
            fit_d=FIT_D,
            independent_poly=INDEPENDENT_POLY,
        )
        if v.shape[0] != expected_k:
            print(f"  [skip] {os.path.basename(pf_path)}: saved length "
                  f"{v.shape[0]} != num_params_ld(m={M_FACTORS}, N={N}, "
                  f"fit_d={FIT_D}, independent_poly={INDEPENDENT_POLY}) "
                  f"= {expected_k}. The file is from an "
                  f"older parameter-layout; refit this cell with the current "
                  f"BIC_monthly_OU before re-running the LLR.")
            continue

        logL = evaluate_logL(v, y_resid, maturity, delivery,
                              DT, M_FACTORS, N)
        n_eff = n_active_poly_params(N)

        k_full = ld.num_params_ld(M_FACTORS, N,
                                    fit_d=FIT_D,
                                    independent_poly=INDEPENDENT_POLY)
        if N == 1:
            k_full -= 1                                # p_delta
            k_full -= M_FACTORS if INDEPENDENT_POLY else 1  # p_beta(_arr)
        bic = k_full * np.log(n_obs_bic) - 2.0 * logL

        # Summed (pooled across maturities) in-sample RMSE in EUR/MWh.
        # Run the same EKF diagnostics pass used by run_in_sample_rmse so
        # the residuals are identical.
        rmse_eur = float("nan")
        try:
            params = ld.unpack_ld(
                v, M_FACTORS, N_poly=N,
                fit_d=FIT_D,
                independent_poly=INDEPENDENT_POLY,
            )
            x0 = ld._mu_P(params).reshape(-1, 1)
            # Full stationary covariance with rho-driven cross terms (was diagonal).
            P0 = ld.stationary_cov(params)
            diag = _ekf_run_with_diagnostics_ou(
                params, x0, P0,
                y_resid, maturity, delivery, DT, N_pricing=N,
            )
            rmse_eur = float(_safe_rmse(diag["resid"]) * price_scale)
        except Exception as exc:
            print(f"  [m={M_FACTORS} N={N}] RMSE pass failed ({exc})")

        row = {"m": M_FACTORS, "N_poly": N,
               "n_active_poly_params": n_eff,
               "k_total":  k_full,
               "n_obs":    n_obs_bic,
               "logL":     logL,
               "BIC":      bic,
               "rmse_eur": rmse_eur,
               "params_file": os.path.basename(pf_path)}
        row.update(params_to_row(v, M_FACTORS, N))
        rows.append(row)
        cells[N] = dict(params_vec=v, logL=logL, n_eff=n_eff,
                         BIC=bic, rmse_eur=rmse_eur, k_total=k_full)
        print(f"  m={M_FACTORS}  N={N}  logL={logL:.4f}  "
              f"BIC={bic:.2f}  RMSE={rmse_eur:.3f} EUR/MWh  "
              f"(n_active_poly_params={n_eff}, k_total={k_full})")

    # Pairwise LLR for the three nested baselines.
    pair_rows = []
    for (Nr, Nf) in [(1, 3), (1, 5), (3, 5)]:
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
                             f"llr_baseline_m{M_FACTORS}{FLAG_TAG}_{PERIOD_TAG}.csv")
    out = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    # Write the per-cell rows + an empty row + the pairwise rows in one csv.
    with open(base_csv, "w") as f:
        f.write("# Per-cell log-likelihoods\n")
        out.to_csv(f, index=False)
        f.write("\n# Pairwise LLR\n")
        pair_df.to_csv(f, index=False)
    print(f"  saved -> {base_csv}")
    return cells


# ---------------------------------------------------------------
# Block 2 — deg-3 sweep over p_beta upper bound
# ---------------------------------------------------------------

def run_deg3_beta_sweep(data, baseline_cells):
    print("\n=== Block 2: deg-3 p_beta upper-bound sweep ===")
    if 1 not in baseline_cells:
        print("  ! deg-1 baseline missing -- cannot run LLR vs deg 1. Skipping.")
        return None
    logL_deg1 = baseline_cells[1]["logL"]
    df_vs1    = n_active_poly_params(3) - n_active_poly_params(1)

    y_resid  = data["y_resid"]
    maturity = data["maturity"]
    delivery = data["delivery"]

    rows = []
    for ub in BETA_UPPERS:
        print(f"\n  -> fitting m={M_FACTORS} N=3 with p_beta in (0, {ub}) ...")
        t0 = time.time()
        bounds = make_bounds_capped(M_FACTORS, 3, p_beta_ub=ub)
        try:
            res    = fit_with_bounds(y_resid, maturity, delivery,
                                      DT, M_FACTORS, 3, bounds,
                                      de_maxiter=DE_MAXITER_DEG3)
            logL_F = -float(res.fun) if res.fun < 1e9 else float("nan")
            success = bool(res.success)
            x_F     = res.x
        except Exception as exc:
            print(f"     FAILED: {exc}")
            logL_F, success, x_F = float("nan"), False, None

        LR, p, _ = llr_test(logL_deg1, logL_F, df_vs1)
        elapsed = time.time() - t0
        row = {"m": M_FACTORS, "N_poly": 3,
               "p_beta_lb": 0.0, "p_beta_ub": ub,
               "logL_full": logL_F,
               "logL_deg1": logL_deg1,
               "df": df_vs1, "LR": LR, "p_value_vs_deg1": p,
               "success": success, "elapsed_s": elapsed}
        row.update(params_to_row(x_F, M_FACTORS, 3))
        rows.append(row)
        print(f"     logL={logL_F:.4f}  LR_vs_deg1={LR:.4f}  "
              f"p={p:.3e}  ({elapsed:.1f}s)")

        # Save the fitted params so the user can later inspect / re-use.
        if x_F is not None and success:
            fname = (f"params_llr_m{M_FACTORS}_N3"
                     f"_betaUB{ub:g}{FLAG_TAG}.npy")
            np.save(os.path.join(OUT_DIR, fname), x_F)

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR,
                            f"llr_deg3_beta_sweep_m{M_FACTORS}{FLAG_TAG}_{PERIOD_TAG}.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n  saved -> {out_csv}")
    return df_out


# ---------------------------------------------------------------
# Block 3 — deg-5 joint 5x5 sweep over (p_beta, p_gamma) upper bounds
# ---------------------------------------------------------------

def run_deg5_joint_sweep(data, baseline_cells):
    print("\n=== Block 3: deg-5 joint (p_beta, p_gamma) upper-bound sweep ===")
    if 1 not in baseline_cells:
        print("  ! deg-1 baseline missing -- cannot run LLR vs deg 1. Skipping.")
        return None
    logL_deg1 = baseline_cells[1]["logL"]
    df_vs1    = n_active_poly_params(5) - n_active_poly_params(1)

    y_resid  = data["y_resid"]
    maturity = data["maturity"]
    delivery = data["delivery"]

    rows  = []
    total = len(BETA_UPPERS) * len(GAMMA_UPPERS)
    k     = 0
    for ub_b in BETA_UPPERS:
        for ub_g in GAMMA_UPPERS:
            k += 1
            print(f"\n  [{k:2d}/{total}]  fitting m={M_FACTORS} N=5  "
                  f"p_beta in (0, {ub_b})   p_gamma in (0.01, {ub_g}) ...")
            t0 = time.time()
            bounds = make_bounds_capped(M_FACTORS, 5,
                                         p_beta_ub=ub_b, p_gamma_ub=ub_g)
            try:
                res    = fit_with_bounds(y_resid, maturity, delivery,
                                          DT, M_FACTORS, 5, bounds,
                                          de_maxiter=DE_MAXITER_DEG5)
                logL_F = -float(res.fun) if res.fun < 1e9 else float("nan")
                success = bool(res.success)
                x_F     = res.x
            except Exception as exc:
                print(f"     FAILED: {exc}")
                logL_F, success, x_F = float("nan"), False, None

            LR, p, _ = llr_test(logL_deg1, logL_F, df_vs1)
            elapsed = time.time() - t0
            row = {"m": M_FACTORS, "N_poly": 5,
                   "p_beta_ub": ub_b, "p_gamma_ub": ub_g,
                   "logL_full": logL_F,
                   "logL_deg1": logL_deg1,
                   "df": df_vs1, "LR": LR, "p_value_vs_deg1": p,
                   "success": success, "elapsed_s": elapsed}
            row.update(params_to_row(x_F, M_FACTORS, 5))
            rows.append(row)
            print(f"     logL={logL_F:.4f}  LR_vs_deg1={LR:.4f}  "
                  f"p={p:.3e}  ({elapsed:.1f}s)")

            if x_F is not None and success:
                fname = (f"params_llr_m{M_FACTORS}_N5"
                         f"_bUB{ub_b:g}_gUB{ub_g:g}{FLAG_TAG}.npy")
                np.save(os.path.join(OUT_DIR, fname), x_F)

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR,
                            f"llr_deg5_joint_sweep_m{M_FACTORS}{FLAG_TAG}_{PERIOD_TAG}.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n  saved -> {out_csv}")
    return df_out


# ---------------------------------------------------------------
# Block X — 1-D logL slice figure
# ---------------------------------------------------------------

def _greek(name, i=None, j=None):
    """LaTeX-rendered parameter label for matplotlib (e.g. r'\theta' → θ)."""
    if i is None and j is None:
        return f"${name}$"
    if j is None:
        return f"${name}_{{{i}}}$"
    return f"${name}_{{{i},{j}}}$"


# Labels (or label prefixes) the slice figure should NOT render. Slots with
# these names are still present in the param vector — they're just hidden
# from the figure (and skipped during sweeping for speed).
SLICE_SKIP_PATTERNS = ()


def param_labels(m, N_poly):
    """Ordered Greek-letter labels matching the OU pack/unpack layout for"""
    labels = []
    for i in range(m):
        labels.append(_greek(r"\theta",  i if m > 1 else None))
    labels.append(_greek(r"\mu"))
    if FIT_LAM:
        for i in range(m):
            labels.append(_greek(r"\lambda", i if m > 1 else None))
    for i in range(m):
        # OU diffusion `c` is conventionally σ; render with that symbol.
        labels.append(_greek(r"\sigma", i if m > 1 else None))
    if FIT_D:
        for i in range(m):
            labels.append(_greek("d",     i if m > 1 else None))
    n_rho = m * (m - 1) // 2
    for k in range(n_rho):
        labels.append(_greek(r"\rho",     k if n_rho > 1 else None))
    labels.append(_greek(r"\delta"))
    if INDEPENDENT_POLY:
        for i in range(m):
            labels.append(_greek(r"\beta", i if m > 1 else None))
        if N_poly >= 5:
            for i in range(m):
                labels.append(_greek(r"\gamma", i if m > 1 else None))
            for i in range(m):
                labels.append(_greek("K",       i if m > 1 else None))
    else:
        labels.append(_greek(r"\beta"))
        if N_poly >= 5:
            labels.append(_greek(r"\gamma"))
            labels.append(_greek("K"))
    labels.append(r"$p_e$")
    return labels


def _is_skipped(label):
    return any(label == p or label.startswith(p + "[")
                for p in SLICE_SKIP_PATTERNS)


def make_slice_grid(lo, hi, mle, n_pts=SLICE_GRID_PTS,
                     rel_range=SLICE_REL_RANGE,
                     min_half_frac=SLICE_MIN_HALF):
    """Linspace around `mle` clipped to [lo, hi]. Returns None if pinned"""
    if abs(hi - lo) < 1e-12:
        return None
    half = max(rel_range * abs(mle), min_half_frac * (hi - lo))
    a = max(lo, mle - half)
    b = min(hi, mle + half)
    if b - a < 1e-10:
        return None
    return np.linspace(a, b, n_pts)


def run_slice_figure(data, baseline_cells):
    """For each cell in SLICE_DEGS, sweep each parameter on a grid around its
    MLE (others held fixed) and plot logL(param) as small multiples. Marks
    the MLE with a red line/dot. Saves one PNG per (m, N_poly)."""
    print("\n=== Block: 1-D logL slice figure (OU) ===")
    degs = [d for d in SLICE_DEGS if d in baseline_cells]
    if not degs:
        print("  ! No baseline cells available for slice figure. Skipping.")
        return None

    y_resid  = data["y_resid"]
    maturity = data["maturity"]
    delivery = data["delivery"]

    out_paths = []
    for N in degs:
        v_mle  = np.asarray(baseline_cells[N]["params_vec"], dtype=float).copy()
        # Use the SAME bounds the BIC fit saw — Kalman_filter_LD.make_bounds
        # (NOT the LLR-local make_bounds_capped, whose tighter sweep defaults
        # would mark interior MLEs as out-of-range and incorrectly tag them
        # as pinned).
        bounds = ld.make_bounds(
            M_FACTORS, N, fit_d=FIT_D,
            independent_poly=INDEPENDENT_POLY,
        )
        labels = param_labels(M_FACTORS, N)
        if not (len(v_mle) == len(bounds) == len(labels)):
            print(f"  [skip N={N}] vector/bound/label length mismatch "
                  f"(v={len(v_mle)}, b={len(bounds)}, l={len(labels)})")
            continue

        logL_mle = evaluate_logL(v_mle, y_resid, maturity, delivery,
                                  DT, M_FACTORS, N)
        print(f"\n  m={M_FACTORS}  N={N}   logL(MLE) = {logL_mle:.4f}   "
              f"({len(v_mle)} params, "
              f"{SLICE_GRID_PTS} pts each → "
              f"{len(v_mle) * SLICE_GRID_PTS} EKF evaluations)")

        slices = []                  # (label, grid, logL_grid, mle_val, status)
        t0 = time.time()
        for i in range(len(v_mle)):
            if _is_skipped(labels[i]):
                continue              # skip — saves both screen real-estate and EKF evals
            lo, hi = bounds[i]
            mle_val = v_mle[i]
            grid    = make_slice_grid(lo, hi, mle_val)
            status  = ""
            if grid is None:
                # Pinned slot — sweep a default range AROUND the pinned value
                # so we can confirm it sits at a local optimum (or on a flat
                # region). Bounds are ignored here on purpose.
                grid = np.linspace(mle_val - SLICE_PINNED_HALF,
                                    mle_val + SLICE_PINNED_HALF,
                                    SLICE_GRID_PTS)
                status = "pinned"
            else:
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
                                       DT, M_FACTORS, N)
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
                colour = "#888888"        # gray — pinned, slice is diagnostic only
            elif status in ("lower", "upper"):
                colour = "#D62728"        # red — MLE pegged at a bound
            else:
                colour = "#1F77B4"        # blue — interior MLE
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
            ax.set_title(label + tag, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(len(slices), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.tight_layout()
        out_path = os.path.join(
            FIG_DIR,
            f"slice_ou_m{M_FACTORS}_N{N}{FLAG_TAG}_{PERIOD_TAG}.png",
        )
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    saved -> {out_path}")
        out_paths.append(out_path)

    return out_paths


# ---------------------------------------------------------------
# Block 4 — CDF figure (deg 1 / 3 / 5 baselines, 500 in-history paths)
# ---------------------------------------------------------------

def _simulate_inhistory_prices(params, data, N_poly, n_paths, rng):
    """500 in-history paths starting from the EKF posterior X[0], then
    folded into prices via compute_observations + Stage A seasonality.
    Returns (n_paths, n_days, n_c) EUR/MWh price array."""
    y_resid     = data["y_resid"]
    maturity    = data["maturity"]
    delivery    = data["delivery"]
    trading     = data["trading"]
    seas_beta   = data["seas_beta"]
    annual_h    = data["annual_h"]
    price_scale = data["price_scale"]

    n_days, n_c = maturity.shape

    # Filter through history once to get x_post[0], P_post[0].
    x0 = ld._mu_P(params).reshape(-1, 1)
    # Full stationary covariance with rho-driven cross terms (was diagonal).
    P0 = ld.stationary_cov(params)
    _, _, state_filt, state_cov_d, *_unused = filter_to_end(
        params, x0, P0, y_resid, maturity, delivery,
        DT, N_pricing=N_poly,
    )
    x_start = np.asarray(state_filt[0]).reshape(-1)
    P_start = np.diag(np.maximum(state_cov_d[0], 0.0))

    # Sample n_paths state trajectories and fold to prices.
    state_paths = simulate_state_paths(
        params, x_start, P_start, DT, n_steps=n_days,
        n_paths=n_paths, rng=rng, sample_init=True,
    )                                                # (n_paths, n_days+1, m)
    state_paths_obs = state_paths[:, 1:, :]          # drop t=0

    y_norm_sim = compute_observations(
        params, state_paths_obs,
        T_step=maturity, delta_step=delivery, N_pricing=N_poly,
    )                                                # (n_paths, n_days, n_c)

    if CDF_OBS_NOISE:
        R_diag = ld.precompute_R(
            maturity, params.p_e, tau_ref=ld.TAU_REF_DEFAULT,
        )
        y_norm_sim = (y_norm_sim
                      + rng.standard_normal((n_paths, n_days, n_c))
                        * np.sqrt(R_diag)[None, :, :])

    # Add the (deterministic) seasonal mean and rescale.
    _, S_hist, _ = ld.build_seasonality_matrix(
        np.asarray(trading[:, 0]), maturity, delivery,
        np.zeros((n_days, n_c)),
        annual_h=annual_h,
    )
    g_bar = (S_hist @ seas_beta).reshape(n_days, n_c)

    sim_prices_eur = price_scale * (g_bar[None, :, :] + y_norm_sim)
    return sim_prices_eur


def _compute_xlim(observed, simulated):
    """Display-only x-range for a single (observed, simulated) panel."""
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
    """Build empirical CDFs of `observed` (1D) and `simulated` (1D pooled"""
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


def _save_thesis_histograms(degs, sim_by_deg, y_matrix, n_bins=50):
    """Density-histogram counterpart of the CDF overlay figure. Each panel"""
    fig, axes = plt.subplots(1, len(degs),
                              figsize=(4.6 * len(degs), 3.6),
                              sharey=False)
    if len(degs) == 1:
        axes = [axes]

    pooled_ks = {}
    for ci, N in enumerate(degs):
        ax     = axes[ci]
        sim    = sim_by_deg[N].ravel()
        obs    = y_matrix.ravel()
        obs    = obs[np.isfinite(obs)]
        sim    = sim[np.isfinite(sim)]

        # Compute the pooled KS for the terminal print.
        _, _, _, ks = _ecdf_pair(obs, sim)
        pooled_ks[N] = ks

        # Use the existing display-only x-range so a few extreme simulated
        # prices don't smear the histogram. Bins are shared across observed
        # and simulated within a panel for honest visual comparison.
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
        f"Pooled price density (all maturities)  -  m={M_FACTORS}  "
        f"-  {N_PATHS_CDF} paths",
        fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(THESIS_DIR,
                             f"histogram_m{M_FACTORS}{FLAG_TAG}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return out_path, pooled_ks


def _save_thesis_diff(degs, sim_by_deg, y_matrix):
    """1 row x len(degs) col |F_sim - F_obs| panels, pooled across maturities."""
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
        f"Pooled CDF difference (all maturities)  -  m={M_FACTORS}",
        fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(THESIS_DIR,
                             f"cdf_diff_m{M_FACTORS}{FLAG_TAG}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return out_path


def run_cdf_figure(data, baseline_cells):
    print("\n=== Block 4: CDF overlay figure (deg 1 / 3 / 5) ===")
    degs = [d for d in CDF_DEGS if d in baseline_cells]
    if not degs:
        print("  ! No baseline cells available for CDF figure. Skipping.")
        return None

    y_matrix = data["y_matrix"]
    n_c      = y_matrix.shape[1]

    # Simulate once per degree, cache the (n_paths, n_days, n_c) array.
    rng = np.random.default_rng(SEED)
    sim_by_deg = {}
    for N in degs:
        v = baseline_cells[N]["params_vec"]
        params = ld.unpack_ld(
            v, M_FACTORS, N_poly=N, fit_d=FIT_D,
            independent_poly=INDEPENDENT_POLY)
        print(f"  simulating {N_PATHS_CDF} in-history paths for N={N} ...")
        sim_by_deg[N] = _simulate_inhistory_prices(
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

    ks_summary = {}                                   # {(N, cname): ks}
    for ci, N in enumerate(degs):
        sim_eur = sim_by_deg[N]                       # (n_paths, n_days, n_c)
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

            # Top: CDF overlay
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

            # Bottom: |F_sim - F_obs|
            ax_d.fill_between(grid, 0.0, np.abs(F_sim - F_obs),
                              color="#7F7F7F", alpha=0.45, step="post")
            ax_d.axhline(ks, color="#D62728", ls="--", lw=0.8,
                         alpha=0.6, label=f"KS = {ks:.3f}")
            ax_d.set_ylabel(f"{cname}  |dF|")
            ax_d.set_xlabel("EUR / MWh")
            ax_d.legend(loc="upper right", frameon=False, fontsize=8)
            ax_d.grid(True, alpha=0.3)

            # Apply the display-only cap (KS already computed on full data).
            if xlim is not None:
                ax_cdf.set_xlim(*xlim)
                ax_d.set_xlim(*xlim)
                # Hint to the reader when the simulated tail extends past the
                # display range — easy to mistake a clipped tail for "no tail"
                # otherwise.
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
        f"Empirical vs simulated CDFs  -  m={M_FACTORS}  "
        f"({', '.join(CONTRACT_LABELS)})  -  500 paths",
        fontsize=11, y=1.0)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR,
                             f"cdf_overlay_m{M_FACTORS}{FLAG_TAG}.png")
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

    # Thesis-ready figures: density histogram + pooled |dF| difference,
    # pooled across maturities. KS values are printed to terminal only,
    # not annotated on the histogram figure.
    _, pooled_ks = _save_thesis_histograms(degs, sim_by_deg, y_matrix)
    _save_thesis_diff(degs, sim_by_deg, y_matrix)
    print("\n  Pooled KS (all maturities, sim vs observed CDF):")
    for N in degs:
        print(f"    deg {N}  KS = {pooled_ks[N]:.4f}")
    return out_path


# ---------------------------------------------------------------
# In-sample RMSE / diagnostics
#
# Ports the in-sample half of the standalone compute_rmse_OU.py: filter the
# Stage-B residual through the EKF using the saved (full-sample) params and
# collect residuals, innovation covariances, filtered states, predictions.
# OOS is intentionally skipped — that's still in compute_rmse_OU.py if you
# want it.
# ---------------------------------------------------------------

RMSE_DEGS    = (1, 3, 5)        # which N_poly cells to score
RMSE_NPZ_DIR = os.path.join(OUT_DIR,
                              f"predictions_ou_in_sample_{PERIOD_TAG}")
# CSV filenames are built inside run_in_sample_rmse() using the current
# M_FACTORS / FLAG_TAG, mirroring the rest of this file's per-cell paths.


def _ekf_run_with_diagnostics_ou(params, x0, P0,
                                  y_obs, T, delta, dt, N_pricing,
                                  tau_ref=ld.TAU_REF_DEFAULT):
    """Drive ld.EKF_step over the historical horizon AND collect diagnostics:"""
    n_steps, n_c = y_obs.shape
    m = len(params.theta)

    p_T = ld.build_poly_nd(params, m, N_pricing)
    G   = ld.infinitesimal_generator(params.a, params.b,
                                      params.c, params.d,
                                      params.rho, N_pricing)
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
            x_prior, Q = ld.f_OU(params, x, dt)
            A_jac      = ld.A_OU(params, x, dt)
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
            I_KH        = np.eye(m) - K @ H_jac
            P_post      = I_KH @ P_prior @ I_KH.T + K @ R_mat @ K.T
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
    """Pooled and per-contract RMSE (normalised + EUR/MWh)."""
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
    print("\n=== In-sample RMSE / EKF diagnostics ===")
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

    for N in RMSE_DEGS:
        pf_path = os.path.join(OUT_DIR, params_filename(M_FACTORS, N))
        if not os.path.exists(pf_path):
            print(f"  [m={M_FACTORS} N={N}] missing {pf_path}, skipping")
            continue
        v = np.load(pf_path)
        try:
            params = ld.unpack_ld(
                v, M_FACTORS, N_poly=N,
                fit_d=False,
                independent_poly=INDEPENDENT_POLY,
            )
        except Exception as exc:
            print(f"  [m={M_FACTORS} N={N}] unpack failed ({exc}), skipping")
            continue
        x0 = ld._mu_P(params).reshape(-1, 1)
        # Full stationary covariance with rho-driven cross terms (was diagonal).
        P0 = ld.stationary_cov(params)

        t0 = time.time()
        print(f"  [m={M_FACTORS} N={N}]  filtering ...", flush=True)
        out = _ekf_run_with_diagnostics_ou(
            params, x0, P0,
            y_resid, maturity, delivery, DT, N_pricing=N,
        )
        elapsed = time.time() - t0

        r          = out["resid"]
        y_pred_eur = price_scale * (g_bar + out["y_pred"])

        npz_path = os.path.join(
            RMSE_NPZ_DIR,
            f"predictions_m{M_FACTORS}_N{N}{FLAG_TAG}_in_sample.npz")
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
        f"rmse_insample_ou_m{M_FACTORS}{FLAG_TAG}_{PERIOD_TAG}.csv")
    bias_csv     = os.path.join(OUT_DIR,
        f"rmse_insample_ou_bias_m{M_FACTORS}{FLAG_TAG}_{PERIOD_TAG}.csv")
    sigma_csv    = os.path.join(OUT_DIR,
        f"rmse_insample_ou_sigma_m{M_FACTORS}{FLAG_TAG}_{PERIOD_TAG}.csv")
    coverage_csv = os.path.join(OUT_DIR,
        f"rmse_insample_ou_coverage_m{M_FACTORS}{FLAG_TAG}_{PERIOD_TAG}.csv")
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
    """CLI flags let the user run each block independently from the shell."""
    p = argparse.ArgumentParser(
        description="LLR test framework for LD_PM (OU) polynomial degrees.")
    p.add_argument("--baseline", action="store_true",
                    help="Run the baseline LLR block (deg 1/3/5 from saved params).")
    p.add_argument("--deg3", action="store_true",
                    help="Run the deg-3 p_beta upper-bound sweep (5 refits).")
    p.add_argument("--deg5", action="store_true",
                    help="Run the deg-5 (p_beta, p_gamma) joint 5x5 sweep "
                         "(25 refits — the slowest block).")
    p.add_argument("--cdf", action="store_true",
                    help="Run the CDF overlay figure block (uses baseline params).")
    p.add_argument("--slice", action="store_true",
                    help="Run the 1-D logL slice figure (Sun Fig 6.3 style).")
    p.add_argument("--rmse", action="store_true",
                    help="Run the in-sample RMSE / EKF diagnostics block.")
    p.add_argument("--all", action="store_true",
                    help="Run every block (overrides RUN, equivalent to "
                         "--baseline --deg3 --deg5 --cdf --slice --rmse).")
    p.add_argument("--m", type=int, choices=(1, 2, 3, 4), default=None,
                    help="Override M_FACTORS for this run only "
                         "(supported: 1, 2, 3, 4).")
    p.add_argument("--lam", dest="lam", action="store_true", default=None,
                    help="Override FIT_LAM=True for this run (load and refit "
                         "with lam[0..m-1] free). Default: inherit from "
                         "BIC_monthly_OU.")
    p.add_argument("--no-lam", dest="lam", action="store_false",
                    help="Override FIT_LAM=False for this run.")
    p.add_argument("--indep", dest="indep", action="store_true", default=None,
                    help="Override INDEPENDENT_POLY=True (per-factor "
                         "polynomial map). Default: inherit from "
                         "BIC_monthly_OU.")
    p.add_argument("--no-indep", dest="indep", action="store_false",
                    help="Override INDEPENDENT_POLY=False.")
    return p.parse_args()


def _resolve_run_flags(args):
    """Return a fresh RUN-style dict honouring CLI overrides."""
    if args.all:
        return {"baseline_llr": True, "deg3_beta_sweep": True,
                "deg5_joint_sweep": True, "cdf_figure": True,
                "slice_figure": True, "rmse_insample": True}
    any_block = (args.baseline or args.deg3 or args.deg5
                  or args.cdf or args.slice or args.rmse)
    if any_block:
        return {"baseline_llr":     bool(args.baseline),
                "deg3_beta_sweep":  bool(args.deg3),
                "deg5_joint_sweep": bool(args.deg5),
                "cdf_figure":       bool(args.cdf),
                "slice_figure":     bool(args.slice),
                "rmse_insample":    bool(args.rmse)}
    return dict(RUN)        # fall back to the file-level config


def main():
    args = _parse_args()
    # Apply CLI overrides BEFORE any helper that reads these globals (every
    # helper looks them up at call time, so updating here propagates).
    global M_FACTORS, FIT_LAM, INDEPENDENT_POLY, INDEP_TAG, LAM_TAG, FLAG_TAG
    if args.m is not None:
        M_FACTORS = int(args.m)
    if args.lam is not None:
        FIT_LAM = bool(args.lam)
    if args.indep is not None:
        INDEPENDENT_POLY = bool(args.indep)
    # Re-derive the filename tags so params_filename(...) picks the right
    # .npy after any of the above overrides.
    INDEP_TAG = "_indep" if INDEPENDENT_POLY else ""
    LAM_TAG   = "_lam"   if FIT_LAM         else ""
    FLAG_TAG  = INDEP_TAG + LAM_TAG
    run_flags = _resolve_run_flags(args)

    print(f"=== LLR_monthly_OU :  M_FACTORS={M_FACTORS}  "
          f"INDEPENDENT_POLY={INDEPENDENT_POLY}  FIT_LAM={FIT_LAM} ===")
    print(f"OUT_DIR = {OUT_DIR}")
    print(f"FIG_DIR = {FIG_DIR}")
    print(f"RUN     = {run_flags}")
    data = load_panels_and_residual()

    baseline_cells = {}
    if run_flags["baseline_llr"]:
        baseline_cells = run_baseline_llr(data)

    if run_flags["deg3_beta_sweep"]:
        if not baseline_cells:
            # Fallback: load deg-1 baseline only so the LLR has a reference.
            print("  (loading deg-1 baseline so the bound sweep can LLR vs it) ...")
            baseline_cells = run_baseline_llr(data)
        run_deg3_beta_sweep(data, baseline_cells)

    if run_flags["deg5_joint_sweep"]:
        if not baseline_cells:
            print("  (loading deg-1 baseline so the bound sweep can LLR vs it) ...")
            baseline_cells = run_baseline_llr(data)
        run_deg5_joint_sweep(data, baseline_cells)

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

    print("\nAll requested LLR_monthly_OU blocks finished.")


if __name__ == "__main__":
    main()
