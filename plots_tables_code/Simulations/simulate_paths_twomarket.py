"""Forward-simulate price paths under the calibrated two-market Kalman model.

"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

import GetData as gd                                  # noqa: F401  (parity)
import Kalman_filter_TwoMarket as tm
import BIC_monthly_twomarkets   as drv


# ---------------------------------------------------------------
# Inherited config (panel, thinning, period tag) from BIC driver
# ---------------------------------------------------------------

OUT_DIR       = drv.OUT_DIR
PERIOD_TAG    = drv.PERIOD_TAG
FIG_DIR       = os.path.join(OUT_DIR,
                              f"figures_simulation_twomarket_{PERIOD_TAG}")
THESIS_DIR    = os.path.join(OUT_DIR, f"figures_thesis_{PERIOD_TAG}")
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(THESIS_DIR, exist_ok=True)

CONTRACT_LABELS = list(drv.SUBSET_LABELS)
TAU_REF         = drv.TAU_REF

# Match calibration cadence — drv.DT_EKF is 7/365 (weekly) or 1/252 (daily).
DT_SIM = drv.DT_EKF


def params_filename(m, N_poly):
    return f"params_{PERIOD_TAG}_twomarket_m{m}_N{N_poly}.npy"


def _legacy_params_filename(m, N_poly):
    """Fallback for pre-period-tag files."""
    return f"params_monthly_twomarket_m{m}_N{N_poly}.npy"


MODEL_SPECS = {f"m{m}n{N_poly}": dict(m=m, N_poly=N_poly,
                                         label=f"TwoMarket m={m} N={N_poly}")
               for m in drv.M_GRID for N_poly in drv.N_POLY_GRID}


# ---------------------------------------------------------------
# Data loading (joint DE + FR Stage B residual)
# ---------------------------------------------------------------

def load_panels_and_residuals():
    """Load Stage A and Stage B panels for both markets, fit per-market
    Stage A seasonality, return the per-market Stage B residual and all
    metadata the simulator needs (price scales, seas_beta, etc.)."""
    print("Loading two-market Stage A panels (DE + FR) ...")
    (y_a_DE, mat_a_DE, del_a_DE, tra_a_DE), \
    (y_a_FR, mat_a_FR, del_a_FR, tra_a_FR) = drv.load_stage_a_data()
    print(f"  Stage A DE: {y_a_DE.shape[0]} days × {y_a_DE.shape[1]} contracts")
    print(f"  Stage A FR: {y_a_FR.shape[0]} days × {y_a_FR.shape[1]} contracts")

    print("\nLoading joint Stage B panel ...")
    y1, y2, mat_1, mat_2, del_1, del_2, trading = drv.load_stage_b_data()
    print(f"  Stage B (joined): {y1.shape[0]} days × {y1.shape[1]} contracts "
          f"per market ({CONTRACT_LABELS})")

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
              f"DE {idx_a_DE}, FR {idx_a_FR}, B {idx_b} rows dropped; "
              f"new Stage B size = {y1.shape[0]}")

    price_scale_1 = float(y_a_DE.mean())
    price_scale_2 = float(y_a_FR.mean())
    y1_a_norm = y_a_DE / price_scale_1
    y2_a_norm = y_a_FR / price_scale_2
    y1_norm   = y1     / price_scale_1
    y2_norm   = y2     / price_scale_2
    print(f"  price_scale DE = {price_scale_1:.4f} EUR/MWh")
    print(f"  price_scale FR = {price_scale_2:.4f} EUR/MWh")

    # Per-market Stage A seasonality grid.
    print("\nFitting Stage A seasonality grid (DE) ...")
    best_1 = None
    for ah in drv.ANNUAL_GRID:
        info = tm.seasonality_bic(tra_a_DE[:, 0], mat_a_DE, del_a_DE,
                                   y1_a_norm, ah)
        if best_1 is None or info["BIC"] < best_1["BIC"]:
            best_1 = info
    seas_beta_1 = best_1["beta"]

    print("Fitting Stage A seasonality grid (FR) ...")
    best_2 = None
    for ah in drv.ANNUAL_GRID:
        info = tm.seasonality_bic(tra_a_FR[:, 0], mat_a_FR, del_a_FR,
                                   y2_a_norm, ah)
        if best_2 is None or info["BIC"] < best_2["BIC"]:
            best_2 = info
    seas_beta_2 = best_2["beta"]

    # Rebuild design on Stage B for each market.
    n_t, n_c = mat_1.shape
    _, S_1, _ = tm.build_seasonality_matrix(
        trading[:, 0], mat_1, del_1, y1_norm,
        annual_h=int(best_1["annual_h"]),
    )
    _, S_2, _ = tm.build_seasonality_matrix(
        trading[:, 0], mat_2, del_2, y2_norm,
        annual_h=int(best_2["annual_h"]),
    )
    g_bar_1 = (S_1 @ seas_beta_1).reshape(n_t, n_c)
    g_bar_2 = (S_2 @ seas_beta_2).reshape(n_t, n_c)
    y_resid_1 = y1_norm - g_bar_1
    y_resid_2 = y2_norm - g_bar_2
    print(f"\n  Stage B residual DE: mean={y_resid_1.mean():+.5f}  "
          f"std={y_resid_1.std():.5f}")
    print(f"  Stage B residual FR: mean={y_resid_2.mean():+.5f}  "
          f"std={y_resid_2.std():.5f}")

    # Data already at calibration cadence (weekly when USE_WEEKLY_SAMPLING).
    print(f"  Calibration cadence: "
          f"{'weekly ISO-Mon' if drv.USE_WEEKLY_SAMPLING else 'daily'}; "
          f"Stage B size = {y1.shape[0]}, dt = {DT_SIM:.6f}")

    return dict(
        y1=y1, y2=y2, mat_1=mat_1, mat_2=mat_2, del_1=del_1, del_2=del_2,
        trading=trading,
        y_resid_1=y_resid_1, y_resid_2=y_resid_2,
        g_bar_1=g_bar_1,    g_bar_2=g_bar_2,
        price_scale_1=price_scale_1, price_scale_2=price_scale_2,
        seas_beta_1=seas_beta_1, seas_beta_2=seas_beta_2,
        annual_h_1=int(best_1["annual_h"]),
        annual_h_2=int(best_2["annual_h"]),
    )


# ---------------------------------------------------------------
# Filter through history (records per-step posterior)
# ---------------------------------------------------------------

def filter_to_end(params, data, N_pricing):
    """Run the joint two-market EKF through the in-sample window and
    return diagnostics needed for the extension warm-start, the in-
    sample-fit figure, and the latent-state figure.

    Returns
    -------
    x_final, P_final : final posterior state and covariance
    state_filt       : posterior state trajectory (x_post[t])
    state_cov_d      : diagonal of posterior covariance per step
    y_pred_eur_1, y_pred_eur_2 : (n_steps, n_c) one-step-ahead predicted
        prices in raw EUR/MWh per market, computed from x_prior[t] via
        h(x_prior) and the price-scale / seasonality un-normalisation.
        These are what `Var(Y - h(x_prior))` integrates to give the EKF
        innovation RMSE.
    """
    n_steps, n_c = data["y_resid_1"].shape
    helper = tm._PredictHelper(params, DT_SIM, N_pred=2)
    G_Q    = tm.infinitesimal_generator_two_market(params, N=N_pricing,
                                                    use_P=False)
    p_T_1  = tm.build_poly_market(params, market=1, N=N_pricing)
    p_T_2  = tm.build_poly_market(params, market=2, N=N_pricing)
    Mp_1_all, expm_cache = tm._precompute_Mp(G_Q, p_T_1,
                                              data["mat_1"], data["del_1"])
    Mp_2_all, _          = tm._precompute_Mp(G_Q, p_T_2,
                                              data["mat_2"], data["del_2"],
                                              expm_cache=expm_cache)
    R_1 = tm.precompute_R(data["mat_1"], params.p_e_1, tau_ref=TAU_REF)
    R_2 = tm.precompute_R(data["mat_2"], params.p_e_2, tau_ref=TAU_REF)

    x0, P0 = tm._initial_state(params)
    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    P = np.atleast_2d(P0)
    n_state = params.n_state

    state_filt    = np.full((n_steps, n_state), np.nan)  # x_post[t]
    state_cov_d   = np.full((n_steps, n_state), np.nan)  # diag(P_post[t])
    state_prior   = np.full((n_steps, n_state), np.nan)  # x_prior[t]
    prior_cov_d   = np.full((n_steps, n_state), np.nan)  # diag(P_prior[t])
    y_pred_norm_1 = np.full((n_steps, n_c), np.nan)
    y_pred_norm_2 = np.full((n_steps, n_c), np.nan)

    for t in range(n_steps):
        # Capture x_prior, P_prior by replaying the predict step BEFORE the
        # EKF step. tm._predict gives the conditional mean and noise covariance
        # under P; A_jac is its Jacobian wrt the previous state. P_prior is
        # then A_jac @ P_post_{t-1} @ A_jac.T + Q (the usual Kalman predict).
        x_prior_t, Q_t, A_jac_t = tm._predict(
            np.asarray(x).flatten(), helper)
        P_prior_t = A_jac_t @ np.asarray(P) @ A_jac_t.T + Q_t
        state_prior[t] = x_prior_t
        prior_cov_d[t] = np.diag(P_prior_t)

        y_t      = np.concatenate([data["y_resid_1"][t].flatten(),
                                    data["y_resid_2"][t].flatten()])
        R_diag_t = np.concatenate([R_1[t], R_2[t]])
        x, P, resid, _ = tm.EKF_step_two_market(
            params, x, P, helper,
            Mp_1_all[t], Mp_2_all[t],
            y_t, R_diag_t, N_pricing,
        )
        state_filt[t]  = np.asarray(x).flatten()
        state_cov_d[t] = np.diag(P)
        # resid = y_t - h(x_prior[t]); recover the one-step-ahead
        # prediction in normalised-residual units as y_t - resid.
        n_c1 = n_c
        y_pred_norm_1[t] = data["y_resid_1"][t] - resid[:n_c1]
        y_pred_norm_2[t] = data["y_resid_2"][t] - resid[n_c1:]

    # Convert back to raw EUR/MWh: add seasonality and apply per-market
    # price scale.
    y_pred_eur_1 = data["price_scale_1"] * (y_pred_norm_1 + data["g_bar_1"])
    y_pred_eur_2 = data["price_scale_2"] * (y_pred_norm_2 + data["g_bar_2"])

    return (np.asarray(x).flatten(),    # x_final
            P,                           # P_final
            state_filt, state_cov_d,
            state_prior, prior_cov_d,
            y_pred_eur_1, y_pred_eur_2)


# ---------------------------------------------------------------
# State-path simulation under P (forward, no observation correction)
# ---------------------------------------------------------------

def _safe_chol(mat, jitter=1e-10):
    """Cholesky with eigenvalue-floor fallback for borderline-PSD matrices."""
    n = mat.shape[0]
    M = 0.5 * (mat + mat.T) + jitter * np.eye(n)
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh(M)
        ev = np.maximum(ev, jitter)
        return V @ np.diag(np.sqrt(ev))


def simulate_state_paths(params, x_start, P_start, dt, n_steps, n_paths, rng,
                          sample_init=True):
    """Forward-simulate `n_paths` joint two-market state trajectories of
    length `n_steps` starting from N(x_start, P_start). Uses the
    polynomial-preserving conditional-moment update with a Gaussian
    sample at each step. The R component is clipped to (-0.999, 0.999)
    so the correlation never leaves its state space.
    """
    helper  = tm._PredictHelper(params, dt, N_pred=2)
    n_state = params.n_state
    paths   = np.zeros((n_paths, n_steps + 1, n_state))

    x0 = np.asarray(x_start).reshape(-1)
    if sample_init:
        L0 = _safe_chol(np.atleast_2d(P_start))
        paths[:, 0, :] = x0[None, :] + rng.standard_normal((n_paths, n_state)) @ L0.T
    else:
        paths[:, 0, :] = x0[None, :]

    Ri = params.R_index
    for n in range(n_steps):
        for j in range(n_paths):
            x_prior, Q, _ = tm._predict(paths[j, n, :], helper)
            L = _safe_chol(Q)
            paths[j, n + 1, :] = x_prior + L @ rng.standard_normal(n_state)
            paths[j, n + 1, Ri] = np.clip(paths[j, n + 1, Ri], -0.999, 0.999)
    return paths


def compute_observations(params, state_paths, T_step, delta_step, N_pricing):
    """For each (path, time), evaluate the per-market polynomial-pricing
    operator at the joint state to get the dimensionless predicted
    observation y_norm_{1,2}. Returns (y_norm_1, y_norm_2), each of
    shape (n_paths, n_steps, n_c).
    """
    n_paths, n_steps, _ = state_paths.shape
    n_c = T_step.shape[1]
    G_Q   = tm.infinitesimal_generator_two_market(params, N=N_pricing,
                                                   use_P=False)
    p_T_1 = tm.build_poly_market(params, market=1, N=N_pricing)
    p_T_2 = tm.build_poly_market(params, market=2, N=N_pricing)
    Mp_1_all, expm_cache = tm._precompute_Mp(G_Q, p_T_1, T_step, delta_step)
    Mp_2_all, _          = tm._precompute_Mp(G_Q, p_T_2, T_step, delta_step,
                                              expm_cache=expm_cache)

    y_norm_1 = np.zeros((n_paths, n_steps, n_c))
    y_norm_2 = np.zeros((n_paths, n_steps, n_c))
    for t in range(n_steps):
        Mp_1_t = Mp_1_all[t]
        Mp_2_t = Mp_2_all[t]
        for j in range(n_paths):
            H_x = tm.build_H(state_paths[j, t, :], N_pricing)
            y_norm_1[j, t, :] = np.array([H_x @ Mp for Mp in Mp_1_t])
            y_norm_2[j, t, :] = np.array([H_x @ Mp for Mp in Mp_2_t])
    return y_norm_1, y_norm_2


# ---------------------------------------------------------------
# Simulation entry points
# ---------------------------------------------------------------

def simulate_in_history(params, n_paths, dt, data, x_start, P_start, N_pricing,
                          rng):
    """Forward-simulate `n_paths` paths over the in-sample horizon
    starting from N(x_start, P_start). Reassembles per-market prices via
    Stage A seasonality and price_scale.
    Returns (sim_eur_1, sim_eur_2) each of shape (n_paths, n_days, n_c).
    """
    n_days, n_c = data["mat_1"].shape
    state_paths = simulate_state_paths(
        params, x_start, P_start, dt,
        n_steps=n_days, n_paths=n_paths, rng=rng, sample_init=True,
    )
    state_paths_obs = state_paths[:, 1:, :]   # drop t=0 to align with obs
    y_norm_1, y_norm_2 = compute_observations(
        params, state_paths_obs,
        T_step=data["mat_1"], delta_step=data["del_1"],
        N_pricing=N_pricing,
    )
    sim_eur_1 = data["price_scale_1"] * (data["g_bar_1"][None, :, :] + y_norm_1)
    sim_eur_2 = data["price_scale_2"] * (data["g_bar_2"][None, :, :] + y_norm_2)
    return sim_eur_1, sim_eur_2, state_paths


def simulate_extension(params, x_final, P_final, n_paths, n_steps, dt,
                        fut_t, fut_mat, fut_del, fut_g_bar_1, fut_g_bar_2,
                        price_scale_1, price_scale_2, N_pricing, rng,
                        add_obs_noise=False):
    """Forward extension from the EKF posterior at end-of-sample."""
    state_paths = simulate_state_paths(
        params, x_final, P_final, dt,
        n_steps=n_steps, n_paths=n_paths, rng=rng, sample_init=False,
    )
    state_paths_obs = state_paths[:, 1:, :]
    y_norm_1, y_norm_2 = compute_observations(
        params, state_paths_obs,
        T_step=fut_mat, delta_step=fut_del,
        N_pricing=N_pricing,
    )
    if add_obs_noise:
        R_1 = tm.precompute_R(fut_mat, params.p_e_1, tau_ref=TAU_REF)
        R_2 = tm.precompute_R(fut_mat, params.p_e_2, tau_ref=TAU_REF)
        y_norm_1 = y_norm_1 + rng.standard_normal(y_norm_1.shape) * np.sqrt(R_1)[None, :, :]
        y_norm_2 = y_norm_2 + rng.standard_normal(y_norm_2.shape) * np.sqrt(R_2)[None, :, :]
    sim_eur_1 = price_scale_1 * (fut_g_bar_1[None, :, :] + y_norm_1)
    sim_eur_2 = price_scale_2 * (fut_g_bar_2[None, :, :] + y_norm_2)
    return sim_eur_1, sim_eur_2, state_paths


# ---------------------------------------------------------------
# Rolling forward schedule (mirrors the OU/Jacobi simulators)
# ---------------------------------------------------------------

def _detect_cycle_len(maturity_col, fallback=21):
    diffs = np.diff(np.asarray(maturity_col, dtype=float))
    roll_idx = np.where(diffs > 0)[0]
    if len(roll_idx) < 2:
        return int(fallback)
    return int(np.median(np.diff(roll_idx)))


def _build_per_col_rolling_schedule(mat_hist, del_hist, n_steps, cycle_lens):
    n_c = mat_hist.shape[1]
    fut_mat = np.empty((n_steps, n_c))
    fut_del = np.empty((n_steps, n_c))
    for c in range(n_c):
        cl = max(1, int(cycle_lens[c]))
        mat_cyc = np.asarray(mat_hist[-cl:, c], dtype=float)
        del_cyc = np.asarray(del_hist[-cl:, c], dtype=float)
        idx = np.arange(n_steps) % cl
        fut_mat[:, c] = mat_cyc[idx]
        fut_del[:, c] = del_cyc[idx]
    return fut_mat, fut_del


# ---------------------------------------------------------------
# Plotting (per-market in-history overlay + extension + histograms)
# ---------------------------------------------------------------

_PALETTE = ["#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]
_HIST_OBS_COLOR = "#1F77B4"
_HIST_SIM_COLOR = "#D62728"


def _trading_to_dt(years):
    """Convert decimal trading-years to numpy datetime64 (1-Jan-Y + days)."""
    years = np.asarray(years, dtype=float)
    out = np.empty(years.shape, dtype="datetime64[D]")
    for i, y in enumerate(years):
        yr = int(y)
        doy = int(round((y - yr) * 365.0))
        out[i] = np.datetime64(f"{yr}-01-01") + doy
    return out


def _ks_2samp(a, b):
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    grid = np.unique(np.concatenate([a, b]))
    Fa = np.searchsorted(a, grid, side="right") / len(a)
    Fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))


def plot_in_history_overlay(hist_dt, y_obs, sim_paths, contract_labels,
                              market_label, save_path, path_idx=0):
    """Solid = observed, dashed = simulated path `path_idx`, one colour per contract."""
    n_c = y_obs.shape[1]
    fig, ax = plt.subplots(figsize=(11, 5))
    for c, cname in enumerate(contract_labels):
        col = _PALETTE[c % len(_PALETTE)]
        ax.plot(hist_dt, y_obs[:, c],
                color=col, lw=1.1, alpha=0.95, label=f"{cname} obs")
        ax.plot(hist_dt, sim_paths[path_idx, :, c],
                color=col, lw=1.1, alpha=0.65, ls="--", label=f"{cname} sim")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("EUR / MWh")
    ax.legend(ncol=n_c, fontsize=8, loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_in_history_xprior_fit_one_contract(hist_dt, y_obs_c, y_pred_eur_c,
                                              contract_label, market_label,
                                              save_path):
    """Single-contract one-step-ahead fit: observed (solid blue) vs the
    EKF's h(x_prior[t]) prediction (dashed red). One figure per contract
    so each maturity gets a readable panel. The per-contract RMSE is
    annotated in the corner — this is the same number the EKF
    innovation RMSE integrates to."""
    rmse_c = float(np.sqrt(np.nanmean((y_obs_c - y_pred_eur_c) ** 2)))
    bias_c = float(np.nanmean(y_pred_eur_c - y_obs_c))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist_dt, y_obs_c,
            color="#1F77B4", lw=1.1, alpha=0.95,
            label=f"{contract_label} observed")
    ax.plot(hist_dt, y_pred_eur_c,
            color="#D62728", lw=1.0, alpha=0.85, linestyle="--",
            label=f"{contract_label} h(x_prior)")
    ax.text(0.99, 0.02,
            f"RMSE = {rmse_c:.3f} EUR/MWh   "
            f"bias = {bias_c:+.3f} EUR/MWh",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#999999", alpha=0.85))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("EUR / MWh")
    ax.set_title(f"{market_label} {contract_label} one-step-ahead in-sample fit "
                  "(h(x_prior) vs observed)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_latent_states_twomarket(hist_dt, state_filt, state_cov_d,
                                    state_prior, prior_cov_d,
                                    params, save_path, show_post_band=True,
                                    show_prior_band=False):
    """Per-factor latent-state trace for the joint two-market state
    (Z, Y, R). Solid coloured line = posterior x_post[t] with one-sigma
    band from the posterior covariance; dashed grey line = prior
    x_prior[t] (one-step-ahead state estimate). For m_per_market=2 the
    state has five components (Z_slow, Z_fast, Y_slow, Y_fast, R)."""
    n_state = state_filt.shape[1]
    m       = params.m_per_market
    if m == 1:
        names = [r"$X^{(1)}$ (DE)", r"$X^{(2)}$ (FR)",
                 r"$X^{(3)}$ (correlation)"]
    else:  # m == 2
        names = ["Z_slow (DE)", "Z_fast (DE)",
                 "Y_slow (FR)", "Y_fast (FR)",
                 "R (correlation)"]
    while len(names) < n_state:
        names.append(f"X_{len(names)}")

    fig, axes = plt.subplots(n_state, 1,
                              figsize=(11, 1.7 * n_state + 1.0),
                              sharex=True)
    if n_state == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        # Posterior band + line
        if show_post_band:
            sd_post = np.sqrt(np.maximum(state_cov_d[:, i], 0.0))
            ax.fill_between(hist_dt,
                            state_filt[:, i] - sd_post,
                            state_filt[:, i] + sd_post,
                            color=_PALETTE[0], alpha=0.15)
        ax.plot(hist_dt, state_filt[:, i],
                color=_PALETTE[0], lw=1.0,
                label="x_post (filtered)")
        # Prior line (and optional band)
        if show_prior_band:
            sd_prior = np.sqrt(np.maximum(prior_cov_d[:, i], 0.0))
            ax.fill_between(hist_dt,
                            state_prior[:, i] - sd_prior,
                            state_prior[:, i] + sd_prior,
                            color="#666666", alpha=0.10)
        ax.plot(hist_dt, state_prior[:, i],
                color="#444444", lw=0.9, linestyle="--",
                label="x_prior (one-step-ahead)")
        ax.set_ylabel(names[i], fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        axes[-1].xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_market_with_seasonality(hist_dt, t_years, y_obs, mat, dlt,
                                    seas_beta, annual_h, price_scale,
                                    contract_labels, market_label,
                                    save_path, eval_idx=-1):

    n_c = y_obs.shape[1]
    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Stage-A reference contracts as solid coloured lines.
    for c in range(n_c):
        ax.plot(hist_dt, y_obs[:, c],
                color=_PALETTE[c % len(_PALETTE)],
                lw=1.0, alpha=0.9, label=contract_labels[c])

    # Smooth g(t): same basis, fixed mean maturity/delivery from eval_idx.
    n_g    = len(t_years)
    mat_e  = np.full((n_g, 1), float(np.mean(np.asarray(mat)[:, eval_idx])))
    del_e  = np.full((n_g, 1), float(np.mean(np.asarray(dlt)[:, eval_idx])))
    _, S_g, _ = tm.build_seasonality_matrix(
        np.asarray(t_years, dtype=float),
        mat_e, del_e,
        np.zeros((n_g, 1)),
        annual_h=int(annual_h),
    )
    g_eur = price_scale * (S_g @ np.asarray(seas_beta))
    ax.plot(hist_dt, g_eur,
            color="black", lw=1.6, linestyle="--",
            label=f"g(t) | N={int(annual_h)}")

    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              ncol=max(2, n_c + 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_raw_prices_de_fr_short_long(hist_dt, y1, y2, contract_labels,
                                       save_path):

    short_idx = 0
    long_idx  = len(contract_labels) - 1
    short_lbl = contract_labels[short_idx]
    long_lbl  = contract_labels[long_idx]

    DE_COLOR = "#1F77B4"     # blue
    FR_COLOR = "#D62728"     # red

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist_dt, y1[:, short_idx], color=DE_COLOR, ls="-",  lw=1.1,
            label=f"DE {short_lbl}")
    ax.plot(hist_dt, y1[:, long_idx],  color=DE_COLOR, ls="--", lw=1.1,
            label=f"DE {long_lbl}")
    ax.plot(hist_dt, y2[:, short_idx], color=FR_COLOR, ls="-",  lw=1.1,
            label=f"FR {short_lbl}")
    ax.plot(hist_dt, y2[:, long_idx],  color=FR_COLOR, ls="--", lw=1.1,
            label=f"FR {long_lbl}")

    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_extension(hist_dt, y_obs, sim_paths_ext, fut_dt, contract_labels,
                    market_label, save_path, n_paths_show=5):
    """Show last in-sample window in EUR/MWh + several simulated forward paths."""
    n_c = y_obs.shape[1]
    fig, axes = plt.subplots(n_c, 1, figsize=(11, 2.6 * n_c), sharex=True)
    if n_c == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        col = _PALETTE[c % len(_PALETTE)]
        ax.plot(hist_dt, y_obs[:, c], color=col, lw=1.1,
                label=f"{contract_labels[c]} obs")
        for j in range(min(n_paths_show, sim_paths_ext.shape[0])):
            ax.plot(fut_dt, sim_paths_ext[j, :, c],
                    color="#888888", lw=0.6, alpha=0.5)
        ax.axvline(hist_dt[-1], color="k", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylabel(f"{contract_labels[c]}\nEUR/MWh")
        ax.legend(loc="best", frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        axes[-1].xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_increment_histogram(y_obs, sim_paths, contract_labels,
                               market_label, save_path, n_bins=60):
    """Pooled-across-contracts Δprice histogram, observed vs simulated."""
    obs_inc = np.diff(y_obs, axis=0).ravel()
    sim_inc = np.diff(sim_paths, axis=1).ravel()
    # Always cover the full observed range; clip simulated at 0.5/99.5 so a
    # few extreme sim paths don't compress the bulk of the distribution.
    sim_lo, sim_hi = np.percentile(sim_inc, [0.5, 99.5])
    lo = min(float(np.min(obs_inc)), float(sim_lo))
    hi = max(float(np.max(obs_inc)), float(sim_hi))
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(obs_inc, bins=bins, density=True, alpha=0.45,
             color=_HIST_OBS_COLOR, label=f"observed (pooled, {y_obs.shape[1]} mats)",
             edgecolor="white", linewidth=0.3)
    ax.hist(sim_inc, bins=bins, density=True, alpha=0.45,
             color=_HIST_SIM_COLOR,
             label=f"simulated ({sim_paths.shape[0]} paths × {y_obs.shape[1]} mats)",
             edgecolor="white", linewidth=0.3)
    ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(f"{market_label} Δ price (EUR/MWh)")
    ax.set_ylabel("density")
    ax.set_xlim(lo, hi)
    ax.grid(True, alpha=0.3)
    ks = _ks_2samp(obs_inc, sim_inc)
    ax.text(0.98, 0.95, f"KS = {ks:.3f}",
             transform=ax.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#999999", alpha=0.85))
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}  (KS={ks:.3f})")


def _simulate_in_history_for_degree_twomarket(m, N_poly, *, n_paths, data,
                                                 seed):
    """In-history sim for one (m, N) cell. Returns (sim_eur_1, sim_eur_2) for
    DE and FR, or (None, None) if no params file exists for that cell."""
    pf_name = params_filename(m, N_poly)
    pf_path = os.path.join(OUT_DIR, pf_name)
    if not os.path.exists(pf_path):
        legacy = os.path.join(OUT_DIR, _legacy_params_filename(m, N_poly))
        if os.path.exists(legacy):
            pf_path = legacy
        else:
            print(f"  (skipping m={m},N={N_poly}: no params file)")
            return None, None
    v = np.load(pf_path)
    params = tm.unpack(v, m_per_market=m, N_poly=N_poly)
    x0_init, P0_init = tm._initial_state(params)
    rng = np.random.default_rng(seed)
    sim_eur_1, sim_eur_2, _ = simulate_in_history(
        params, n_paths, DT_SIM, data,
        x_start=np.asarray(x0_init).flatten(),
        P_start=np.atleast_2d(P0_init),
        N_pricing=N_poly,
        rng=rng,
    )
    return sim_eur_1, sim_eur_2


def plot_increment_histogram_by_degree(y_obs, sim_by_degree, market_idx,
                                         market_label, save_path, n_bins=60):
    """One panel per polynomial degree of pooled Δprice histograms for ONE
    market (DE or FR). 
    """
    items = [(N, sims) for N, sims in sim_by_degree.items()
             if sims is not None and sims[market_idx] is not None]
    if not items:
        print(f"  (no simulations available for {market_label} — "
              f"skipping {save_path})")
        return
    obs_inc = np.diff(np.asarray(y_obs), axis=0).ravel()
    sim_pools = []
    for _, sims in items:
        sim_market = np.asarray(sims[market_idx])
        sim_pools.append(np.diff(sim_market, axis=1).ravel())
    sim_all = np.concatenate(sim_pools)
    sim_lo, sim_hi = np.percentile(sim_all, [0.5, 99.5])
    lo = min(float(np.min(obs_inc)), float(sim_lo))
    hi = max(float(np.max(obs_inc)), float(sim_hi))
    bins = np.linspace(lo, hi, n_bins + 1)

    var_obs = float(np.var(obs_inc, ddof=1))
    print(f"  Δprice variance ({market_label}, pooled, EUR/MWh)^2:")
    print(f"    observed         = {var_obs:9.4f}")
    for (N, _), sim_pool in zip(items, sim_pools):
        v_sim = float(np.var(sim_pool, ddof=1))
        ratio = v_sim / var_obs if var_obs > 0 else float("nan")
        print(f"    simulated N={N}    = {v_sim:9.4f}   sim/obs={ratio:.3f}")

    n_panels = len(items)
    # Single row for 1-3 panels, 2-column grid beyond that. So 3 degrees
    # render as 1x3, 4 degrees as 2x2, 5+ as 2 × ceil(n/2).
    if n_panels <= 3:
        n_rows, n_cols = 1, n_panels
    else:
        n_cols = 2
        n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.6 * n_cols, 3.8 * n_rows),
                              sharey=False)
    axes = np.atleast_1d(axes).ravel()
    n_c = np.asarray(y_obs).shape[1]
    for ax, (N, sims), sim_pool in zip(axes, items, sim_pools):
        sim_market = np.asarray(sims[market_idx])
        n_paths    = sim_market.shape[0]
        ax.hist(obs_inc, bins=bins, density=True, alpha=0.45,
                color=_HIST_OBS_COLOR,
                label=f"observed (pooled, {n_c} mats)",
                edgecolor="white", linewidth=0.3)
        ax.hist(sim_pool, bins=bins, density=True, alpha=0.45,
                color=_HIST_SIM_COLOR,
                label=f"simulated ({n_paths} paths × {n_c} mats)",
                edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(f"deg N={N}")
        ax.set_xlabel(f"{market_label} Δ price (EUR/MWh)")
        ax.set_ylabel("density")
        ax.set_xlim(lo, hi)
        ax.grid(True, alpha=0.3)
        ks = _ks_2samp(obs_inc, sim_pool)
        ax.text(0.98, 0.95, f"KS = {ks:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#999999", alpha=0.85))
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    for ax_unused in axes[len(items):]:
        ax_unused.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------
# Main + CLI
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Forward-simulate price paths under the two-market model.")
    p.add_argument("--model", default=None,
                    help=f"Cell to simulate from. Options: "
                          f"{sorted(MODEL_SPECS.keys())}. Defaults to the "
                          f"smallest (m, N) in M_GRID × N_POLY_GRID.")
    p.add_argument("--n-paths", type=int, default=500)
    p.add_argument("--years",   type=float, default=2.0)
    p.add_argument("--n-paths-show", type=int, default=5)
    p.add_argument("--no-obs-noise", action="store_true",
                    help="Do not add observation noise on the extension paths.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.model is None:
        m_smallest = min(drv.M_GRID)
        n_smallest = min(drv.N_POLY_GRID)
        args.model = f"m{m_smallest}n{n_smallest}"
    if args.model not in MODEL_SPECS:
        raise SystemExit(f"Unknown --model {args.model!r}; "
                         f"choose from {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[args.model]
    m, N_poly = spec["m"], spec["N_poly"]
    label = spec["label"]

    pf_name = params_filename(m, N_poly)
    pf_path = os.path.join(OUT_DIR, pf_name)
    if not os.path.exists(pf_path):
        legacy = os.path.join(OUT_DIR, _legacy_params_filename(m, N_poly))
        if os.path.exists(legacy):
            print(f"  (using legacy {os.path.basename(legacy)} — re-run "
                  f"BIC to produce the period-tagged version)")
            pf_path = legacy
        else:
            raise SystemExit(f"Missing params file: {pf_name} (or its "
                             f"legacy alias). Run BIC_monthly_twomarkets "
                             f"first to fit cell (m={m}, N={N_poly}).")

    print(f"=== Simulating two-market model '{args.model}' ({label}) ===")
    print(f"Loading params from {pf_path}")
    v = np.load(pf_path)
    params = tm.unpack(v, m_per_market=m, N_poly=N_poly)
    print(f"  m_per_market={m}, N_poly={N_poly}")
    print(f"  kappa_Z = {params.kappa_Z}  theta_Z = {params.theta_Z}  "
          f"sigma_Z = {params.sigma_Z}  lam_Z = {params.lam_Z}")
    print(f"  kappa_Y = {params.kappa_Y}  sigma_Y = {params.sigma_Y}  "
          f"lam_Y = {params.lam_Y}")
    print(f"  kappa_R = {params.kappa_R}  theta_R = {params.theta_R}  "
          f"sigma_R = {params.sigma_R}  lam_R = {params.lam_R}")
    print(f"  p_delta_1 = {params.p_delta_1:+.4f}  p_beta_1 = {params.p_beta_1:.4f}")
    print(f"  p_delta_2 = {params.p_delta_2:+.4f}  p_beta_2 = {params.p_beta_2:.4f}")
    if N_poly >= 5:
        print(f"  p_gamma_1 = {params.p_gamma_1:.4f}  p_K_1 = {params.p_K_1:+.4f}")
        print(f"  p_gamma_2 = {params.p_gamma_2:.4f}  p_K_2 = {params.p_K_2:+.4f}")
    print(f"  p_e_1     = {params.p_e_1:.4f}      p_e_2 = {params.p_e_2:.4f}")


    _sz0 = float(params.sigma_Z[0])
    if abs(_sz0) > 1e-12:
        phi_Z_slow    = (float(params.kappa_Z[0])
                          * float(params.lam_Z[0]) / _sz0)
        coupling_slow = float(params.sigma_Y[0]) * phi_Z_slow
        print(f"  phi_Z (slow)   = {phi_Z_slow:+.4f}  "
              f"<-- DE market price of risk implied by (kappa_Z, lam_Z, sigma_Z)")
        print(f"  Girsanov coupling on Y_0 drift = "
              f"{coupling_slow:+.4f} * R_t   "
              f"<-- new under-P term added to Y_0")
    else:
        print(f"  (sigma_Z[0] = 0; Girsanov coupling skipped)")

    data = load_panels_and_residuals()

    print("\nFiltering joint Stage B residual through EKF ...")
    (x_final, P_final,
     state_filt, state_cov_d,
     state_prior, prior_cov_d,
     fit_eur_1, fit_eur_2) = filter_to_end(
        params, data, N_pricing=N_poly,
    )
    print(f"  x_final       = {x_final}")
    print(f"  diag(P_final) = {np.diag(P_final)}")

    rng = np.random.default_rng(args.seed)

    # ---- (A) In-history simulation from stationary initial law ----
    print(f"\nSimulating {args.n_paths} in-history paths from the "
          f"stationary initial law ...")
    x0_init, P0_init = tm._initial_state(params)
    sim_in_hist_eur_1, sim_in_hist_eur_2, _ = simulate_in_history(
        params, args.n_paths, DT_SIM, data,
        x_start=np.asarray(x0_init).flatten(),
        P_start=np.atleast_2d(P0_init),
        N_pricing=N_poly,
        rng=rng,
    )
    print(f"  DE sim shape: {sim_in_hist_eur_1.shape}")
    print(f"  FR sim shape: {sim_in_hist_eur_2.shape}")

    # ---- (B) Forward extension from EKF posterior at end of sample ----
    n_steps_ext = int(round(args.years / DT_SIM))
    n_c         = data["mat_1"].shape[1]
    cycle_lens  = [_detect_cycle_len(data["mat_1"][:, c]) for c in range(n_c)]
    fut_mat_1, fut_del_1 = _build_per_col_rolling_schedule(
        data["mat_1"], data["del_1"], n_steps_ext, cycle_lens)
    fut_mat_2, fut_del_2 = _build_per_col_rolling_schedule(
        data["mat_2"], data["del_2"], n_steps_ext, cycle_lens)
    t_last = float(data["trading"][-1, 0])
    fut_t  = t_last + DT_SIM * np.arange(1, n_steps_ext + 1)

    # Future-horizon seasonality on the same Fourier bases fitted in Stage A.
    _, S_fut_1, _ = tm.build_seasonality_matrix(
        fut_t, fut_mat_1, fut_del_1, np.zeros((n_steps_ext, n_c)),
        annual_h=data["annual_h_1"])
    _, S_fut_2, _ = tm.build_seasonality_matrix(
        fut_t, fut_mat_2, fut_del_2, np.zeros((n_steps_ext, n_c)),
        annual_h=data["annual_h_2"])
    g_fut_1 = (S_fut_1 @ data["seas_beta_1"]).reshape(n_steps_ext, n_c)
    g_fut_2 = (S_fut_2 @ data["seas_beta_2"]).reshape(n_steps_ext, n_c)

    print(f"\nSimulating {args.n_paths} forward-extension paths "
          f"({n_steps_ext} steps over ~{args.years} years) ...")
    rng_ext = np.random.default_rng(args.seed + 1)
    sim_ext_eur_1, sim_ext_eur_2, _ = simulate_extension(
        params, x_final, P_final,
        n_paths=args.n_paths, n_steps=n_steps_ext, dt=DT_SIM,
        fut_t=fut_t, fut_mat=fut_mat_1, fut_del=fut_del_1,
        fut_g_bar_1=g_fut_1, fut_g_bar_2=g_fut_2,
        price_scale_1=data["price_scale_1"],
        price_scale_2=data["price_scale_2"],
        N_pricing=N_poly,
        rng=rng_ext,
        add_obs_noise=(not args.no_obs_noise),
    )
    print(f"  DE extension shape: {sim_ext_eur_1.shape}")
    print(f"  FR extension shape: {sim_ext_eur_2.shape}")

    # ---- (C) Figures, per market ----
    hist_dt = _trading_to_dt(data["trading"][:, 0])
    fut_dt  = _trading_to_dt(fut_t)

    for (mkt, y_obs, sim_in, sim_ext) in [
        ("DE", data["y1"], sim_in_hist_eur_1, sim_ext_eur_1),
        ("FR", data["y2"], sim_in_hist_eur_2, sim_ext_eur_2),
    ]:
        inhist_path = os.path.join(
            FIG_DIR,
            f"sim_twomarket_{PERIOD_TAG}_{args.model}_inhist_{mkt}.png")
        plot_in_history_overlay(
            hist_dt, y_obs, sim_in,
            contract_labels=CONTRACT_LABELS,
            market_label=mkt, save_path=inhist_path,
            path_idx=0,
        )

        ext_path = os.path.join(
            FIG_DIR,
            f"sim_twomarket_{PERIOD_TAG}_{args.model}_extension_{mkt}_{n_steps_ext}d.png")
        plot_extension(
            hist_dt, y_obs, sim_ext, fut_dt,
            contract_labels=CONTRACT_LABELS,
            market_label=mkt, save_path=ext_path,
            n_paths_show=args.n_paths_show,
        )

        inc_path = os.path.join(
            THESIS_DIR,
            f"sim_twomarket_{PERIOD_TAG}_{args.model}_increments_{mkt}.png")
        plot_increment_histogram(
            y_obs, sim_in,
            contract_labels=CONTRACT_LABELS,
            market_label=mkt, save_path=inc_path,
        )

    # One-step-ahead in-sample fit per market (h(x_prior[t]) vs observed),
    # written as one PNG per contract so each maturity is readable.
    # This is the genuine state-space-model fit — its squared residuals
    # integrate to the in-sample EKF RMSE — distinct from the free
    # `plot_in_history_overlay` above which plots one simulated path from
    # the stationary law.
    for (mkt, y_obs_mkt, fit_eur_mkt) in [
        ("DE", data["y1"], fit_eur_1),
        ("FR", data["y2"], fit_eur_2),
    ]:
        for c, cname in enumerate(CONTRACT_LABELS):
            xprior_fit_path = os.path.join(
                FIG_DIR,
                f"sim_twomarket_{PERIOD_TAG}_{args.model}"
                f"_xprior_fit_{mkt}_{cname}.png")
            plot_in_history_xprior_fit_one_contract(
                hist_dt,
                y_obs_mkt[:, c], fit_eur_mkt[:, c],
                contract_label=cname,
                market_label=mkt,
                save_path=xprior_fit_path,
            )

    # Latent-state trace: x_post (filtered, with one-sigma band) and
    # x_prior (one-step-ahead) per state component. Mirrors the
    # plot_latent_states figure used by the OU / Jacobi simulators.
    latent_path = os.path.join(
        FIG_DIR,
        f"sim_twomarket_{PERIOD_TAG}_{args.model}_latent.png")
    plot_latent_states_twomarket(
        hist_dt, state_filt, state_cov_d,
        state_prior, prior_cov_d,
        params=params, save_path=latent_path,
        show_post_band=False,
    )

    # French observed price with single-variant fitted seasonality
    # overlaid — same style as the OU `seasonality_N2` thesis figure.
    fr_seas_path = os.path.join(
        THESIS_DIR,
        f"sim_twomarket_{PERIOD_TAG}_{args.model}_seasonality_FR.png")
    plot_market_with_seasonality(
        hist_dt=hist_dt,
        t_years=data["trading"][:, 0],
        y_obs=data["y2"],
        mat=data["mat_2"],
        dlt=data["del_2"],
        seas_beta=data["seas_beta_2"],
        annual_h=data["annual_h_2"],
        price_scale=data["price_scale_2"],
        contract_labels=CONTRACT_LABELS,
        market_label="FR",
        save_path=fr_seas_path,
    )

    # Deseasonalized & normalized French data
    # (y_resid_2 = y_FR / price_scale_FR - g_bar_FR), one line per contract.
    fr_deseason_path = os.path.join(
        THESIS_DIR,
        f"seasonality_FR_deseasonalized_{PERIOD_TAG}.png")
    _y2 = data["y_resid_2"]
    _nc = _y2.shape[1]
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(_nc):
        ax.plot(hist_dt, _y2[:, c], lw=1.0, alpha=0.85,
                label=str(CONTRACT_LABELS[c]))
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax.set_title("Deseasonalized & normalized French (FR) data")
    ax.set_xlabel("date")
    ax.set_ylabel("normalized deseasonalized price")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=_nc)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fr_deseason_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fr_deseason_path}")

    # Raw DE+FR forward prices for the shortest and longest maturities
    # over the in-sample window — used as a context figure in the thesis.
    raw_short_long_path = os.path.join(
        THESIS_DIR,
        f"sim_twomarket_{PERIOD_TAG}_{args.model}_raw_prices_1mah_4mah.png")
    plot_raw_prices_de_fr_short_long(
        hist_dt=hist_dt,
        y1=data["y1"], y2=data["y2"],
        contract_labels=CONTRACT_LABELS,
        save_path=raw_short_long_path,
    )

    # Multi-degree pooled-Δprice histogram, one figure per market — same
    # layout convention as the Jacobi thesis figure (2x2 grid when N_POLY_GRID
    # has 3-4 entries). Re-simulates each degree using its own saved params;
    # cells with no params file are silently skipped by the plotter.
    print("\nBuilding pooled Δprice histograms by polynomial degree ...")
    INC_HIST_DEGREES = tuple(drv.N_POLY_GRID)
    sim_by_degree    = {}
    for N_deg in INC_HIST_DEGREES:
        if N_deg == N_poly:
            sim_by_degree[N_deg] = (sim_in_hist_eur_1, sim_in_hist_eur_2)
            continue
        print(f"  [increments-hist] simulating two-market m={m} N={N_deg} ...")
        sim_by_degree[N_deg] = _simulate_in_history_for_degree_twomarket(
            m, N_deg, n_paths=args.n_paths, data=data, seed=args.seed,
        )
    for market_idx, (mkt, y_obs) in enumerate([
        ("DE", data["y1"]),
        ("FR", data["y2"]),
    ]):
        by_deg_path = os.path.join(
            THESIS_DIR,
            f"sim_twomarket_{PERIOD_TAG}_{args.model}"
            f"_increments_hist_all_{mkt}.png")
        plot_increment_histogram_by_degree(
            y_obs, sim_by_degree, market_idx=market_idx,
            market_label=mkt, save_path=by_deg_path,
        )

    # ---- (D) Console summary ----
    print("\n--- Variability check (per market, per contract, EUR/MWh) ---")
    for mkt, y_obs, sim_in, sim_ext in [
        ("DE", data["y1"], sim_in_hist_eur_1, sim_ext_eur_1),
        ("FR", data["y2"], sim_in_hist_eur_2, sim_ext_eur_2),
    ]:
        print(f"  {mkt}:")
        for c, cname in enumerate(CONTRACT_LABELS):
            hist_std = float(np.std(y_obs[:, c]))
            in_std   = float(np.std(sim_in[:, :, c]))
            ext_std  = float(np.std(sim_ext[:, :, c]))
            print(f"    {cname:6s}  hist_std={hist_std:8.3f}   "
                  f"sim_in_hist_std={in_std:8.3f}   "
                  f"sim_ext_std={ext_std:8.3f}")

    # ---- (E) Cross-market correlation check ----

    print("\n--- Cross-market correlation check (DE vs FR) ---")
    print(f"  Model parameter-implied E^P[R_t] "
          f"= theta_R + lam_R = "
          f"{params.theta_R + params.lam_R:+.4f}   "
          f"<-- instantaneous Corr(dZ, dY) under P at stationarity")

    obs_inc_1 = np.diff(data["y1"], axis=0)
    obs_inc_2 = np.diff(data["y2"], axis=0)
    sim_inc_1 = np.diff(sim_in_hist_eur_1, axis=1)
    sim_inc_2 = np.diff(sim_in_hist_eur_2, axis=1)
    n_paths   = sim_in_hist_eur_1.shape[0]

    def _per_path_corr(a_paths, b_paths):
        """Per-path Pearson correlation, returned as (mean, std) over paths.
        a_paths, b_paths shape (n_paths, T) — 1D series per path."""
        rs = []
        for p in range(a_paths.shape[0]):
            a = a_paths[p]; b = b_paths[p]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            rs.append(float(np.corrcoef(a, b)[0, 1]))
        if not rs:
            return float("nan"), float("nan")
        return float(np.mean(rs)), float(np.std(rs))

    print(f"  Level correlation, per contract:")
    for c, cname in enumerate(CONTRACT_LABELS):
        r_obs = float(np.corrcoef(data["y1"][:, c], data["y2"][:, c])[0, 1])
        r_sim_m, r_sim_s = _per_path_corr(
            sim_in_hist_eur_1[:, :, c], sim_in_hist_eur_2[:, :, c])
        print(f"    {cname:6s}  obs={r_obs:+.4f}   "
              f"sim_mean={r_sim_m:+.4f}   sim_std={r_sim_s:.4f}")

    print(f"  Δprice correlation, per contract (more economically meaningful):")
    for c, cname in enumerate(CONTRACT_LABELS):
        r_obs = float(np.corrcoef(obs_inc_1[:, c], obs_inc_2[:, c])[0, 1])
        r_sim_m, r_sim_s = _per_path_corr(
            sim_inc_1[:, :, c], sim_inc_2[:, :, c])
        print(f"    {cname:6s}  obs={r_obs:+.4f}   "
              f"sim_mean={r_sim_m:+.4f}   sim_std={r_sim_s:.4f}")

    # Pooled across contracts (concatenate per path).
    r_obs_pool   = float(np.corrcoef(obs_inc_1.ravel(),
                                        obs_inc_2.ravel())[0, 1])
    sim_pool_1   = sim_inc_1.reshape(n_paths, -1)
    sim_pool_2   = sim_inc_2.reshape(n_paths, -1)
    r_sim_m_p, r_sim_s_p = _per_path_corr(sim_pool_1, sim_pool_2)
    print(f"  Δprice correlation, pooled across "
          f"{len(CONTRACT_LABELS)} contracts:")
    print(f"    obs={r_obs_pool:+.4f}   "
          f"sim_mean={r_sim_m_p:+.4f}   sim_std={r_sim_s_p:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
