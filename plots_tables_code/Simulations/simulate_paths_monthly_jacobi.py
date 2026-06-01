"""Forward-simulate price paths under the calibrated Jacobi PMPP model. Sister to simulate_paths_monthly.py."""
from __future__ import annotations

import os
import sys
import argparse
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import expm

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

import GetData as gd
import kalman_filter_jacobi as jac
import BIC_monthly_jacobi    as drv

# Reuse OU simulator's plot helpers + roll-cycle detection.
from simulate_paths_monthly import (
    detect_cycle_len,
    build_per_col_rolling_schedule,
    plot_in_history_one_contract,
    plot_extension_one_contract,
    plot_latent_states,
    plot_in_sample_one_contract,
    trading_days_to_dt,
)


# Keep X strictly inside (0, 1) so sqrt(X(1-X)) stays real.
JACOBI_EPS = 1e-4


# Kernels and the single-market engine live in shared modules. Re-imported
# here so LLR_monthly_jacobi keeps importing these names from this script.
from sim_kernel_jacobi import (
    compute_observations_jacobi,
    simulate_state_paths_jacobi,
    filter_to_end_jacobi,
    ADAPTER as _JACOBI_ADAPTER,
)
import sim_engine


def simulate_in_history_jacobi(*args, **kwargs):
    """Jacobi in-history simulation (delegates to the shared engine + Jacobi kernels)."""
    return sim_engine.simulate_in_history(_JACOBI_ADAPTER, *args, **kwargs)


def simulate_extension_jacobi(*args, **kwargs):
    """Jacobi forward extension (delegates to the shared engine + Jacobi kernels)."""
    return sim_engine.simulate_extension(_JACOBI_ADAPTER, *args, **kwargs)


# Match calibration cadence — drv.DT_EKF is 7/365 (weekly) or 1/252 (daily).
DT_SIM = drv.DT_EKF


try:
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid", palette="deep")
    PALETTE = sns.color_palette("deep")
except ImportError:
    PALETTE = ["#4C72B0", "#DD8452", "#55A467", "#C44E52",
               "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["figure.dpi"]   = 100
plt.rcParams["savefig.dpi"]  = 300
plt.rcParams["font.family"]  = "sans-serif"


OUT_DIR = drv.OUT_DIR
# Period tag (e.g. 'weekly', 'monthly', 'weekly_monthly') from the BIC
# driver's STAGE_B_INCLUDE. Used to name figure folders and PNG files so
# weekly and monthly runs do not clobber each other.
PERIOD_TAG     = drv.PERIOD_TAG
FIG_DIR        = os.path.join(OUT_DIR,
                                f"figures_simulation_jacobi_{PERIOD_TAG}")
THESIS_FIG_DIR = os.path.join(OUT_DIR,
                                f"figures_thesis_{PERIOD_TAG}")
os.makedirs(FIG_DIR,        exist_ok=True)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)

MARKET_LABEL    = "DE"
CONTRACT_LABELS = list(drv.SUBSET_LABELS)


def _spec(m, N_poly):
    return dict(m=m, N_poly=N_poly,
                label=f"Jacobi m={m} N={N_poly} "
                      f"(monthly: {'/'.join(drv.SUBSET_LABELS)})")


MODEL_SPECS = {f"m{m}n{N_poly}": _spec(m, N_poly)
               for m in drv.M_GRID for N_poly in drv.N_POLY_GRID}
# Jacobi excludes N=1 (degenerate), so the smallest combination is whatever m1nN is.
_smallest_m  = min(drv.M_GRID)
_smallest_n  = min(drv.N_POLY_GRID)
MODEL_SPECS["benchmark"] = _spec(_smallest_m, _smallest_n)


def _mode_tag(per_factor_c=None):
    if per_factor_c is None:
        per_factor_c = drv.PER_FACTOR_C
    return "perfac" if per_factor_c else "global"


def params_filename(m, N_poly, per_factor_c=None):
    """Current-format filename for the (lam_ratio, σ) layout (Section 3.5)."""
    tag = _mode_tag(per_factor_c)
    return f"params_{PERIOD_TAG}_jacobi_m{m}_N{N_poly}_{tag}_lamratio.npy"


def load_params_vec(m, N_poly, per_factor_c=None):
    """Load the current `*_lamratio.npy` vector for (m, N_poly).

    Returns (path_used, vector), or (None, None) if no file is found.
    """
    if per_factor_c is None:
        per_factor_c = drv.PER_FACTOR_C
    new_name = params_filename(m, N_poly, per_factor_c)
    new_path = os.path.join(OUT_DIR, new_name)
    if os.path.exists(new_path):
        return new_path, np.load(new_path)
    return None, None


def resolve_best_key():
    """Lowest-BIC successful row in the mode-tagged BIC CSV """
    mode_tag = _mode_tag()
    candidates = [
        os.path.join(OUT_DIR,
                     f"bic_ekf_jacobi_{PERIOD_TAG}_{mode_tag}.csv"),
        os.path.join(OUT_DIR, f"bic_ekf_jacobi_monthly_{mode_tag}.csv"),
        os.path.join(OUT_DIR, "bic_ekf_jacobi_monthly.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        fallback = f"m{_smallest_m}n{_smallest_n}"
        print(f"  (no BIC CSV found in {[os.path.basename(c) for c in candidates]} — "
              f"falling back to {fallback})")
        return fallback
    df = pd.read_csv(path)
    if "success" in df.columns:
        ok = df[df["success"].astype(bool)]
        if not ok.empty:
            df = ok
    row = df.sort_values("BIC").iloc[0]
    key = f"m{int(row['m'])}n{int(row['N_poly'])}"
    print(f"  resolved --model best  →  {key}  (BIC={row['BIC']:.2f})")
    return key


def load_subset_data():
    y, mat, dlt, tra = drv.load_stage_b_data()
    y_a, mat_a, del_a, tra_a = drv.load_stage_a_data()
    return y, mat, dlt, tra, y_a, mat_a, del_a, tra_a


def compute_in_sample_prediction_jacobi(params, state_filt_hist,
                                         maturity_hist, delivery_hist,
                                         t_years_hist, seas_beta,
                                         annual_h,
                                         price_scale, N_pricing):
    """Filtered Jacobi state → polynomial map + seasonality."""
    n_days, n_c = maturity_hist.shape
    state_paths = np.asarray(state_filt_hist)[None, :, :]   # (1, n_days, m)

    y_norm_pred = compute_observations_jacobi(
        params, state_paths,
        T_step=maturity_hist, delta_step=delivery_hist,
        N_pricing=N_pricing,
    )
    y_norm_pred = y_norm_pred[0]

    _, S_hist, _ = jac.build_seasonality_matrix(
        np.asarray(t_years_hist), maturity_hist, delivery_hist,
        np.zeros((n_days, n_c)),
        annual_h=annual_h,
    )
    g_bar = (S_hist @ seas_beta).reshape(n_days, n_c)
    return price_scale * (y_norm_pred + g_bar)


# Local palette refs so Jacobi figures match OU without importing.
_THESIS_HIST_COLOR        = PALETTE[0]
_THESIS_SINGLE_PATH_COLOR = "#D62728"
_THESIS_REF_COLORS = ["#4C72B0", "#DD8452", "#55A467", "#C44E52",
                      "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def _plot_increments_per_maturity_thesis(hist_dt, obs_prices, sim_prices,
                                           contract_labels, save_path):
    """Per-maturity Δprice overlay (observed vs one sim path)."""
    obs_inc = np.diff(np.asarray(obs_prices), axis=0)
    sim_inc = np.diff(np.asarray(sim_prices), axis=0)
    dt_inc  = np.asarray(hist_dt)[1:]
    n_c     = obs_inc.shape[1]

    ncols = 2 if n_c > 1 else 1
    nrows = int(np.ceil(n_c / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(11, 2.7 * nrows),
                              sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for c in range(n_c):
        ax = axes[c]
        ax.plot(dt_inc, obs_inc[:, c],
                color=_THESIS_HIST_COLOR, lw=0.9, alpha=0.85,
                label="observed Δprice")
        ax.plot(dt_inc, sim_inc[:, c],
                color=_THESIS_SINGLE_PATH_COLOR, lw=0.9, alpha=0.75,
                label="simulated Δprice")
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylabel(f"{contract_labels[c]}\nΔEUR/MWh")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
            ax.xaxis.get_major_locator()))
    for c in range(n_c, len(axes)):
        axes[c].axis("off")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _ks_2samp(obs, sim):
    """Two-sample Kolmogorov–Smirnov via empirical CDFs on the union grid."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    obs = obs[np.isfinite(obs)]
    sim = sim[np.isfinite(sim)]
    if len(obs) == 0 or len(sim) == 0:
        return float("nan")
    grid  = np.unique(np.concatenate([obs, sim]))
    F_obs = np.searchsorted(np.sort(obs), grid, side="right") / len(obs)
    F_sim = np.searchsorted(np.sort(sim), grid, side="right") / len(sim)
    return float(np.max(np.abs(F_sim - F_obs)))


# Palette matched to LLR_monthly_jacobi's histogram figures.
_HIST_OBS_COLOR = "#1F77B4"
_HIST_SIM_COLOR = "#D62728"
_HIST_ALPHA     = 0.45


def _plot_increments_hist_per_maturity_thesis(obs_prices, sim_prices_all_paths,
                                                contract_labels, save_path,
                                                n_bins=60):
    """Per-maturity Δprice histograms (obs vs pooled sim), KS per panel."""
    obs_inc = np.diff(np.asarray(obs_prices), axis=0)
    sim_inc = np.diff(np.asarray(sim_prices_all_paths), axis=1)
    n_c     = obs_inc.shape[1]
    n_paths = sim_inc.shape[0]

    # Sanity-print of the variance match.
    print(f"  Δprice variance (per maturity, EUR/MWh)^2:")
    for c in range(n_c):
        v_obs = float(np.var(obs_inc[:, c], ddof=1))
        v_sim = float(np.var(sim_inc[:, :, c].ravel(), ddof=1))
        ratio = v_sim / v_obs if v_obs > 0 else float("nan")
        print(f"    {contract_labels[c]:6s}  obs={v_obs:9.4f}   "
              f"sim={v_sim:9.4f}   sim/obs={ratio:.3f}")

    ncols = 2 if n_c > 1 else 1
    nrows = int(np.ceil(n_c / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(11, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for c in range(n_c):
        ax = axes[c]
        obs_c = obs_inc[:, c]
        sim_c = sim_inc[:, :, c].ravel()
        # Always cover the full observed range; clip simulated at 0.5/99.5.
        sim_lo, sim_hi = np.percentile(sim_c, [0.5, 99.5])
        lo = min(float(np.min(obs_c)), float(sim_lo))
        hi = max(float(np.max(obs_c)), float(sim_hi))
        bins = np.linspace(lo, hi, n_bins + 1)
        ax.hist(obs_c, bins=bins, density=True, alpha=_HIST_ALPHA,
                color=_HIST_OBS_COLOR, label="observed",
                edgecolor="white", linewidth=0.3)
        ax.hist(sim_c, bins=bins, density=True, alpha=_HIST_ALPHA,
                color=_HIST_SIM_COLOR,
                label=f"simulated (n={n_paths})",
                edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel(f"{contract_labels[c]} Δprice (EUR/MWh)")
        ax.set_ylabel("density")
        ax.set_xlim(lo, hi)
        ax.grid(True, alpha=0.3)
        ks = _ks_2samp(obs_c, sim_c)
        ax.text(0.98, 0.95, f"KS = {ks:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#999999", alpha=0.85))
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    for c in range(n_c, len(axes)):
        axes[c].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")



def _simulate_in_history_for_degree_jacobi(m, N_poly, *,
                                              n_paths, dt,
                                              maturity, delivery, trading,
                                              seas_beta, annual_h, price_scale,
                                              seed):
    """In-history sim from theta_P with stationary covariance. None if params file missing."""
    pf_path, v = load_params_vec(m, N_poly)
    if v is None:
        print(f"  (skipping m={m},N={N_poly}: no params file)")
        return None
    p = jac.unpack_Jacobi(v, m, N_poly=N_poly,
                          per_factor_c=drv.PER_FACTOR_C)
    theta_P = np.asarray(p.theta + p.lam / p.kappa, dtype=float)
    a_jac = 2.0 * p.kappa * theta_P          / p.sigma ** 2
    b_jac = 2.0 * p.kappa * (1.0 - theta_P)  / p.sigma ** 2
    x_start = theta_P
    P_start = np.diag(theta_P * (1.0 - theta_P) / (a_jac + b_jac + 1.0))
    rng = np.random.default_rng(seed)
    sim_eur, _ = simulate_in_history_jacobi(
        p, n_paths=n_paths, dt=dt,
        maturity_hist=maturity, delivery_hist=delivery,
        t_years_hist=trading[:, 0],
        seas_beta=seas_beta, annual_h=annual_h,
        price_scale=price_scale, N_pricing=N_poly,
        x_start=x_start, P_start=P_start,
        rng=rng,
    )
    return sim_eur


def _plot_increments_hist_all_by_degree_thesis(obs_prices,
                                                  sim_by_degree,
                                                  contract_labels,
                                                  save_path,
                                                  n_bins=60):
    """One panel per polynomial degree of pooled Δprice histograms. Mirrors the OU figure."""
    obs_inc = np.diff(np.asarray(obs_prices), axis=0)
    obs_pool = obs_inc.ravel()

    items = [(N, s) for N, s in sim_by_degree.items() if s is not None]
    if not items:
        print(f"  (no simulations available — skipping {save_path})")
        return
    n_panels = len(items)


    sim_pools = []
    for _, sim_prices in items:
        sim_inc = np.diff(np.asarray(sim_prices), axis=1)
        sim_pools.append(sim_inc.ravel())
    sim_all = np.concatenate(sim_pools) if sim_pools else np.array([0.0])
    sim_lo, sim_hi = np.percentile(sim_all, [0.5, 99.5])
    lo = min(float(np.min(obs_pool)), float(sim_lo))
    hi = max(float(np.max(obs_pool)), float(sim_hi))
    bins = np.linspace(lo, hi, n_bins + 1)

    var_obs = float(np.var(obs_pool, ddof=1))
    print(f"  Δprice variance (pooled across maturities, EUR/MWh)^2:")
    print(f"    observed         = {var_obs:9.4f}")
    for (N, _), sim_pool in zip(items, sim_pools):
        v_sim = float(np.var(sim_pool, ddof=1))
        ratio = v_sim / var_obs if var_obs > 0 else float("nan")
        print(f"    simulated N={N}    = {v_sim:9.4f}   sim/obs={ratio:.3f}")

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
    for ax, (N, sim_prices), sim_pool in zip(axes, items, sim_pools):
        n_paths = np.asarray(sim_prices).shape[0]
        n_c     = obs_inc.shape[1]
        ax.hist(obs_pool, bins=bins, density=True, alpha=_HIST_ALPHA,
                color=_HIST_OBS_COLOR,
                label=f"observed (pooled, {n_c} mats)",
                edgecolor="white", linewidth=0.3)
        ax.hist(sim_pool, bins=bins, density=True, alpha=_HIST_ALPHA,
                color=_HIST_SIM_COLOR,
                label=f"simulated ({n_paths} paths × {n_c} mats)",
                edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(f"deg N={N}")
        ax.set_xlabel("Δ price (EUR/MWh)")
        ax.set_ylabel("density")
        ax.set_xlim(lo, hi)
        ax.grid(True, alpha=0.3)
        ks = _ks_2samp(obs_pool, sim_pool)
        ax.text(0.98, 0.95, f"KS = {ks:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#999999", alpha=0.85))
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    # Hide any unused cells in the grid (e.g. 3 panels in a 2x2 grid).
    for ax_unused in axes[len(items):]:
        ax_unused.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _plot_inhist_overlay_degrees_thesis(hist_dt, obs_prices,
                                          sim_prices_by_degree, degrees,
                                          contract_label, contract_idx,
                                          save_path, path_idx=0):
    """Overlay observed price for one contract with one simulated path per
    polynomial degree. Used for the thesis head-to-head between even (N=2)
    and odd (N=3) Jacobi maps; `degrees` controls which fits are drawn.

    sim_prices_by_degree: dict {N -> (n_paths, n_t, n_c) array | None}.
    Missing entries (no params file for that N) are skipped silently.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist_dt, obs_prices[:, contract_idx],
            color="#1F77B4", lw=1.2, alpha=0.95,
            label=f"{contract_label} observed")
    palette = ["#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]
    drawn_any = False
    for j, N in enumerate(degrees):
        sim = sim_prices_by_degree.get(N)
        if sim is None:
            print(f"  (overlay: no simulation for N={N}, skipping)")
            continue
        p_idx = max(0, min(int(path_idx), sim.shape[0] - 1))
        ax.plot(hist_dt, sim[p_idx, :, contract_idx],
                color=palette[j % len(palette)], lw=1.0, alpha=0.8,
                linestyle="--",
                label=f"{contract_label} simulated (N={N}, path {p_idx})")
        drawn_any = True
    if not drawn_any:
        print(f"  (overlay: no degree had a valid simulation — "
              f"figure not written)")
        plt.close(fig)
        return
    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(
        f"In-history overlay: {contract_label}, degrees "
        + ", ".join(f"N={N}" for N in degrees)
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=9,
              ncol=max(2, len(degrees) + 1))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _plot_increments_all_in_one_thesis(hist_dt, obs_prices, sim_prices,
                                         contract_labels, save_path):
    """Δprice for all maturities on one axes. Solid = observed, dashed = simulated."""
    obs_inc = np.diff(np.asarray(obs_prices), axis=0)
    sim_inc = np.diff(np.asarray(sim_prices), axis=0)
    dt_inc  = np.asarray(hist_dt)[1:]
    n_c     = obs_inc.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    for c in range(n_c):
        color = _THESIS_REF_COLORS[c % len(_THESIS_REF_COLORS)]
        ax.plot(dt_inc, obs_inc[:, c],
                color=color, lw=0.9, alpha=0.85,
                label=f"{contract_labels[c]} observed")
        ax.plot(dt_inc, sim_inc[:, c],
                color=color, lw=0.9, alpha=0.75, linestyle="--",
                label=f"{contract_labels[c]} simulated")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel("Δ price (EUR/MWh)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="upper left", frameon=False, fontsize=8,
              ncol=max(2, n_c))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")


# Combined in-history thesis figure: observed (solid) + one simulated (dashed) per contract.
def plot_in_history_combined(hist_dt, y_matrix, sim_paths,
                              contract_labels, label, save_path,
                              path_idx=0, corridor_eur=None):
    """sim_paths: (P, n_t, n_c); path_idx picks which sim path to draw.


    """
    n_c = y_matrix.shape[1]
    fig, ax = plt.subplots(figsize=(11, 5))
    for c, cname in enumerate(contract_labels):
        col = PALETTE[c % len(PALETTE)]
        ax.plot(hist_dt, y_matrix[:, c],
                color=col, lw=1.1, alpha=0.95,
                label=f"{cname} obs")
        ax.plot(hist_dt, sim_paths[path_idx, :, c],
                color=col, lw=1.1, alpha=0.65, ls="--",
                label=f"{cname} sim")
        if corridor_eur is not None:
            lower = corridor_eur[:, c, 0]
            upper = corridor_eur[:, c, 1]
            ax.plot(hist_dt, lower, color=col, lw=0.8, ls=":",
                    alpha=0.7)
            ax.plot(hist_dt, upper, color=col, lw=0.8, ls=":",
                    alpha=0.7)
    # Add a single legend entry for the corridor (avoid clutter).
    if corridor_eur is not None:
        from matplotlib.lines import Line2D
        bound_line = Line2D([0], [0], color="black", lw=0.8, ls=":",
                              alpha=0.7, label="model bounds")
        handles, labels_lbl = ax.get_legend_handles_labels()
        handles.append(bound_line); labels_lbl.append("model bounds")
        ax.legend(handles, labels_lbl, ncol=n_c + 1, fontsize=8,
                   loc="best", frameon=False)
    else:
        ax.legend(ncol=n_c, fontsize=8, loc="best", frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("EUR / MWh")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best",
                        help=(f"Which calibrated cell to simulate from. "
                              f"Options: {sorted(MODEL_SPECS.keys())} + 'best'."))
    parser.add_argument("--n-paths", type=int, default=500)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--n-paths-show", type=int, default=5)
    parser.add_argument("--show-bands", action="store_true")
    parser.add_argument("--inhist-path-idx", type=int, default=0,
                        help="Which simulated path index (0..n_paths-1) to "
                             "render on the combined thesis inhist figure.")
    parser.add_argument("--no-obs-noise", action="store_true")
    parser.add_argument("--show-history-years", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model == "best":
        args.model = resolve_best_key()
    if args.model not in MODEL_SPECS:
        raise SystemExit(
            f"--model {args.model!r} not available. "
            f"Choose from {sorted(MODEL_SPECS)} or 'best'.")

    spec = MODEL_SPECS[args.model]
    m, N_poly = spec["m"], spec["N_poly"]
    label = spec["label"]

    print(f"=== Simulating model '{args.model}' ({label}) ===")
    params_path, v = load_params_vec(m, N_poly)
    if v is None:
        new_name = params_filename(m, N_poly)
        raise SystemExit(
            f"Missing params file. Looked for {new_name} in {OUT_DIR}.\n"
            f"Run BIC_monthly_jacobi.py first to fit cell "
            f"(m={m}, N={N_poly}).")
    print(f"Loading params from {params_path} "
          f"(per_factor_c={drv.PER_FACTOR_C})")
    params = jac.unpack_Jacobi(v, m, N_poly=N_poly,
                                per_factor_c=drv.PER_FACTOR_C)
    theta_P = params.theta + params.lam / params.kappa
    print(f"  m={m}, N_poly={N_poly}  (Ware Appendix B sum mode)")
    print(f"  kappa     = {params.kappa}")
    print(f"  theta     = {params.theta}")
    print(f"  lam       = {params.lam}")
    print(f"  theta_P   = {theta_P}    <-- P-mean (sim drifts here, in [0,1])")
    print(f"  sigma     = {params.sigma}")
    print(f"  half-life (yr) = {np.log(2.0) / params.kappa}")

    # ----- Feller-type boundary conditions for the Jacobi state -----

    feller_a_P = 2.0 * np.asarray(params.kappa) * np.asarray(theta_P) \
                  / np.asarray(params.sigma) ** 2
    feller_b_P = 2.0 * np.asarray(params.kappa) * (1.0 - np.asarray(theta_P)) \
                  / np.asarray(params.sigma) ** 2
    feller_a_Q = 2.0 * np.asarray(params.kappa) * np.asarray(params.theta) \
                  / np.asarray(params.sigma) ** 2
    feller_b_Q = 2.0 * np.asarray(params.kappa) * (1.0 - np.asarray(params.theta)) \
                  / np.asarray(params.sigma) ** 2
    print(f"  Feller (P): 2 kappa theta_P / sigma^2     = {feller_a_P}   "
          f"<-- need >= 1 to stay off X=0")
    print(f"  Feller (P): 2 kappa (1-theta_P) / sigma^2 = {feller_b_P}   "
          f"<-- need >= 1 to stay off X=1")
    print(f"  Feller (Q): 2 kappa theta / sigma^2       = {feller_a_Q}   "
          f"<-- pricing operator e^(G tau) p_T assumes this >= 1")
    print(f"  Feller (Q): 2 kappa (1-theta) / sigma^2   = {feller_b_Q}")
    _viol_P = (feller_a_P < 1.0) | (feller_b_P < 1.0)
    _viol_Q = (feller_a_Q < 1.0) | (feller_b_Q < 1.0)
    if np.any(_viol_P) or np.any(_viol_Q):
        bad_factors_P = np.where(_viol_P)[0].tolist()
        bad_factors_Q = np.where(_viol_Q)[0].tolist()
        print(f"  ** WARNING: Feller violated.  P-factors: {bad_factors_P}, "
              f"Q-factors: {bad_factors_Q}.")
        print(f"     The continuous Jacobi process touches a boundary, so the "
              f"JACOBI_EPS={JACOBI_EPS:g} clip in simulate_state_paths_jacobi "
              f"is papering over a calibration outside the admissible region, "
              f"not just an Euler-overshoot fix. Any off-centre / skewed "
              f"increment distribution in the sim may be driven by clip bias, "
              f"not by the model.")
    else:
        print(f"  (Feller satisfied under both P and Q for all factors "
              f"→ JACOBI_EPS clip is a pure Euler safety net.)")
    print(f"  p_delta = {params.p_delta:+.5f}    "
          f"<-- additive offset on Φ_total")
    print(f"  per_factor_c = {params.per_factor_c}    "
          f"<-- Option 1 (per-factor) vs Option 2 (global)")
    print(f"  c_tilde   = {None if params.c_tilde is None else params.c_tilde.tolist()}")
    print(f"  c         = {params.c.tolist()}    "
          f"<-- per-factor amplitude on Φ_i(x): Φ_i: [0,1] → [0, c_i]")
    print(f"  corridor for Σ_i c_i Φ_i(X_i): "
          f"[0, {params.c.sum():.4f}]; "
          f"after p_delta shift: [{params.p_delta:+.4f}, "
          f"{params.p_delta + params.c.sum():+.4f}]")
    if params.k > 0:
        print(f"  alpha_tilde =\n{params.alpha_tilde}")
        print(f"  beta_tilde  =\n{params.beta_tilde}")
        print(f"  alpha (mapped) =\n{params.alpha}")
        print(f"  beta  (mapped) =\n{params.beta}")
    else:
        print(f"  (N_poly=1 → identity map Φ_i(x) = x)")
    print(f"  p_e (noise scalar) = {params.p_e:.4e}")

    print(f"\nLoading Stage A panel {list(drv.STAGE_A_LABELS)} "
          f"and Stage B panel {list(CONTRACT_LABELS)} ...")
    (y_matrix, maturity, delivery, trading,
     y_stagea,  mat_stagea, del_stagea, tra_stagea) = load_subset_data()
    n_days, n_c = y_matrix.shape
    print(f"  Stage A: {y_stagea.shape[0]} days × {y_stagea.shape[1]} contracts "
          f"({tuple(drv.STAGE_A_LABELS)})")
    print(f"  Stage B: {n_days} days × {n_c} contracts "
          f"({tuple(CONTRACT_LABELS)})")

    if drv.START_DATE is not None:
        y_stagea, mat_stagea, del_stagea, tra_stagea, idx_a = \
            drv.slice_panel_after_date(drv.START_DATE, y_stagea, mat_stagea,
                                        del_stagea, tra_stagea)
        y_matrix, maturity, delivery, trading, idx_b = \
            drv.slice_panel_after_date(drv.START_DATE, y_matrix, maturity,
                                        delivery, trading)
        n_days = y_matrix.shape[0]
        print(f"  Restricting to dates >= {drv.START_DATE}: "
              f"Stage A {idx_a} dropped, "
              f"Stage B {idx_b} dropped, "
              f"new Stage B size = {n_days}")

    # Match BIC_monthly_jacobi.main() — seasonality must be fit on the same
    # pre-thinned data the saved params were calibrated against.
    price_scale    = float(y_stagea.mean())
    y_stagea_norm  = y_stagea / price_scale
    y_norm         = y_matrix / price_scale
    print(f"  shared price_scale (Stage A mean) = {price_scale:.4f} EUR/MWh "
          f"(on {y_stagea.shape[0]} pre-thin rows)")

    print("Running Stage A seasonality grid (on full post-cutoff data) ...")
    best = None
    for ah in drv.ANNUAL_GRID:
        info = jac.seasonality_bic(tra_stagea[:, 0],
                                    mat_stagea, del_stagea,
                                    y_stagea_norm, ah)
        if best is None or info["BIC"] < best["BIC"]:
            best = info
    seas_beta = best["beta"]
    annual_h  = int(best["annual_h"])
    print(f"  best (a={annual_h})  BIC={best['BIC']:.1f}")

    _, S_hist, _ = jac.build_seasonality_matrix(
        trading[:, 0], maturity, delivery, y_norm,
        annual_h=annual_h,
    )
    g_bar_hist = (S_hist @ seas_beta).reshape(n_days, n_c)
    y_resid    = y_norm - g_bar_hist
    print(f"  Stage B residual (pre-thin): mean={y_resid.mean():+.5f}  "
          f"std={y_resid.std():.5f}")

    # ---- Model-implied price bounds in raw EUR/MWh ----
    # Phi(X) lives in [p_delta, p_delta + sum_i c_i] in normalised units.
    # In raw EUR/MWh per contract: price_scale * (g(t,tau_c) + corridor).
    # The width is constant in (t, tau); only the location slides with the
    # seasonal curve g(t, tau_c).
    c_sum_norm = float(np.asarray(params.c).sum())
    width_eur  = price_scale * c_sum_norm
    print(f"\n  --- Price bounds enforced by the polynomial map ---")
    print(f"  Normalised-residual corridor: "
          f"[{params.p_delta:+.4f}, {params.p_delta + c_sum_norm:+.4f}]  "
          f"(width = {c_sum_norm:.4f})")
    print(f"  Raw-EUR/MWh corridor width (constant):  "
          f"price_scale · Σ c_i = {price_scale:.2f} · {c_sum_norm:.4f} = "
          f"{width_eur:.1f} EUR/MWh")
    for c in range(n_c):
        g_min = float(g_bar_hist[:, c].min())
        g_max = float(g_bar_hist[:, c].max())
        lower_eur = price_scale * (g_min + params.p_delta)
        upper_eur = price_scale * (g_max + params.p_delta + c_sum_norm)
        print(f"  {CONTRACT_LABELS[c]:5s}  EUR/MWh envelope over window: "
              f"[{lower_eur:7.1f}, {upper_eur:7.1f}]   "
              f"(g range: [{g_min:.4f}, {g_max:.4f}])")
    print()

    # Data already at calibration cadence (weekly when USE_WEEKLY_SAMPLING).
    print(f"  Calibration cadence: "
          f"{'weekly ISO-Mon' if drv.USE_WEEKLY_SAMPLING else 'daily'}; "
          f"DT_SIM={DT_SIM:.6f} years; Stage B: {n_days} days × {n_c} contracts")

    print("Filtering through historical data ...")
    x0 = theta_P.reshape(-1, 1)
    a_jac = 2.0 * params.kappa * theta_P          / params.sigma ** 2
    b_jac = 2.0 * params.kappa * (1.0 - theta_P)  / params.sigma ** 2
    P0    = np.diag(theta_P * (1.0 - theta_P) / (a_jac + b_jac + 1.0))
    (x_final, P_final,
     state_filt, state_cov_d,
     state_prior, prior_cov_d,
     y_pred_norm_prior) = filter_to_end_jacobi(
        params, x0, P0, y_resid, maturity, delivery,
        DT_SIM, N_pricing=N_poly,
    )
    print(f"  x_final = {x_final}")
    print(f"  diag(P_final) = {np.diag(P_final)}")

    # One-step-ahead in-sample fit: predict each y_t from x_prior[t], which
    # is what the EKF innovation RMSE integrates. The seasonality g_bar is
    # added back inside the prediction (the EKF works on the deseasonalised
    # residual y_resid).
    print("\nComputing in-sample reconstruction from x_prior (one-step-ahead) ...")
    n_days_local, n_c_local = maturity.shape
    _, S_hist, _ = jac.build_seasonality_matrix(
        np.asarray(trading[:, 0]), maturity, delivery,
        np.zeros((n_days_local, n_c_local)),
        annual_h=annual_h,
    )
    g_bar_full = (S_hist @ seas_beta).reshape(n_days_local, n_c_local)
    in_sample_pred_eur = price_scale * (y_pred_norm_prior + g_bar_full)
    in_sample_rmse = np.sqrt(np.mean(
        (in_sample_pred_eur - y_matrix) ** 2, axis=0))
    in_sample_bias = (in_sample_pred_eur - y_matrix).mean(axis=0)
    print("  Per-contract one-step-ahead in-sample fit (EUR/MWh):")
    for c, cname in enumerate(CONTRACT_LABELS):
        print(f"    {cname:5s}  RMSE={in_sample_rmse[c]:7.3f}   "
              f"bias={in_sample_bias[c]:+7.3f}")

    # (A) In-history — start from the long-run mean (theta_P) rather than X[0].
    x_start_inhist = theta_P.copy()
    P_start_inhist = np.diag(theta_P * (1.0 - theta_P) / (a_jac + b_jac + 1.0))
    print(f"\nSimulating in-history paths from long-run mean theta_P = "
          f"{x_start_inhist}")
    rng_in_hist = np.random.default_rng(args.seed)
    sim_in_hist_eur, _ = simulate_in_history_jacobi(
        params, n_paths=args.n_paths, dt=DT_SIM,
        maturity_hist=maturity, delivery_hist=delivery,
        t_years_hist=trading[:, 0],
        seas_beta=seas_beta, annual_h=annual_h,
        price_scale=price_scale, N_pricing=N_poly,
        x_start=x_start_inhist, P_start=P_start_inhist,
        rng=rng_in_hist,
    )
    print(f"  in-history sim shape: {sim_in_hist_eur.shape}")

    # (B) Forward extension.
    n_steps = int(round(args.years / DT_SIM))
    cycle_lens = [detect_cycle_len(maturity[:, c]) for c in range(n_c)]
    print(f"\nDetected per-column roll cycles (simulator steps): "
          f"{dict(zip(CONTRACT_LABELS, cycle_lens))}")
    fut_mat, fut_del = build_per_col_rolling_schedule(
        maturity, delivery, n_steps, cycle_lens)

    t_last = float(trading[-1, 0])
    fut_t  = t_last + DT_SIM * np.arange(1, n_steps + 1)

    print(f"Simulating forward extension: {n_steps} steps, "
          f"{args.n_paths} paths ...")
    rng_ext = np.random.default_rng(args.seed + 1)
    sim_ext_eur, sim_ext_state = simulate_extension_jacobi(
        params, x_final, P_final,
        n_paths=args.n_paths, n_steps=n_steps, dt=DT_SIM,
        fut_t=fut_t, fut_mat=fut_mat, fut_del=fut_del,
        seas_beta=seas_beta, annual_h=annual_h,
        price_scale=price_scale, N_pricing=N_poly,
        add_obs_noise=(not args.no_obs_noise),
        last_hist_mat=maturity[-1:], last_hist_del=delivery[-1:],
        rng=rng_ext,
    )

    hist_dt_full = trading_days_to_dt(trading[:, 0])
    fut_dt       = trading_days_to_dt(fut_t)
    n_keep = max(1, int(round(args.show_history_years / DT_SIM)))
    n_keep = min(n_keep, len(hist_dt_full))
    hist_dt_show = hist_dt_full[-n_keep:]

    obs_suffix   = "_noobsnoise" if args.no_obs_noise else ""
    suffix = obs_suffix

    print(f"\nWriting figures to {FIG_DIR}")
    for c in range(n_c):
        cname = CONTRACT_LABELS[c]

        insample_path = os.path.join(
            FIG_DIR,
            f"sim_jacobi_{PERIOD_TAG}_{args.model}_insample_{cname}.png",
        )
        plot_in_sample_one_contract(
            hist_dt_full, y_matrix[:, c], in_sample_pred_eur[:, c],
            contract_name=cname, label=label, save_path=insample_path,
        )

        ext_path = os.path.join(
            FIG_DIR,
            f"sim_jacobi_{PERIOD_TAG}_{args.model}_extension_{cname}_{n_steps}d{suffix}.png",
        )
        plot_extension_one_contract(
            hist_dt_show, y_matrix[-n_keep:, c],
            fut_dt, sim_ext_eur[:, :, c],
            contract_name=cname, label=label, save_path=ext_path,
            show_bands=args.show_bands,
        )

    # One representative sim path → one-to-one Δprice comparison.
    inc_path_idx = int(getattr(args, "inhist_path_idx", 0))
    inc_path_idx = max(0, min(inc_path_idx, sim_in_hist_eur.shape[0] - 1))
    inc_per_mat_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_jacobi_{PERIOD_TAG}_{args.model}_increments_per_maturity.png",
    )
    _plot_increments_per_maturity_thesis(
        hist_dt_full, y_matrix, sim_in_hist_eur[inc_path_idx],
        contract_labels=CONTRACT_LABELS,
        save_path=inc_per_mat_path,
    )
    inc_all_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_jacobi_{PERIOD_TAG}_{args.model}_increments_all.png",
    )
    _plot_increments_all_in_one_thesis(
        hist_dt_full, y_matrix, sim_in_hist_eur[inc_path_idx],
        contract_labels=CONTRACT_LABELS,
        save_path=inc_all_path,
    )

    # Histogram companions: pool all sim paths.
    inc_hist_per_mat_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_jacobi_{PERIOD_TAG}_{args.model}_increments_hist_per_maturity.png",
    )
    _plot_increments_hist_per_maturity_thesis(
        y_matrix, sim_in_hist_eur,
        contract_labels=CONTRACT_LABELS,
        save_path=inc_hist_per_mat_path,
    )
    # Multi-degree pooled-Δprice histogram.
    INC_HIST_DEGREES = (1, 2, 3, 4, 5)
    sim_by_degree = {}
    for N_deg in INC_HIST_DEGREES:
        if N_deg == N_poly:
            sim_by_degree[N_deg] = sim_in_hist_eur
            continue
        print(f"  [increments-hist] simulating Jacobi m={m} N={N_deg} ...")
        sim_by_degree[N_deg] = _simulate_in_history_for_degree_jacobi(
            m, N_deg,
            n_paths=args.n_paths, dt=DT_SIM,
            maturity=maturity, delivery=delivery, trading=trading,
            seas_beta=seas_beta, annual_h=annual_h,
            price_scale=price_scale,
            seed=args.seed,
        )

    # Two 2×2 panels
    canonical_groups = [
        ([1, 2, 3, 4], "deg1_4"),
        ([2, 3, 4, 5], "deg2_5"),
    ]
    full_groups = [(g, suf) for (g, suf) in canonical_groups
                    if all(sim_by_degree.get(d) is not None for d in g)]
    if full_groups:
        for group, suf in full_groups:
            sim_subset = {d: sim_by_degree[d] for d in group}
            save_path = os.path.join(
                THESIS_FIG_DIR,
                f"sim_jacobi_{PERIOD_TAG}_{args.model}_"
                f"increments_hist_all_{suf}.png",
            )
            _plot_increments_hist_all_by_degree_thesis(
                y_matrix, sim_subset,
                contract_labels=CONTRACT_LABELS,
                save_path=save_path,
            )
    else:

        inc_hist_all_path = os.path.join(
            THESIS_FIG_DIR,
            f"sim_jacobi_{PERIOD_TAG}_{args.model}_increments_hist_all.png",
        )
        _plot_increments_hist_all_by_degree_thesis(
            y_matrix, sim_by_degree,
            contract_labels=CONTRACT_LABELS,
            save_path=inc_hist_all_path,
        )

    # Thesis overlay: observed price for OVERLAY_TARGET (default '1WAH') with
    # one simulated path each from the N=2 and N=3 fits — the head-to-head
    # comparison from the even/odd model-selection tables.
    OVERLAY_TARGET  = "1WAH"
    OVERLAY_DEGREES = (2, 3)
    if OVERLAY_TARGET in CONTRACT_LABELS:
        overlay_contract_idx = CONTRACT_LABELS.index(OVERLAY_TARGET)
        overlay_path = os.path.join(
            THESIS_FIG_DIR,
            f"sim_jacobi_{PERIOD_TAG}_{args.model}_overlay_"
            f"{OVERLAY_TARGET}_"
            + "_".join(f"N{N}" for N in OVERLAY_DEGREES)
            + ".png",
        )
        _plot_inhist_overlay_degrees_thesis(
            hist_dt_full, y_matrix, sim_by_degree, OVERLAY_DEGREES,
            contract_label=OVERLAY_TARGET,
            contract_idx=overlay_contract_idx,
            save_path=overlay_path,
            path_idx=int(getattr(args, "inhist_path_idx", 0)),
        )
    else:
        print(f"  (overlay skipped: {OVERLAY_TARGET!r} not in "
              f"contracts {CONTRACT_LABELS})")

    # ---- Per-(day, contract) price-bound corridor in raw EUR/MWh -----
    # P_hat(t, tau) ∈ price_scale · [g(t, tau) + p_delta,
    #                                g(t, tau) + p_delta + Σ_i c_i].
    # Compute it day-by-day so the bound line tracks the seasonal curve,
    # then pass it to the combined inhist plot for visual overlay.
    c_sum_norm = float(np.asarray(params.c).sum())
    corridor_lo_norm = g_bar_hist + params.p_delta                  # (n_t, n_c)
    corridor_hi_norm = g_bar_hist + params.p_delta + c_sum_norm
    corridor_eur = np.stack([
        price_scale * corridor_lo_norm,
        price_scale * corridor_hi_norm,
    ], axis=-1)                                                      # (n_t, n_c, 2)
    print(f"\n  Price-bound corridor (EUR/MWh, raw):  "
          f"width = price_scale · Σ c_i = {price_scale * c_sum_norm:.1f}")
    for c in range(n_c):
        lo_min = float(corridor_eur[:, c, 0].min())
        hi_max = float(corridor_eur[:, c, 1].max())
        print(f"    {CONTRACT_LABELS[c]:5s}  envelope over window: "
              f"[{lo_min:7.1f}, {hi_max:7.1f}] EUR/MWh")

    # Combined in-history thesis fig: all observed (solid) + matched sim (dashed).
    combined_inhist_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_jacobi_{PERIOD_TAG}_{args.model}_inhist_combined.png",
    )
    plot_in_history_combined(
        hist_dt_full, y_matrix, sim_in_hist_eur,
        contract_labels=CONTRACT_LABELS,
        label=label, save_path=combined_inhist_path,
        path_idx=int(getattr(args, "inhist_path_idx", 0)),
        corridor_eur=None,   # bounds drawn dotted made the figure unreadable
    )
    print(f"  combined inhist figure -> {combined_inhist_path}")

    # Latent-state diagnostic.
    state_filt_show  = state_filt[-n_keep:, :]
    state_cov_d_show = state_cov_d[-n_keep:, :]
    state_prior_show = state_prior[-n_keep:, :]
    prior_cov_d_show = prior_cov_d[-n_keep:, :]
    states_path = os.path.join(
        FIG_DIR,
        f"sim_jacobi_{PERIOD_TAG}_{args.model}_{n_steps}d_latent.png",
    )
    plot_latent_states(
        hist_dt_show, state_filt_show, state_cov_d_show,
        fut_dt, sim_ext_state, label, states_path,
        show_bands=args.show_bands,
        state_prior_hist=state_prior_show,
        prior_cov_d_hist=prior_cov_d_show,
    )

    print("\n--- Variability check (per contract, EUR/MWh) ---")
    for c in range(n_c):
        cname = CONTRACT_LABELS[c]
        hist_std = float(np.std(y_matrix[:, c]))
        in_hist_std = float(np.std(sim_in_hist_eur[:, :, c]))
        ext_std = float(np.std(sim_ext_eur[:, :, c]))
        print(f"  {cname:5s}  hist_std={hist_std:8.3f}   "
              f"sim_in_hist_std={in_hist_std:8.3f}   "
              f"sim_ext_std={ext_std:8.3f}")

    # (Price bounds were printed once near the top of main, after Stage A
    # seasonality and price_scale were available — see the "Price bounds
    # enforced by the polynomial map" block.)

    print("\nDone.")


if __name__ == "__main__":
    main()
