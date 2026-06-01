"""Forward-simulate price paths for the monthly Stage-B panel from a calibrated OU model."""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import combinations_with_replacement
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
import Kalman_filter_LD as ld
import BIC_monthly_OU   as drv


# Compute kernels and the single-market engine live in shared modules. These
# names are re-imported here so LLR_monthly_OU and the Jacobi script keep
# importing them from simulate_paths_monthly.
from sim_common import (
    trading_days_to_dt,
    detect_cycle_len,
    build_per_col_rolling_schedule,
)
from sim_kernel_ou import (
    filter_to_end,
    simulate_state_paths,
    compute_observations,
    ADAPTER as _OU_ADAPTER,
)
import sim_engine


def compute_in_sample_prediction(params, state_filt_hist,
                                  maturity_hist, delivery_hist,
                                  t_years_hist, seas_beta,
                                  annual_h,
                                  price_scale, N_pricing):
    """Filtered X → polynomial map + seasonality → predicted prices in EUR/MWh."""
    n_days, n_c = maturity_hist.shape
    state_paths = np.asarray(state_filt_hist)[None, :, :]   # (1, n_days, m)

    y_norm_pred = compute_observations(params, state_paths,
                                        maturity_hist, delivery_hist,
                                        N_pricing=N_pricing)
    y_norm_pred = y_norm_pred[0]

    _, S_hist, _ = ld.build_seasonality_matrix(
        np.asarray(t_years_hist), maturity_hist, delivery_hist,
        np.zeros((n_days, n_c)),
        annual_h=annual_h,
    )
    g_bar = (S_hist @ seas_beta).reshape(n_days, n_c)

    return price_scale * (y_norm_pred + g_bar)


def plot_prices_and_seasonality(price_refs, seasonality_eval_ref, fits,
                                 price_scale, save_path, title=None):
    """Single-panel overlay of contract prices + one seasonality curve per fit (EUR/MWh)."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    REF_COLORS = ["#4C72B0", "#DD8452", "#55A467", "#C44E52",
                  "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
    SEAS_LINESTYLES = ["--", ":", "-."]
    SEAS_COLOR = "black"

    for ri, ref in enumerate(price_refs):
        t_axis  = np.asarray(ref["t_axis"], dtype=float)
        dt_axis = trading_days_to_dt(t_axis)
        color   = REF_COLORS[ri % len(REF_COLORS)]
        ax.plot(dt_axis, np.asarray(ref["prices"]),
                color=color, lw=1.0, alpha=0.9,
                label=f"{ref['name']} price")

    eval_t   = np.asarray(seasonality_eval_ref["t_axis"], dtype=float)
    n_g      = len(eval_t)
    eval_mat = np.asarray(seasonality_eval_ref["mat"]).reshape(n_g, 1)
    eval_del = np.asarray(seasonality_eval_ref["del"]).reshape(n_g, 1)
    eval_dt  = trading_days_to_dt(eval_t)
    for vi, fit in enumerate(fits):
        ah     = int(fit["annual_h"])
        beta   = np.asarray(fit["beta"])
        _, S, _ = ld.build_seasonality_matrix(
            eval_t, eval_mat, eval_del, np.zeros((n_g, 1)),
            annual_h=ah,
        )
        g_eur = price_scale * (S @ beta)
        ls    = SEAS_LINESTYLES[vi % len(SEAS_LINESTYLES)]
        ax.plot(eval_dt, g_eur, color=SEAS_COLOR, lw=1.6, linestyle=ls,
                label=(f"g(t) on {seasonality_eval_ref['name']} | "
                       f"a={ah}"))

    ax.set_ylabel("EUR / MWh")
    if title is None:
        v_str    = " / ".join(f"a={int(f['annual_h'])}"
                               for f in fits)
        ref_str  = ", ".join(r["name"] for r in price_refs)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="upper left", frameon=False, fontsize=8,
              ncol=max(2, len(price_refs)))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# Honour BIC_monthly_OU.USE_WEEKLY_SAMPLING — sim dt must match calibration
# dt or the mean-reversion speed will be wrong. drv.DT_EKF is the matching
# constant: 7/365 in weekly mode, 1/252 in daily mode.
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

SINGLE_PATH_COLOR = "#D62728"
HIST_COLOR        = PALETTE[0]


OUT_DIR        = drv.OUT_DIR
PERIOD_TAG     = drv.PERIOD_TAG
FIG_DIR        = os.path.join(OUT_DIR,
                                f"figures_simulation_ou_{PERIOD_TAG}")
THESIS_FIG_DIR = os.path.join(OUT_DIR, f"figures_thesis_{PERIOD_TAG}")
os.makedirs(FIG_DIR,        exist_ok=True)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)

MARKET_LABEL    = "DE"
CONTRACT_LABELS = list(drv.SUBSET_LABELS)

def _spec(m, N_poly):
    return dict(m=m, N_poly=N_poly,
                label=f"OU m={m} N={N_poly} "
                      f"(monthly: {'/'.join(drv.SUBSET_LABELS)})")

MODEL_SPECS = {f"m{m}n{N_poly}": _spec(m, N_poly)
               for m in drv.M_GRID for N_poly in drv.N_POLY_GRID}
# Register m=1 explicitly so the M1N1/M1N3 thesis overlay works even if drv.M_GRID skips it.
MODEL_SPECS.setdefault("m1n1", _spec(1, 1))
MODEL_SPECS.setdefault("m1n3", _spec(1, 3))
MODEL_SPECS["benchmark"] = _spec(1, 1)
# `best` is resolved at runtime from bic_ekf_ou_monthly.csv.


def params_filename(m, N_poly, indep, lam):
    indep_tag = "_indep" if indep else ""
    lam_tag   = "_lam"   if lam   else ""
    return f"params_{PERIOD_TAG}_ou_m{m}_N{N_poly}{indep_tag}{lam_tag}.npy"


def resolve_best_key():
    """Lowest-BIC successful row in bic_ekf_ou_monthly.csv, else m3n3."""
    path = os.path.join(OUT_DIR, "bic_ekf_ou_monthly.csv")
    if not os.path.exists(path):
        print(f"  (bic_ekf_ou_monthly.csv not found — falling back to m3n3)")
        return "m3n3"
    df = pd.read_csv(path)
    if "success" in df.columns:
        ok = df[df["success"].astype(bool)]
        if not ok.empty:
            df = ok
    row = df.sort_values("BIC").iloc[0]
    key = f"m{int(row['m'])}n{int(row['N_poly'])}"
    print(f"  resolved --model best  →  {key}  (BIC={row['BIC']:.2f})")
    return key


# Same loaders the BIC driver uses, so the simulator sees the panels Stages A/B were calibrated on.
def load_subset_data():
    y, mat, dlt, tra = drv.load_stage_b_data()
    y_a, mat_a, del_a, tra_a = drv.load_stage_a_data()
    return y, mat, dlt, tra, y_a, mat_a, del_a, tra_a


def simulate_in_history(*args, **kwargs):
    """OU in-history simulation (delegates to the shared engine + OU kernels)."""
    return sim_engine.simulate_in_history(_OU_ADAPTER, *args, **kwargs)


def simulate_extension(*args, **kwargs):
    """OU forward extension (delegates to the shared engine + OU kernels)."""
    return sim_engine.simulate_extension(_OU_ADAPTER, *args, **kwargs)


# Thesis figures — no titles, captions go in LaTeX.

_THESIS_REF_COLORS = ["#4C72B0", "#DD8452", "#55A467", "#C44E52",
                       "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def _plot_seasonality_thesis(price_refs, seasonality_eval_ref, fit,
                              price_scale, save_path):
    """Single-variant version of plot_prices_and_seasonality with no on-figure title."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))

    for ri, ref in enumerate(price_refs):
        t_axis  = np.asarray(ref["t_axis"], dtype=float)
        dt_axis = trading_days_to_dt(t_axis)
        ax.plot(dt_axis, np.asarray(ref["prices"]),
                color=_THESIS_REF_COLORS[ri % len(_THESIS_REF_COLORS)],
                lw=1.0, alpha=0.9, label=ref["name"])

    eval_t   = np.asarray(seasonality_eval_ref["t_axis"], dtype=float)
    n_g      = len(eval_t)
    eval_mat = np.asarray(seasonality_eval_ref["mat"]).reshape(n_g, 1)
    eval_del = np.asarray(seasonality_eval_ref["del"]).reshape(n_g, 1)
    eval_dt  = trading_days_to_dt(eval_t)

    ah   = int(fit["annual_h"])
    beta = np.asarray(fit["beta"])
    _, S, _ = ld.build_seasonality_matrix(
        eval_t, eval_mat, eval_del, np.zeros((n_g, 1)),
        annual_h=ah,
    )
    g_eur = price_scale * (S @ beta)
    ax.plot(eval_dt, g_eur, color="black", lw=1.6, linestyle="--",
            label=f"g(t) | N={ah}")

    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              ncol=max(2, len(price_refs)))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _plot_deseasonalized_normalized(t_axis, labels, y_resid, save_path):
    """Deseasonalized + normalized prices (y_norm - g_bar), one line per contract."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    dt_axis = trading_days_to_dt(np.asarray(t_axis, dtype=float))
    for ci, lbl in enumerate(labels):
        ax.plot(dt_axis, y_resid[:, ci],
                color=_THESIS_REF_COLORS[ci % len(_THESIS_REF_COLORS)],
                lw=1.0, alpha=0.9, label=lbl)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.set_ylabel("normalized residual")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              ncol=max(2, len(labels)))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _filter_and_in_sample_pred(m, N_poly, indep, lam, *,
                                y_resid, maturity, delivery, trading,
                                seas_beta, annual_h, price_scale):
    """Load params for (m, N_poly), filter, push through poly + seasonality. None if file missing."""
    pf_name = params_filename(m, N_poly, indep, lam)
    pf_path = os.path.join(OUT_DIR, pf_name)
    if not os.path.exists(pf_path):
        print(f"  (skipping {pf_name}: file missing)")
        return None
    v = np.load(pf_path)
    p = ld.unpack_ld(v, m, N_poly=N_poly,
                      fit_d=drv.FIT_D,
                      independent_poly=indep)
    x0 = ld._mu_P(p).reshape(-1, 1)
    # Full stationary covariance with rho-driven cross terms (was diagonal).
    P0 = ld.stationary_cov(p)
    _, _, _, _, _, _, y_pred_norm_local = filter_to_end(
        p, x0, P0, y_resid, maturity, delivery,
        DT_SIM, N_pricing=N_poly,
    )
    # Convert h(x_prior) (normalised-residual space) back to raw EUR/MWh
    # by re-adding seasonality and applying the price scale. This is the
    # one-step-ahead in-sample prediction.
    n_days_local, n_c_local = maturity.shape
    _, S_hist, _ = ld.build_seasonality_matrix(
        np.asarray(trading[:, 0]), maturity, delivery,
        np.zeros((n_days_local, n_c_local)),
        annual_h=annual_h,
    )
    g_bar_full = (S_hist @ seas_beta).reshape(n_days_local, n_c_local)
    return price_scale * (y_pred_norm_local + g_bar_full)


def _simulate_in_history_for_degree(m, N_poly, indep, lam, *,
                                      n_paths, dt,
                                      maturity, delivery, trading,
                                      seas_beta, annual_h, price_scale,
                                      seed):
    """In-history sim from the long-run mean. None if the params file is missing."""
    pf_name = params_filename(m, N_poly, indep, lam)
    pf_path = os.path.join(OUT_DIR, pf_name)
    if not os.path.exists(pf_path):
        print(f"  (skipping {pf_name}: file missing)")
        return None
    v = np.load(pf_path)
    p = ld.unpack_ld(v, m, N_poly=N_poly,
                      fit_d=drv.FIT_D,
                      independent_poly=indep)
    x_start = np.asarray(ld._mu_P(p)).reshape(-1)
    # Full stationary covariance — includes the rho-driven cross terms the
    # old diagonal form dropped. PSD by construction; simulate_state_paths
    # already ridges with 1e-12*I before its Cholesky.
    P_start = ld.stationary_cov(p)
    rng = np.random.default_rng(seed)
    sim_eur, _ = simulate_in_history(
        p, n_paths=n_paths, dt=dt,
        maturity_hist=maturity, delivery_hist=delivery,
        t_years_hist=trading[:, 0],
        seas_beta=seas_beta, annual_h=annual_h,
        price_scale=price_scale, N_pricing=N_poly,
        x_start=x_start, P_start=P_start,
        rng=rng,
    )
    return sim_eur


def _plot_insample_factor_sweep_thesis(hist_dt, hist_prices, pred_by_m,
                                         contract_name, N_fixed, save_path):
    """In-sample x_prior fit overlay for varying m at one fixed polynomial
    degree N_fixed. `pred_by_m` is a dict {m: pred_array | None}; cells
    with None (no params file) are skipped silently. Per-fit RMSE in the
    legend, observed series drawn in HIST_COLOR underneath."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.0))
    ax.plot(hist_dt, hist_prices,
            color=HIST_COLOR, lw=1.1, alpha=0.95,
            label=f"{contract_name} observed", zorder=5)
    factor_colors = {1: "#D62728", 2: "#2CA02C", 3: "#9467BD"}
    z = 6
    for m_val in sorted(pred_by_m.keys()):
        pred = pred_by_m[m_val]
        if pred is None:
            continue
        rmse = float(np.sqrt(np.mean((pred - hist_prices) ** 2)))
        ax.plot(hist_dt, pred,
                color=factor_colors.get(m_val, "#888888"),
                lw=1.0, alpha=0.95,
                label=f"M={m_val}, N={N_fixed} in-sample fit "
                      f"(RMSE={rmse:.2f})",
                zorder=z)
        z += 1
    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _plot_insample_m1n1_m1n3_thesis(hist_dt, hist_prices,
                                      pred_m1n1, pred_m1n3,
                                      contract_name, save_path):
    """In-sample overlay for OU M=1,N=1 vs M=1,N=3 against the observed series. RMSE in the legend."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.0))
    ax.plot(hist_dt, hist_prices,
            color=HIST_COLOR, lw=1.1, alpha=0.95,
            label=f"{contract_name} observed", zorder=5)
    if pred_m1n1 is not None:
        rmse_n1 = float(np.sqrt(np.mean((pred_m1n1 - hist_prices) ** 2)))
        ax.plot(hist_dt, pred_m1n1,
                color="#D62728", lw=1.0, alpha=0.95,
                label=f"M=1, N=1 in-sample fit (RMSE={rmse_n1:.2f})",
                zorder=6)
    if pred_m1n3 is not None:
        rmse_n3 = float(np.sqrt(np.mean((pred_m1n3 - hist_prices) ** 2)))
        ax.plot(hist_dt, pred_m1n3,
                color="#2CA02C", lw=1.0, alpha=0.95,
                label=f"M=1, N=3 in-sample fit (RMSE={rmse_n3:.2f})",
                zorder=7)
    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _plot_increments_per_maturity_thesis(hist_dt, obs_prices, sim_prices,
                                           contract_labels, save_path):
    """Per-maturity Δprice overlay (observed vs one in-history sim path).

    hist_dt        : (n_t,) datetime array
    obs_prices     : (n_t, n_c) observed prices in EUR/MWh
    sim_prices     : (n_t, n_c) simulated prices for one path, EUR/MWh
    contract_labels: list[str] of length n_c
    """
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
                color=HIST_COLOR, lw=0.9, alpha=0.85,
                label="observed Δprice")
        ax.plot(dt_inc, sim_inc[:, c],
                color=SINGLE_PATH_COLOR, lw=0.9, alpha=0.75,
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


# Palette matched to LLR_monthly_jacobi's histogram_m* figures.
_HIST_OBS_COLOR = "#1F77B4"
_HIST_SIM_COLOR = "#D62728"
_HIST_ALPHA     = 0.45


def _plot_increments_hist_per_maturity_thesis(obs_prices, sim_prices_all_paths,
                                                contract_labels, save_path,
                                                n_bins=60):
    """Per-maturity Δprice histograms (obs vs pooled sim), KS in the top-right of each panel."""
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
        # Always cover the full observed range; clip simulated at 0.5/99.5
        # so a few extreme sim paths don't compress the bulk.
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


def _plot_increments_hist_all_by_degree_thesis(obs_prices,
                                                  sim_by_degree,
                                                  contract_labels,
                                                  save_path,
                                                  n_bins=60):
    """One panel per polynomial degree of pooled Δprice histograms; KS in each."""
    obs_inc = np.diff(np.asarray(obs_prices), axis=0)
    obs_pool = obs_inc.ravel()

    items = [(N, s) for N, s in sim_by_degree.items() if s is not None]
    if not items:
        print(f"  (no simulations available — skipping {save_path})")
        return
    n_panels = len(items)

    # Shared x-scale across panels. Always cover the full observed range so
    # the largest data-side increments are visible; clip the simulated tails
    # at their 0.5/99.5 percentile.
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

    fig, axes = plt.subplots(1, n_panels,
                              figsize=(4.6 * n_panels, 3.8),
                              sharey=False)
    if n_panels == 1:
        axes = [axes]
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


def plot_in_history_one_contract(hist_dt, hist_prices, sim_prices_eur,
                                  contract_name, label, save_path,
                                  n_paths_show=5, show_envelope=True):
    """sim_prices_eur: (n_paths, n_days); hist_prices: (n_days,) for a single contract."""
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.0))

    n_paths = sim_prices_eur.shape[0]
    show = min(int(n_paths_show), n_paths)
    for p in range(show):
        ax.plot(hist_dt, sim_prices_eur[p],
                color=SINGLE_PATH_COLOR, lw=0.6, alpha=0.35,
                zorder=2)

    if show_envelope and n_paths >= 20:
        p05 = np.percentile(sim_prices_eur,  5, axis=0)
        p95 = np.percentile(sim_prices_eur, 95, axis=0)
        ax.fill_between(hist_dt, p05, p95,
                         color=SINGLE_PATH_COLOR, alpha=0.10,
                         label="sim 5–95 %", zorder=1)

    ax.plot(hist_dt, sim_prices_eur[0],
            color=SINGLE_PATH_COLOR, lw=1.0, alpha=1.0,
            label="single sim path (X₀ from prior)",
            zorder=4)

    ax.plot(hist_dt, hist_prices,
            color=HIST_COLOR, lw=1.1, alpha=0.95,
            label="historical observed", zorder=5)

    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_in_sample_one_contract(hist_dt, hist_prices, pred_prices,
                                 contract_name, label, save_path):
    """In-sample reconstruction overlay. pred_prices should track hist_prices closely if the fit is consistent."""
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.0))
    ax.plot(hist_dt, hist_prices,
            color=HIST_COLOR, lw=1.1, alpha=0.95,
            label="historical observed", zorder=5)
    ax.plot(hist_dt, pred_prices,
            color=SINGLE_PATH_COLOR, lw=1.0, alpha=0.95,
            label="in-sample fit (filtered X → price)", zorder=6)
    err = pred_prices - hist_prices
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(err.mean())
    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}  (RMSE={rmse:.2f}, bias={bias:+.2f})")


def plot_extension_one_contract(hist_dt, hist_prices, fut_dt, sim_prices_eur,
                                  contract_name, label, save_path,
                                  show_bands=True):
    """sim_prices_eur: (n_paths, n_steps); hist_prices: (n_hist_show,)."""
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.0))

    ax.plot(hist_dt, hist_prices,
            color=HIST_COLOR, lw=1.0,
            label="historical observed", alpha=0.9)

    if show_bands and sim_prices_eur.shape[0] >= 20:
        mean_path = sim_prices_eur.mean(axis=0)
        p05 = np.percentile(sim_prices_eur,  5, axis=0)
        p25 = np.percentile(sim_prices_eur, 25, axis=0)
        p75 = np.percentile(sim_prices_eur, 75, axis=0)
        p95 = np.percentile(sim_prices_eur, 95, axis=0)
        ax.fill_between(fut_dt, p05, p95,
                        color=PALETTE[1], alpha=0.15, label="sim 5–95 %")
        ax.fill_between(fut_dt, p25, p75,
                        color=PALETTE[1], alpha=0.30, label="sim 25–75 %")
        ax.plot(fut_dt, mean_path, color=PALETTE[1], lw=1.4,
                label="sim mean")

    ax.plot(fut_dt, sim_prices_eur[0],
            color=SINGLE_PATH_COLOR, lw=1.0, alpha=1.0,
            label="single sim path", zorder=10)
    ax.axvline(hist_dt[-1], color="k", ls="--", lw=0.8, alpha=0.5)

    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_latent_states(hist_dt, state_filt_hist, state_cov_d_hist,
                        fut_dt, sim_state_paths, label, save_path,
                        show_bands=False,
                        state_prior_hist=None, prior_cov_d_hist=None):
    """Per-factor latent state. Solid coloured line = posterior (x_post)
    with one-sigma band from `state_cov_d_hist`; dashed grey line =
    optional prior x_prior (one-step-ahead) when `state_prior_hist` is
    provided. `sim_state_paths`: (n_paths, n_fut+1, m), t=0 dropped to
    align with `fut_dt`."""
    m = state_filt_hist.shape[1]
    if m == 1:
        names = ["X (OU)"]
    elif m == 2:
        names = ["X_slow", "X_fast"]
    elif m == 3:
        names = ["X_slow", "X_med", "X_fast"]
    else:
        names = [f"X_{i+1}" for i in range(m)]

    sim_paths_fut = sim_state_paths[:, 1:, :]

    fig, axes = plt.subplots(m, 1, figsize=(11, 1.8 * m + 1.0),
                              sharex=True)
    if m == 1:
        axes = [axes]

    if show_bands:
        mean_path = sim_paths_fut.mean(axis=0)
        p05 = np.percentile(sim_paths_fut,  5, axis=0)
        p25 = np.percentile(sim_paths_fut, 25, axis=0)
        p75 = np.percentile(sim_paths_fut, 75, axis=0)
        p95 = np.percentile(sim_paths_fut, 95, axis=0)

    for i, ax in enumerate(axes):
        sd_i = np.sqrt(np.maximum(state_cov_d_hist[:, i], 0.0))
        ax.fill_between(hist_dt,
                        state_filt_hist[:, i] - sd_i,
                        state_filt_hist[:, i] + sd_i,
                        color=PALETTE[0], alpha=0.15)
        ax.plot(hist_dt, state_filt_hist[:, i],
                color=PALETTE[0], lw=1.0, label="x_post (filtered)")
        if state_prior_hist is not None:
            ax.plot(hist_dt, state_prior_hist[:, i],
                    color="#444444", lw=0.9, linestyle="--",
                    label="x_prior (one-step-ahead)")

        if show_bands:
            ax.fill_between(fut_dt, p05[:, i], p95[:, i],
                            color=PALETTE[1], alpha=0.15, label="sim 5–95 %")
            ax.fill_between(fut_dt, p25[:, i], p75[:, i],
                            color=PALETTE[1], alpha=0.30, label="sim 25–75 %")
            ax.plot(fut_dt, mean_path[:, i],
                    color=PALETTE[1], lw=1.4, label="sim mean")

        ax.plot(fut_dt, sim_paths_fut[0, :, i],
                color=SINGLE_PATH_COLOR, lw=1.0, alpha=1.0,
                label="single sim path", zorder=10)

        ax.axvline(hist_dt[-1], color="k", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)
        ax.set_ylabel(names[i] if i < len(names) else f"X_{i}")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center",
               bbox_to_anchor=(0.5, 1.02 + 0.005 * m), ncol=5,
               frameon=False, fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        axes[-1].xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# Combined in-history thesis figure: observed (solid) + one simulated (dashed) per contract.
def plot_in_history_combined(hist_dt, y_matrix, sim_paths,
                              contract_labels, label, save_path,
                              path_idx=0):
    """sim_paths: (P, n_t, n_c); path_idx picks which sim path to draw."""
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
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("EUR / MWh")
    ax.legend(ncol=n_c, fontsize=8, loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best",
                        help=("Which calibrated cell to simulate from. "
                              f"Options: {sorted(MODEL_SPECS.keys())} "
                              "+ 'best'."))
    parser.add_argument("--n-paths", type=int, default=500,
                        help="Number of simulated paths (both views).")
    parser.add_argument("--years", type=float, default=3.0,
                        help="Years of trading days to simulate forward "
                             "in the extension view.")
    parser.add_argument("--n-paths-show", type=int, default=5,
                        help="Number of faint stochastic paths drawn on "
                             "the in-history overlay (always at least 1).")
    parser.add_argument("--show-bands", action="store_true",
                        help="Draw 5–95 / 25–75 envelope bands in the "
                             "forward-extension plot. Off by default.")
    parser.add_argument("--inhist-path-idx", type=int, default=0,
                        help="Which simulated path index (0..n_paths-1) to "
                             "render on the combined thesis inhist figure.")
    parser.add_argument("--no-obs-noise", action="store_true",
                        help="Skip observation noise in simulated prices "
                             "(produces smoother paths).")
    parser.add_argument("--show-history-years", type=float, default=99.0,
                        help="How many years of historical data to display "
                             "in the extension figure.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indep", action="store_true",
                        help="Load the INDEPENDENT (per-factor) polynomial-map "
                             "fit (params_..._indep.npy) and unpack with "
                             "independent_poly=True. Off by default → uses the "
                             "shared params file.")
    parser.add_argument("--lam", action="store_true",
                        help="Load the FIT_LAM=True fit "
                             "(params_..._lam.npy) and unpack with "
                             "fit_lam=True. Off by default → uses the "
                             "no-lam params file (lam fixed at zero).")
    args = parser.parse_args()

    if args.model == "best":
        args.model = resolve_best_key()
    if args.model not in MODEL_SPECS:
        raise SystemExit(
            f"--model {args.model!r} not available. "
            f"Choose from {sorted(MODEL_SPECS)} or 'best'.")

    spec = MODEL_SPECS[args.model]
    m, N_poly = spec["m"], spec["N_poly"]
    flag_tag = ""
    if args.indep: flag_tag += " [indep]"
    if args.lam:   flag_tag += " [lam]"
    label = spec["label"] + flag_tag
    pf_name = params_filename(m, N_poly, args.indep, args.lam)
    params_path = os.path.join(OUT_DIR, pf_name)
    if not os.path.exists(params_path):
        raise SystemExit(
            f"Missing params file: {params_path}\n"
            f"Run BIC_monthly_OU.py first to fit cell (m={m}, N={N_poly}) "
            f"with INDEPENDENT_POLY={args.indep} and FIT_LAM={args.lam}.")

    print(f"=== Simulating model '{args.model}' ({label}) ===")
    print(f"Loading params from {params_path}")
    v = np.load(params_path)
    params = ld.unpack_ld(v, m, N_poly=N_poly,
                           fit_d=drv.FIT_D,
                           independent_poly=args.indep)
    print(f"  p_e (noise scalar) = {params.p_e:.4e}")
    print(f"  m={m}, N_poly={N_poly}  "
          f"independent_poly={args.indep}  fit_lam={args.lam}")
    print(f"  theta (kappa)   = {params.theta}")
    print(f"  mu              = {params.mu}")
    print(f"  lam             = {params.lam}")
    print(f"  P-mean (mu + lam/theta) = {ld._mu_P(params)}")
    print(f"  c (sigma)       = {params.c}")
    print(f"  rho             =\n{params.rho}")
    if args.indep:
        print(f"  p_delta         = {params.p_delta}")
        print(f"  p_beta_arr      = {params.p_beta_arr}")
        if N_poly >= 5:
            print(f"  p_gamma_arr     = {params.p_gamma_arr}")
            print(f"  p_K_arr         = {params.p_K_arr}")
    else:
        print(f"  p_delta = {params.p_delta}  p_beta = {params.p_beta}")
        if N_poly >= 5:
            print(f"  p_gamma = {params.p_gamma}  p_K = {params.p_K}")

    print(f"\nLoading Stage A panel {list(drv.STAGE_A_LABELS)} "
          f"and Stage B panel {list(CONTRACT_LABELS)} ...")
    (y_matrix, maturity, delivery, trading,
     y_stagea,  mat_stagea, del_stagea, tra_stagea) = load_subset_data()
    n_days, n_c = y_matrix.shape
    print(f"  Stage A: {y_stagea.shape[0]} days × {y_stagea.shape[1]} contracts "
          f"({tuple(drv.STAGE_A_LABELS)})")
    print(f"  Stage B: {n_days} days × {n_c} contracts "
          f"({tuple(CONTRACT_LABELS)})")

    # Same cut as BIC_monthly_OU.main() so the sim sees the same window the EKF was fit on.
    if drv.START_DATE is not None:
        y_stagea, mat_stagea, del_stagea, tra_stagea, idx_a = \
            drv.slice_panel_after_date(drv.START_DATE, y_stagea, mat_stagea,
                                        del_stagea, tra_stagea)
        y_matrix, maturity, delivery, trading, idx_b = \
            drv.slice_panel_after_date(drv.START_DATE, y_matrix, maturity,
                                        delivery, trading)
        n_days = y_matrix.shape[0]
        print(f"  Restricting to dates >= {drv.START_DATE}: "
              f"Stage A {idx_a} rows dropped, "
              f"Stage B {idx_b} rows dropped, "
              f"new Stage B size = {n_days}")

    # Data is already at the calibration cadence (weekly when
    # drv.USE_WEEKLY_SAMPLING, daily otherwise) — see drv._load_panel.
    print(f"  Calibration cadence: "
          f"{'weekly ISO-Mon' if drv.USE_WEEKLY_SAMPLING else 'daily'}; "
          f"DT_SIM={DT_SIM:.6f} years")

    price_scale    = float(y_stagea.mean())
    y_stagea_norm  = y_stagea / price_scale
    y_norm         = y_matrix / price_scale
    print(f"  shared price_scale (Stage A mean) = {price_scale:.4f} EUR/MWh")

    print("Running Stage A seasonality grid on the 4-contract panel ...")
    best = None
    for ah in drv.ANNUAL_GRID:
        info = ld.seasonality_bic(tra_stagea[:, 0],
                                   mat_stagea, del_stagea,
                                   y_stagea_norm, ah)
        if best is None or info["BIC"] < best["BIC"]:
            best = info
    seas_beta = best["beta"]
    annual_h  = int(best["annual_h"])
    print(f"  best (a={annual_h})  BIC={best['BIC']:.1f}")

    # Two reference variants so the figure shows how harmonic count changes g(t).
    print("Fitting reference seasonality variants for the overlay plot ...")
    SEAS_VARIANTS = [1, 2]
    variant_fits = [{
        "annual_h": annual_h, "beta": seas_beta,
    }]
    for ah_v in SEAS_VARIANTS:
        info_v = ld.seasonality_bic(tra_stagea[:, 0],
                                     mat_stagea, del_stagea,
                                     y_stagea_norm, ah_v)
        variant_fits.append({
            "annual_h": ah_v, "beta": info_v["beta"],
        })
        print(f"  variant (a={ah_v}): k={info_v['k']:2d}  "
              f"logL={info_v['logL']:.1f}  BIC={info_v['BIC']:.1f}")

    # Mean maturity / delivery for the eval ref → smooth g(t) (rolling matures cause sawtooth).
    price_refs = [
        {"name":   drv.STAGE_A_LABELS[k],
         "t_axis": tra_stagea[:, k],
         "prices": y_stagea[:, k]}
        for k in range(y_stagea.shape[1])
    ]
    n_g = mat_stagea.shape[0]
    ones_g = np.ones(n_g)
    # Longest contract → g(t) curve sits at the level the eye reads off.
    eval_idx = len(drv.STAGE_A_LABELS) - 1
    eval_name = drv.STAGE_A_LABELS[eval_idx]
    eval_ref = {
        "name":   eval_name,
        "t_axis": tra_stagea[:, eval_idx],
        "mat":    ones_g * float(np.mean(mat_stagea[:, eval_idx])),
        "del":    ones_g * float(np.mean(del_stagea[:, eval_idx])),
    }
    seasonality_path = os.path.join(
        FIG_DIR, "seasonality_stage_a.png")
    plot_prices_and_seasonality(
        price_refs=price_refs,
        seasonality_eval_ref=eval_ref,
        fits=variant_fits,
        price_scale=price_scale,
        save_path=seasonality_path,
        title=(f"Stage A seasonality — prices "
               f"({', '.join(drv.STAGE_A_LABELS)}); "
               f"g(t) on {eval_name} | "
               f"best a={annual_h} + "
               f"variants {SEAS_VARIANTS}"),
    )

    # Thesis copy of the same panel, only the N=2 variant of g(t).
    n2_fit = next((f for f in variant_fits if int(f["annual_h"]) == 2), None)
    if n2_fit is not None:
        _plot_seasonality_thesis(
            price_refs=price_refs,
            seasonality_eval_ref=eval_ref,
            fit=n2_fit,
            price_scale=price_scale,
            save_path=os.path.join(THESIS_FIG_DIR, "seasonality_N2.png"),
        )

    # ---- Build historical residual ----
    _, S_hist, _ = ld.build_seasonality_matrix(
        trading[:, 0], maturity, delivery, y_norm,
        annual_h=annual_h,
    )
    g_bar_hist = (S_hist @ seas_beta).reshape(n_days, n_c)
    y_resid    = y_norm - g_bar_hist

    _plot_deseasonalized_normalized(
        t_axis=trading[:, 0],
        labels=list(CONTRACT_LABELS),
        y_resid=y_resid,
        save_path=os.path.join(THESIS_FIG_DIR,
                                "deseasonalized_normalized.png"),
    )

    # ---- Filter through historical data → x_final, P_final, state_filt ----
    print("Filtering through historical data ...")
    x0 = ld._mu_P(params).reshape(-1, 1)
    # Full stationary covariance with rho-driven cross terms (was diagonal).
    P0 = ld.stationary_cov(params)
    (x_final, P_final,
     state_filt, state_cov_d,
     state_prior, prior_cov_d,
     y_pred_norm_prior) = filter_to_end(
        params, x0, P0, y_resid, maturity, delivery,
        DT_SIM, N_pricing=N_poly,
    )
    print(f"  x_final = {x_final}")
    print(f"  diag(P_final) = {np.diag(P_final)}")

    # (0) One-step-ahead in-sample reconstruction from x_prior. This is what
    # the EKF innovation RMSE integrates: predict y_t from the state
    # estimate that uses observations only through t-1 (x_prior[t]),
    # then re-add seasonality and apply price_scale.
    print("\nComputing one-step-ahead in-sample reconstruction from x_prior ...")
    n_days_main, n_c_main = maturity.shape
    _, S_hist_main, _ = ld.build_seasonality_matrix(
        np.asarray(trading[:, 0]), maturity, delivery,
        np.zeros((n_days_main, n_c_main)),
        annual_h=annual_h,
    )
    g_bar_full_main = (S_hist_main @ seas_beta).reshape(n_days_main, n_c_main)
    in_sample_pred_eur = price_scale * (y_pred_norm_prior + g_bar_full_main)
    in_sample_rmse = np.sqrt(np.mean(
        (in_sample_pred_eur - y_matrix) ** 2, axis=0))
    in_sample_bias = (in_sample_pred_eur - y_matrix).mean(axis=0)
    print("  Per-contract one-step-ahead in-sample fit (EUR/MWh):")
    for c, cname in enumerate(CONTRACT_LABELS):
        print(f"    {cname:5s}  RMSE={in_sample_rmse[c]:7.3f}   "
              f"bias={in_sample_bias[c]:+7.3f}")

    # (A) In-history — start from the LONG-RUN P-MEAN with the OU
    # stationary covariance. The data-anchored X[0] start biases the early sim
    # toward the data; the unconditional mean is cleaner for the thesis figure.
    x_start_inhist = np.asarray(ld._mu_P(params)).reshape(-1)
    # Full stationary covariance (rho * c_i*c_j / (theta_i+theta_j)), not just
    # the diagonal — keeps the initial draw consistent with the long-run joint
    # distribution when factors are correlated.
    P_start_inhist = ld.stationary_cov(params)
    print(f"\nSimulating in-history paths from long-run P-mean = "
          f"{x_start_inhist}")
    print(f"  diag(P_start) = {np.diag(P_start_inhist)}")
    rng_in_hist = np.random.default_rng(args.seed)
    sim_in_hist_eur, _ = simulate_in_history(
        params, n_paths=args.n_paths, dt=DT_SIM,
        maturity_hist=maturity, delivery_hist=delivery,
        t_years_hist=trading[:, 0],
        seas_beta=seas_beta, annual_h=annual_h,
        price_scale=price_scale, N_pricing=N_poly,
        x_start=x_start_inhist, P_start=P_start_inhist,
        rng=rng_in_hist,
    )
    print(f"  in-history sim shape: {sim_in_hist_eur.shape}  "
          f"(paths × days × contracts)")

    # (B) Forward extension from (x_final, P_final) with per-column rolling maturity.
    n_steps = int(round(args.years / DT_SIM))
    cycle_lens = [detect_cycle_len(maturity[:, c]) for c in range(n_c)]
    print(f"\nDetected per-column roll cycles (trading days): "
          f"{dict(zip(CONTRACT_LABELS, cycle_lens))}")
    fut_mat, fut_del = build_per_col_rolling_schedule(
        maturity, delivery, n_steps, cycle_lens)

    t_last = float(trading[-1, 0])
    fut_t  = t_last + DT_SIM * np.arange(1, n_steps + 1)

    print(f"Simulating forward extension: {n_steps} steps, "
          f"{args.n_paths} paths ...")
    rng_ext = np.random.default_rng(args.seed + 1)
    sim_ext_eur, sim_ext_state = simulate_extension(
        params, x_final, P_final,
        n_paths=args.n_paths, n_steps=n_steps, dt=DT_SIM,
        fut_t=fut_t, fut_mat=fut_mat, fut_del=fut_del,
        seas_beta=seas_beta, annual_h=annual_h,
        price_scale=price_scale, N_pricing=N_poly,
        add_obs_noise=(not args.no_obs_noise),
        last_hist_mat=maturity[-1:], last_hist_del=delivery[-1:],
        rng=rng_ext,
    )
    print(f"  extension sim shape: {sim_ext_eur.shape}")

    hist_dt_full = trading_days_to_dt(trading[:, 0])
    fut_dt       = trading_days_to_dt(fut_t)
    # show_history_years → rows: each row is DT_SIM years.
    n_keep = max(1, int(round(args.show_history_years / DT_SIM)))
    n_keep = min(n_keep, len(hist_dt_full))
    hist_dt_show = hist_dt_full[-n_keep:]

    indep_tag = "_indep" if args.indep else ""
    lam_tag   = "_lam"   if args.lam   else ""
    flag_suffix = indep_tag + lam_tag
    suffix = ("_noobsnoise" if args.no_obs_noise else "") + flag_suffix

    print(f"\nWriting figures to {FIG_DIR}")
    for c in range(n_c):
        cname = CONTRACT_LABELS[c]
        insample_path = os.path.join(
            FIG_DIR,
            f"sim_ou_{PERIOD_TAG}_{args.model}_insample_{cname}{flag_suffix}.png",
        )
        plot_in_sample_one_contract(
            hist_dt_full, y_matrix[:, c], in_sample_pred_eur[:, c],
            contract_name=cname, label=label, save_path=insample_path,
        )
        ext_path = os.path.join(
            FIG_DIR,
            f"sim_ou_{PERIOD_TAG}_{args.model}_extension_{cname}_{n_steps}d{suffix}.png",
        )
        plot_extension_one_contract(
            hist_dt_show, y_matrix[-n_keep:, c],
            fut_dt, sim_ext_eur[:, :, c],
            contract_name=cname, label=label, save_path=ext_path,
            show_bands=args.show_bands,
        )

    # Thesis 1MAH in-sample overlay (M1N1 vs M1N3). Skipped if params files missing.
    if "1MAH" in CONTRACT_LABELS:
        one_mah_idx = CONTRACT_LABELS.index("1MAH")
        print("\nBuilding thesis 1MAH in-sample overlay (M1N1 vs M1N3) ...")
        pred_m1n1 = _filter_and_in_sample_pred(
            1, 1, args.indep, args.lam,
            y_resid=y_resid, maturity=maturity, delivery=delivery,
            trading=trading, seas_beta=seas_beta, annual_h=annual_h,
            price_scale=price_scale,
        )
        pred_m1n3 = _filter_and_in_sample_pred(
            1, 3, args.indep, args.lam,
            y_resid=y_resid, maturity=maturity, delivery=delivery,
            trading=trading, seas_beta=seas_beta, annual_h=annual_h,
            price_scale=price_scale,
        )
        thesis_1mah_path = os.path.join(
            THESIS_FIG_DIR,
            f"sim_ou_{PERIOD_TAG}_m1n1_m1n3_insample_1MAH{flag_suffix}.png",
        )
        _plot_insample_m1n1_m1n3_thesis(
            hist_dt_full, y_matrix[:, one_mah_idx],
            None if pred_m1n1 is None else pred_m1n1[:, one_mah_idx],
            None if pred_m1n3 is None else pred_m1n3[:, one_mah_idx],
            contract_name="1MAH",
            save_path=thesis_1mah_path,
        )

    # Thesis 1WAH overlays (analogous to the 1MAH block above):
    #   A) m-factor sweep at fixed N=3 (one curve per m in {1, 2, 3})
    #   B) benchmark m1n1 vs m1n3, ONLY when this run is simulating m1n3.
    # Both use the one-step-ahead `_filter_and_in_sample_pred` predictions
    # (h(x_prior)). Cells with no saved params file are silently skipped.
    if "1WAH" in CONTRACT_LABELS:
        one_wah_idx = CONTRACT_LABELS.index("1WAH")

        # ---- (A) m={1,2,3} at N=3 ----
        print("\nBuilding thesis 1WAH in-sample sweep (m={1,2,3} at N=3) ...")
        pred_by_m_n3 = {}
        for m_val in (1, 2, 3):
            pred_by_m_n3[m_val] = _filter_and_in_sample_pred(
                m_val, 3, args.indep, args.lam,
                y_resid=y_resid, maturity=maturity, delivery=delivery,
                trading=trading, seas_beta=seas_beta, annual_h=annual_h,
                price_scale=price_scale,
            )
        sweep_1wah_path = os.path.join(
            THESIS_FIG_DIR,
            f"sim_ou_{PERIOD_TAG}_m_sweep_N3_insample_1WAH{flag_suffix}.png",
        )
        _plot_insample_factor_sweep_thesis(
            hist_dt_full, y_matrix[:, one_wah_idx],
            pred_by_m={m: (None if v is None else v[:, one_wah_idx])
                       for m, v in pred_by_m_n3.items()},
            contract_name="1WAH",
            N_fixed=3,
            save_path=sweep_1wah_path,
        )

        # ---- (B) m1n1 vs m1n3, only when this run is the m1n3 cell ----
        if args.model == "m1n3":
            print("Building thesis 1WAH benchmark overlay (M1N1 vs M1N3) ...")
            pred_m1n1_1wah = _filter_and_in_sample_pred(
                1, 1, args.indep, args.lam,
                y_resid=y_resid, maturity=maturity, delivery=delivery,
                trading=trading, seas_beta=seas_beta, annual_h=annual_h,
                price_scale=price_scale,
            )
            # Re-use the m1n3 prediction from the sweep dict above instead
            # of refitting/refiltering for the same cell.
            pred_m1n3_1wah = pred_by_m_n3.get(1)
            thesis_1wah_path = os.path.join(
                THESIS_FIG_DIR,
                f"sim_ou_{PERIOD_TAG}_m1n1_m1n3_insample_1WAH{flag_suffix}.png",
            )
            _plot_insample_m1n1_m1n3_thesis(
                hist_dt_full, y_matrix[:, one_wah_idx],
                None if pred_m1n1_1wah is None else pred_m1n1_1wah[:, one_wah_idx],
                None if pred_m1n3_1wah is None else pred_m1n3_1wah[:, one_wah_idx],
                contract_name="1WAH",
                save_path=thesis_1wah_path,
            )

    # One representative sim path so the Δprice comparison is one-to-one, not ensemble-averaged.
    inc_path_idx = int(getattr(args, "inhist_path_idx", 0))
    inc_path_idx = max(0, min(inc_path_idx, sim_in_hist_eur.shape[0] - 1))
    inc_per_mat_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_ou_{PERIOD_TAG}_{args.model}_increments_per_maturity{flag_suffix}.png",
    )
    _plot_increments_per_maturity_thesis(
        hist_dt_full, y_matrix, sim_in_hist_eur[inc_path_idx],
        contract_labels=CONTRACT_LABELS,
        save_path=inc_per_mat_path,
    )
    inc_all_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_ou_{PERIOD_TAG}_{args.model}_increments_all{flag_suffix}.png",
    )
    _plot_increments_all_in_one_thesis(
        hist_dt_full, y_matrix, sim_in_hist_eur[inc_path_idx],
        contract_labels=CONTRACT_LABELS,
        save_path=inc_all_path,
    )

    # Histogram companions: pool all sim paths for the empirical density.
    inc_hist_per_mat_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_ou_{PERIOD_TAG}_{args.model}_increments_hist_per_maturity{flag_suffix}.png",
    )
    _plot_increments_hist_per_maturity_thesis(
        y_matrix, sim_in_hist_eur,
        contract_labels=CONTRACT_LABELS,
        save_path=inc_hist_per_mat_path,
    )
    # Multi-degree pooled-Δprice histogram; reuses args.model's sim for the matching degree.
    inc_hist_all_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_ou_{PERIOD_TAG}_{args.model}_increments_hist_all{flag_suffix}.png",
    )
    INC_HIST_DEGREES = (1, 3, 5)
    sim_by_degree = {}
    for N_deg in INC_HIST_DEGREES:
        if N_deg == N_poly:
            sim_by_degree[N_deg] = sim_in_hist_eur
            continue
        print(f"  [increments-hist] simulating m={m} N={N_deg} ...")
        sim_by_degree[N_deg] = _simulate_in_history_for_degree(
            m, N_deg, args.indep, args.lam,
            n_paths=args.n_paths, dt=DT_SIM,
            maturity=maturity, delivery=delivery, trading=trading,
            seas_beta=seas_beta, annual_h=annual_h,
            price_scale=price_scale,
            seed=args.seed,
        )
    _plot_increments_hist_all_by_degree_thesis(
        y_matrix, sim_by_degree,
        contract_labels=CONTRACT_LABELS,
        save_path=inc_hist_all_path,
    )

    # Combined in-history thesis fig: all observed (solid) + matched sim (dashed).
    combined_inhist_path = os.path.join(
        THESIS_FIG_DIR,
        f"sim_ou_{PERIOD_TAG}_{args.model}_inhist_combined{flag_suffix}.png",
    )
    plot_in_history_combined(
        hist_dt_full, y_matrix, sim_in_hist_eur,
        contract_labels=CONTRACT_LABELS,
        label=label, save_path=combined_inhist_path,
        path_idx=int(getattr(args, "inhist_path_idx", 0)),
    )

    # Latent-state diagnostic.
    state_filt_show  = state_filt[-n_keep:, :]
    state_cov_d_show = state_cov_d[-n_keep:, :]
    state_prior_show = state_prior[-n_keep:, :]
    prior_cov_d_show = prior_cov_d[-n_keep:, :]
    states_path = os.path.join(
        FIG_DIR,
        f"sim_ou_{PERIOD_TAG}_{args.model}_{n_steps}d_latent{flag_suffix}.png",
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

    print("\nDone.")


if __name__ == "__main__":
    main()
