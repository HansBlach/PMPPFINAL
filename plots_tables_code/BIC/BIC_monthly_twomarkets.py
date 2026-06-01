"""BIC grid for the two-market Kalman filter (DE leader / FR follower)."""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, OptimizeResult

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
import Kalman_filter_TwoMarket as tm


DATA_YEARS    = range(2023, 2026)
START_DATE    = "2023-05-01"
DATA_PATH_DE  = ("/Users/hansblachfalkenberg/Desktop/Unimat-4/speciale/DE/"
                 "PowerFutureHistory_Phelix-DE_{year}.xlsx")
DATA_PATH_FR  = ("/Users/hansblachfalkenberg/Desktop/Unimat-4/speciale/FR/"
                 "PowerFutureHistory_FR_{year}.xlsx")
OUT_DIR       = os.path.dirname(os.path.abspath(__file__))

# Per-market Excel sheet names; _load_panel reads from these dicts.
SHEETS_DE = {
    "monthly":   "DEBM",
    "quarterly": "DEBQ",
    "yearly":    "DEBY",
    "weekly":    "DEB1-5",
}
SHEETS_FR = {
    "monthly":   "F7BM",
    "quarterly": "F7BQ",
    "yearly":    "F7BY",
    "weekly":    "F7B1-5",
}

# Each entry applies to both markets; synced with the OU/Jacobi panels.
STAGE_B_INCLUDE = {
    "weekly": {
        "enabled": False,
        "1WAH": True, "2WAH": True,  "3WAH": True, "4WAH": True,
    },
    "monthly": {
        "enabled": True,
        "1MAH": True,  "2MAH": True, "3MAH": True, "4MAH": True,
    },
    "quarterly": {
        "enabled": False,
        "1QAH": True,  "2QAH": False, "3QAH": False, "4QAH": False,
    },
    "yearly": {
        "enabled": False,
        "1YAH": True,  "2YAH": True,  "3YAH": True,
    },
}

STAGE_A_INCLUDE = {
    "weekly": {
        "enabled": False,
        "1WAH": True, "2WAH": True, "3WAH": True, "4WAH": True,
    },
    "monthly": {
        "enabled": True,
        "1MAH": True, "2MAH": True, "3MAH": True, "4MAH": True,
    },
    "quarterly": {
        "enabled": False,
        "1QAH": True, "2QAH": True, "3QAH": True, "4QAH": True,
    },
    "yearly": {
        "enabled": False,
        "1YAH": True, "2YAH": True, "3YAH": True,
    },
}

ANNUAL_GRID   = (2,)

M_GRID        = (1,)
# Odd degrees only; the two-market polynomial form uses odd powers (N=1 is the
# degree-1 baseline for the LLR nested tests).
N_POLY_GRID   = (1,3,5)

# Weekly sampling keeps the first trading day of each ISO week; False is daily.
USE_WEEKLY_SAMPLING = True
DT          = 1 / 252.0
DT_WEEKLY   = 7 / 365.0
DT_EKF      = DT_WEEKLY if USE_WEEKLY_SAMPLING else DT
SEED        = 42

# Scalar p_e per market, Var[v_t]_ii = p_e^2 * (tau_i / tau_ref).
TAU_REF       = 1.0

DE_MAXITER_MAP = {
    (1, 1): 200, (2, 1): 25,
    (1, 3): 200, (2, 3): 25,
    (1, 5): 200, (2, 5): 20,
}
DE_POPSIZE      = 10
LBFGS_MAXITER   = 300


_PANEL_CLASS_DEFS = (
    ("weekly",       "WAH",  4),
    ("monthly",      "MAH",  4),
    ("quarterly",    "QAH",  4),
    ("yearly",       "YAH",  3),
)


def _resolve_panel_labels(include):
    labels = []
    for cls, suffix, n_max in _PANEL_CLASS_DEFS:
        cfg = include.get(cls, {})
        if not cfg.get("enabled", False):
            continue
        for k in range(1, n_max + 1):
            label = f"{k}{suffix}"
            if cfg.get(label, False):
                labels.append(label)
    return tuple(labels)


STAGE_A_LABELS = _resolve_panel_labels(STAGE_A_INCLUDE)
SUBSET_LABELS  = _resolve_panel_labels(STAGE_B_INCLUDE)


def _period_tag(include):
    """Tag of enabled period classes (e.g. 'weekly_monthly') for output filenames."""
    enabled = [cls for cls in ("weekly", "monthly", "quarterly", "yearly")
                if include.get(cls, {}).get("enabled", False)]
    return "_".join(enabled) if enabled else "empty"


PERIOD_TAG = _period_tag(STAGE_B_INCLUDE)


def date_to_decimal_year(date_str):
    if date_str is None:
        return None
    dt = np.datetime64(date_str)
    year = int(str(dt)[:4])
    year_start = np.datetime64(f"{year}-01-01")
    doy = int((dt - year_start) / np.timedelta64(1, "D"))
    return year + doy / 365.0


def find_start_index(trading_axis, start_date):
    if start_date is None:
        return 0
    if hasattr(trading_axis, "ndim") and trading_axis.ndim > 1:
        trading_axis = trading_axis[:, 0]
    threshold = date_to_decimal_year(start_date)
    return int(np.searchsorted(np.asarray(trading_axis), threshold,
                                side="left"))


def slice_panel_after_date(start_date, *arrays):
    trading_arr = arrays[-1]
    idx = find_start_index(trading_arr, start_date)
    sliced = tuple(np.asarray(a)[idx:] for a in arrays)
    return sliced + (idx,)


# Per-market loader returns (y, mat, dlt, tra).
def _load_panel(labels, include, data_path_fmt, panel_name="panel",
                 sheet_names=None):
    if not labels:
        raise ValueError(
            f"Empty {panel_name}: the include config selected zero contracts. "
            f"Enable at least one (class, maturity) pair.")

    if sheet_names is None:
        sheet_names = SHEETS_DE

    def _max_offset(cls, suffix, n_max):
        cfg = include.get(cls, {})
        if not cfg.get("enabled", False):
            return 0
        return max(
            (k for k in range(1, n_max + 1) if cfg.get(f"{k}{suffix}", False)),
            default=0,
        )
    n_w_load = _max_offset("weekly",    "WAH", 4)
    n_m_load = _max_offset("monthly",   "MAH", 4)
    n_q_load = _max_offset("quarterly", "QAH", 4)
    n_y_load = _max_offset("yearly",    "YAH", 3)

    parts = []
    for year in DATA_YEARS:
        path = data_path_fmt.format(year=year)

        BM = gd.get_data(path, sheet_names["monthly"])
        BQ = gd.get_data(path, sheet_names["quarterly"])
        BY = gd.get_data(path, sheet_names["yearly"])
        n_m_call = max(n_m_load, 1)
        n_q_call = max(n_q_load, 1)
        n_y_call = max(n_y_load, 1)
        mm, qq, yy = gd.build_settlement_matrix(
            BM, BQ, BY, n_m_call, n_q_call, n_y_call)
        (ms, qs, ys), (md, qd, yd), (mt, qt, yt) = \
            gd.build_date_matrices(
                BM, BQ, BY, n_m_call, n_q_call, n_y_call)

        p_w = s_w = d_w = t_w = None
        if n_w_load > 0:
            BW = gd.get_data(path, sheet_names["weekly"])
            p_w           = gd.build_weekly_settlement_matrix(BW, n_weekly=n_w_load)
            s_w, d_w, t_w = gd.build_weekly_date_matrices    (BW, n_weekly=n_w_load)

        def _set(df, col, new_name):
            return df.set_index("Trading Day")[[col]].rename(columns={col: new_name})

        sheet_for = {
            "WAH": (p_w, s_w, d_w, t_w),
            "MAH": (mm,  ms,  md,  mt),
            "QAH": (qq,  qs,  qd,  qt),
            "YAH": (yy,  ys,  yd,  yt),
        }

        cols = []
        for label in labels:
            suffix = label[1:]
            price_df, start_df, dur_df, tra_df = sheet_for[suffix]
            cols += [
                _set(price_df, label, f"price_{label}"),
                _set(start_df, label, f"start_{label}"),
                _set(dur_df,   label, f"dur_{label}"),
                _set(tra_df,   label, f"tra_{label}"),
            ]
        joined = pd.concat(cols, axis=1, join="inner")
        parts.append(joined)

    full = pd.concat(parts, axis=0).sort_index().reset_index().dropna()

    # Weekly sampling: keep only the first trading day of each ISO week.
    if USE_WEEKLY_SAMPLING:
        full = gd.filter_to_first_trading_day_per_iso_week(full)

    y    = full[[f"price_{c}" for c in labels]].to_numpy()
    mat  = full[[f"start_{c}" for c in labels]].to_numpy() / 365.0
    dlt  = full[[f"dur_{c}"   for c in labels]].to_numpy() / 365.0
    tra  = full[[f"tra_{c}"   for c in labels]].to_numpy()
    return y, mat, dlt, tra


def load_stage_a_data():
    """Returns ((y_DE, mat_DE, del_DE, tra_DE), (y_FR, mat_FR, del_FR, tra_FR))."""
    de = _load_panel(STAGE_A_LABELS, STAGE_A_INCLUDE, DATA_PATH_DE,
                      panel_name="Stage A DE", sheet_names=SHEETS_DE)
    fr = _load_panel(STAGE_A_LABELS, STAGE_A_INCLUDE, DATA_PATH_FR,
                      panel_name="Stage A FR", sheet_names=SHEETS_FR)
    return de, fr


def load_stage_b_data():
    """Inner-joins DE and FR Stage B panels on Trading Day."""
    y1, mat_1, del_1, tra_1 = _load_panel(SUBSET_LABELS, STAGE_B_INCLUDE,
                                            DATA_PATH_DE,
                                            panel_name="Stage B DE",
                                            sheet_names=SHEETS_DE)
    y2, mat_2, del_2, tra_2 = _load_panel(SUBSET_LABELS, STAGE_B_INCLUDE,
                                            DATA_PATH_FR,
                                            panel_name="Stage B FR",
                                            sheet_names=SHEETS_FR)

    t1 = tra_1[:, 0]; t2 = tra_2[:, 0]
    common = np.intersect1d(t1, t2)
    idx1 = np.isin(t1, common)
    idx2 = np.isin(t2, common)
    return (y1[idx1], y2[idx2],
            mat_1[idx1], mat_2[idx2],
            del_1[idx1], del_2[idx2],
            tra_1[idx1])


def run_seasonality_grid(t_years, maturity, delivery, y_obs,
                          annual_grid=ANNUAL_GRID,
                          label: str = ""):
    rows, best = [], None
    for ah in annual_grid:
        info = tm.seasonality_bic(t_years, maturity, delivery, y_obs, ah)
        rows.append({"annual_h": ah,
                     "n_obs":  info["n_obs"], "n_eff": info["n_eff"],
                     "k":      info["k"],     "logL":  info["logL"],
                     "sigma2": info["sigma2"],
                     "BIC":    info["BIC"],   "cond_S": info["cond_S"]})
        if best is None or info["BIC"] < best["BIC"]:
            best = info
        print(f"  [{label}] [a={ah:2d}] k={info['k']:2d}  "
              f"logL={info['logL']:.1f}  BIC={info['BIC']:.1f}  "
              f"cond(S)={info['cond_S']:.2e}")
    df = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    return df, best


def heuristic_init(m_per_market: int, N_poly: int):
    """Feasible starting parameter vector for the joint two-market EKF MLE."""
    if m_per_market == 1:
        kappa_Z = np.array([1.5]); kappa_Y = np.array([1.0])
    elif m_per_market == 2:
        kappa_Z = np.array([0.4, 4.0])
        kappa_Y = np.array([0.4, 5.0])
    else:
        raise ValueError(f"m_per_market must be 1 or 2 (got {m_per_market})")

    m = m_per_market
    params = tm.TwoMarketParams(
        kappa_Z = kappa_Z,
        theta_Z = np.zeros(m),
        sigma_Z = np.full(m, 0.1),
        lam_Z   = np.zeros(m),
        kappa_Y = kappa_Y,
        sigma_Y = np.full(m, 0.1),
        lam_Y   = np.zeros(m),
        kappa_R = 2.0,
        # Seed theta_R at its pinned value when PIN_THETA_R is set.
        theta_R = (float(tm.PIN_THETA_R)
                   if tm.PIN_THETA_R is not None else 0.5),
        sigma_R = 0.3,
        lam_R   = 0.0,
        p_delta_1 = 0.0, p_beta_1 = 0.05,
        p_gamma_1 = 0.05 if N_poly >= 5 else 0.0,
        p_delta_2 = 0.0, p_beta_2 = 0.05,
        p_gamma_2 = 0.05 if N_poly >= 5 else 0.0,
        p_e_1 = 0.03,
        p_e_2 = 0.03,
    )
    return tm.pack(params, N_poly=N_poly)


def fit_ekf_model(y_resid_1, y_resid_2,
                   mat_1, del_1, mat_2, del_2,
                   dt: float, m_per_market: int, N_poly: int,
                   de_maxiter: int = 40,
                   seed: int = SEED,
                   tau_ref: float = TAU_REF):
    # Inputs are at the calibration cadence (see _load_panel).
    bounds = tm.make_bounds(m_per_market, N_poly)
    extra = (tau_ref,)
    args  = (y_resid_1, y_resid_2,
              mat_1, del_1, mat_2, del_2,
              dt, N_poly, m_per_market) + extra

    t0_de = time.time()
    if de_maxiter is None:
        x0 = heuristic_init(m_per_market, N_poly)
        print(f"   [warm-start: heuristic, no DE]")
    else:
        f0 = tm.EKF_MLE(heuristic_init(m_per_market, N_poly), *args)
        print(f"   [DE start: heuristic-init -log_lik = {f0:.2f}]")
        de = differential_evolution(
            tm.EKF_MLE, bounds=bounds, args=args,
            seed=seed, maxiter=de_maxiter, tol=1e-3,
            popsize=DE_POPSIZE, mutation=(0.5, 1), recombination=0.7,
            workers=1, polish=False,
        )
        x0 = de.x
        print(f"   [DE done : -log_lik = {de.fun:.2f} in "
              f"{time.time()-t0_de:.1f}s]")

    f_start = float(tm.EKF_MLE(x0, *args))

    t0_lb = time.time()
    lb = minimize(
        fun=tm.EKF_MLE, x0=x0, args=args,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": LBFGS_MAXITER, "ftol": 1e-10, "gtol": 1e-7},
    )
    print(f"   [L-BFGS done: -log_lik = {lb.fun:.2f} in "
          f"{time.time()-t0_lb:.1f}s, success={lb.success}, "
          f"niter={getattr(lb, 'nit', '?')}]")

    # Keep the DE optimum if the polish diverged or did not improve.
    lb_fun = float(lb.fun)
    if (not np.isfinite(lb_fun)) or (lb_fun >= f_start):
        print(f"   [polish rejected: kept start point "
              f"-log_lik = {f_start:.2f} (polish gave {lb_fun:.2f})]")
        return OptimizeResult(
            x=np.asarray(x0, dtype=float), fun=f_start,
            success=False,
            status=getattr(lb, "status", 4),
            message=(f"L-BFGS polish rejected (start={f_start:.4f}, "
                     f"polish={lb_fun:.4f}); kept DE/warm-start point."),
            nit=getattr(lb, "nit", 0),
            jac=getattr(lb, "jac", None),
        )
    return lb


def run_ekf_grid(y_resid_1, y_resid_2,
                  mat_1, del_1, mat_2, del_2,
                  dt: float,
                  m_grid=M_GRID, n_grid=N_POLY_GRID,
                  out_csv=None):
    rows = []
    y1c, y2c = y_resid_1, y_resid_2
    n_obs = int(y1c.size + y2c.size
                 - np.isnan(y1c).sum() - np.isnan(y2c).sum())
    for m in m_grid:
        for N_poly in n_grid:
            de_maxiter = DE_MAXITER_MAP.get((m, N_poly), 50)
            print(f"\n-> fitting m_per_market={m}, N_poly={N_poly} "
                  f"(de_maxiter={de_maxiter})")
            t0     = time.time()
            result = None
            try:
                result  = fit_ekf_model(y_resid_1, y_resid_2,
                                         mat_1, del_1, mat_2, del_2,
                                         dt=dt, m_per_market=m,
                                         N_poly=N_poly,
                                         de_maxiter=de_maxiter)
                log_lik = -float(result.fun) if result.fun < 1e9 else -1e10
                success = bool(result.success)
            except Exception as exc:
                print(f"   FAILED: {exc}")
                log_lik, success = -1e10, False

            if result is not None:
                params_fname = f"params_{PERIOD_TAG}_twomarket_m{m}_N{N_poly}.npy"
                np.save(os.path.join(OUT_DIR, params_fname), result.x)
                print(f"   Saved params -> {params_fname}")

            k   = tm.num_params(m, N_poly)
            bic = k * np.log(n_obs) - 2 * log_lik
            elapsed = time.time() - t0

            rows.append({"m_per_market": m, "N_poly": N_poly,
                          "n_obs": n_obs, "k": k,
                          "logL":  log_lik, "BIC": bic,
                          "success": success, "elapsed_s": elapsed})
            print(f"   logL={log_lik:.2f}  k={k}  BIC={bic:.2f}  "
                  f"(in {elapsed:.1f}s)")

            if out_csv is not None:
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    return pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)


def main():
    print(f"Loading two-market monthly Stage A panels "
          f"{list(STAGE_A_LABELS)} ...")
    (y_a_DE, mat_a_DE, del_a_DE, tra_a_DE), \
    (y_a_FR, mat_a_FR, del_a_FR, tra_a_FR) = load_stage_a_data()
    print(f"  Stage A DE: {y_a_DE.shape[0]} days x {y_a_DE.shape[1]} contracts")
    print(f"  Stage A FR: {y_a_FR.shape[0]} days x {y_a_FR.shape[1]} contracts")

    print(f"\nLoading joint two-market Stage B panel "
          f"{list(SUBSET_LABELS)} ...")
    y1, y2, mat_1, mat_2, del_1, del_2, trading = load_stage_b_data()
    print(f"  Stage B (joined): {y1.shape[0]} days x {y1.shape[1]} contracts "
          f"per market")

    if START_DATE is not None:
        y_a_DE, mat_a_DE, del_a_DE, tra_a_DE, idx_a_DE = \
            slice_panel_after_date(START_DATE, y_a_DE, mat_a_DE,
                                     del_a_DE, tra_a_DE)
        y_a_FR, mat_a_FR, del_a_FR, tra_a_FR, idx_a_FR = \
            slice_panel_after_date(START_DATE, y_a_FR, mat_a_FR,
                                     del_a_FR, tra_a_FR)
        y1, y2, mat_1, mat_2, del_1, del_2, trading, idx_b = \
            slice_panel_after_date(START_DATE, y1, y2,
                                     mat_1, mat_2, del_1, del_2, trading)
        print(f"\nRestricting historical sample to dates >= {START_DATE}:")
        print(f"  Stage A DE: dropped {idx_a_DE} rows; "
              f"new size = {y_a_DE.shape[0]}")
        print(f"  Stage A FR: dropped {idx_a_FR} rows; "
              f"new size = {y_a_FR.shape[0]}")
        print(f"  Stage B (joined): dropped {idx_b} rows; "
              f"new size = {y1.shape[0]}")

    price_scale_1 = float(y_a_DE.mean())
    price_scale_2 = float(y_a_FR.mean())
    y1_a_norm     = y_a_DE / price_scale_1
    y2_a_norm     = y_a_FR / price_scale_2
    y1_norm       = y1     / price_scale_1
    y2_norm       = y2     / price_scale_2
    print(f"  price_scale DE = {price_scale_1:.4f} EUR/MWh")
    print(f"  price_scale FR = {price_scale_2:.4f} EUR/MWh")

    print(f"\n=== Stage A: seasonality BIC grid (DE) ({list(STAGE_A_LABELS)}) ===")
    seas_df_1, best_1 = run_seasonality_grid(
        tra_a_DE[:, 0], mat_a_DE, del_a_DE, y1_a_norm,
        annual_grid=ANNUAL_GRID, label="DE")
    seas_csv_1 = os.path.join(OUT_DIR,
                                f"bic_seasonality_twomarket_DE_{PERIOD_TAG}.csv")
    seas_df_1.to_csv(seas_csv_1, index=False)
    print(f"\nDE best seasonality: a={int(best_1['annual_h'])} "
          f"(BIC={best_1['BIC']:.1f})")
    print(f"Saved {seas_csv_1}")

    print(f"\n=== Stage A: seasonality BIC grid (FR) ({list(STAGE_A_LABELS)}) ===")
    seas_df_2, best_2 = run_seasonality_grid(
        tra_a_FR[:, 0], mat_a_FR, del_a_FR, y2_a_norm,
        annual_grid=ANNUAL_GRID, label="FR")
    seas_csv_2 = os.path.join(OUT_DIR,
                                f"bic_seasonality_twomarket_FR_{PERIOD_TAG}.csv")
    seas_df_2.to_csv(seas_csv_2, index=False)
    print(f"\nFR best seasonality: a={int(best_2['annual_h'])} "
          f"(BIC={best_2['BIC']:.1f})")
    print(f"Saved {seas_csv_2}")

    seas_beta_1 = best_1["beta"]
    seas_beta_2 = best_2["beta"]
    seas_labels = ["c", "m", "a_1", "b_1", "a_2", "b_2"]
    print("  DE seas_beta =", dict(zip(seas_labels, seas_beta_1.tolist())))
    print("  FR seas_beta =", dict(zip(seas_labels, seas_beta_2.tolist())))
    n_t, n_c = mat_1.shape
    _, S_1, _ = tm.build_seasonality_matrix(
        trading[:, 0], mat_1, del_1, y1_norm,
        annual_h=int(best_1["annual_h"]))
    _, S_2, _ = tm.build_seasonality_matrix(
        trading[:, 0], mat_2, del_2, y2_norm,
        annual_h=int(best_2["annual_h"]))
    g_bar_1 = (S_1 @ seas_beta_1).reshape(n_t, n_c)
    g_bar_2 = (S_2 @ seas_beta_2).reshape(n_t, n_c)
    y_resid_1 = y1_norm - g_bar_1
    y_resid_2 = y2_norm - g_bar_2
    print(f"\n  Stage B residual DE: mean={y_resid_1.mean():+.5f}  "
          f"std={y_resid_1.std():.5f}")
    print(f"  Stage B residual FR: mean={y_resid_2.mean():+.5f}  "
          f"std={y_resid_2.std():.5f}")

    print(f"\n=== Stage B: joint two-market EKF BIC grid "
          f"({list(SUBSET_LABELS)} per market) ===")
    print(f"  USE_WEEKLY_SAMPLING={USE_WEEKLY_SAMPLING}  "
          f"DT_EKF={DT_EKF:.6f} years")
    ekf_csv = os.path.join(OUT_DIR, f"bic_ekf_twomarket_{PERIOD_TAG}.csv")
    ekf_df  = run_ekf_grid(y_resid_1, y_resid_2,
                             mat_1, del_1, mat_2, del_2, DT_EKF,
                             m_grid=M_GRID, n_grid=N_POLY_GRID,
                             out_csv=ekf_csv)
    print("\nFinal EKF BIC table:")
    print(ekf_df)
    print(f"Saved {ekf_csv}")


if __name__ == "__main__":
    main()
