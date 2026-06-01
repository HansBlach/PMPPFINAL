"""BIC grid search for the OU (LD_PM) model on the monthly Stage-A / Stage-B panel."""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

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


DATA_YEARS    = range(2023, 2026)
START_DATE    = "2023-05-01"
DATA_PATH_FMT = ("/Users/hansblachfalkenberg/Desktop/Unimat-4/speciale/DE/"
                 "PowerFutureHistory_Phelix-DE_{year}.xlsx")
OUT_DIR       = os.path.dirname(os.path.abspath(__file__))

STAGE_B_INCLUDE = {
    "weekly": {
        "enabled": True,
        "1WAH": True, "2WAH": True,  "3WAH": True, "4WAH": True,
    },
    "monthly": {
        "enabled": False,
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

ANNUAL_GRID   = (2, 2)
M_GRID        = (1,2,3,4)
N_POLY_GRID   = (1,3,5)

# Weekly sampling keeps the first trading day of each ISO week; False falls
# back to daily observations with DT = 1/252. Saved .npy files are dt-specific.
USE_WEEKLY_SAMPLING = True
DT          = 1 / 252.0     # daily fallback step
DT_WEEKLY   = 7 / 365.0     # one ISO week in calendar years
DT_EKF      = DT_WEEKLY if USE_WEEKLY_SAMPLING else DT
SEED        = 42

TAU_REF       = 1.0

FIT_D         = False

# Fit per-factor risk premium (P-mean = mu+lam, Q-mean = mu). Adds m params.
FIT_LAM       = True

# Per-factor polynomial map vs shared map.
INDEPENDENT_POLY = False

DE_MAXITER_MAP = {
    (1, 3): 200,
    (2, 3): 100,
    (3, 3): 200,
    (4, 3): 150,           # m=4 has a bigger param space, give DE more budget
    (1, 5): 200,
    (2, 5): 100,
    (3, 5): 200,
    (4, 5): 150,
}
DE_POPSIZE    = 20
LBFGS_MAXITER = 500


def date_to_decimal_year(date_str):
    """'YYYY-MM-DD' -> decimal year on the trading axis. Inverse of trading_days_to_dt."""
    if date_str is None:
        return None
    dt = np.datetime64(date_str)
    year = int(str(dt)[:4])
    year_start = np.datetime64(f"{year}-01-01")
    doy = int((dt - year_start) / np.timedelta64(1, "D"))
    return year + doy / 365.0


def find_start_index(trading_axis, start_date):
    """First index i with trading[i] >= start_date; 0 if start_date is None."""
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


STAGE_A_INCLUDE = {
    "weekly": {
        "enabled": True,
        "1WAH": True , "2WAH": True, "3WAH": True, "4WAH": True,
    },
    "monthly": {
        "enabled": False,
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


def _period_tag(include):
    enabled = [cls for cls in ("weekly", "monthly", "quarterly", "yearly")
                if include.get(cls, {}).get("enabled", False)]
    return "_".join(enabled) if enabled else "empty"


PERIOD_TAG = _period_tag(STAGE_B_INCLUDE)
SUBSET_LABELS  = _resolve_panel_labels(STAGE_B_INCLUDE)


def _load_panel(labels, include, panel_name="panel"):
    """Generic panel loader - inner-joins selected contracts on Trading Day."""
    if not labels:
        raise ValueError(
            f"Empty {panel_name}: the include config selected zero contracts. "
            f"Enable at least one (class, maturity) pair.")

    # Highest selected maturity per class so we don't waste pivot rows.
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
        path = DATA_PATH_FMT.format(year=year)

        DEBM = gd.get_data(path, "DEBM")
        DEBQ = gd.get_data(path, "DEBQ")
        DEBY = gd.get_data(path, "DEBY")
        # build_settlement_matrix requires nonzero n's, so pass 1 as a placeholder.
        n_m_call = max(n_m_load, 1)
        n_q_call = max(n_q_load, 1)
        n_y_call = max(n_y_load, 1)
        mm, qq, yy = gd.build_settlement_matrix(
            DEBM, DEBQ, DEBY, n_m_call, n_q_call, n_y_call)
        (ms, qs, ys), (md, qd, yd), (mt, qt, yt) = \
            gd.build_date_matrices(
                DEBM, DEBQ, DEBY, n_m_call, n_q_call, n_y_call)

        p_w = s_w = d_w = t_w = None
        if n_w_load > 0:
            DEBW = gd.get_data(path, "DEB1-5")
            p_w           = gd.build_weekly_settlement_matrix(DEBW, n_weekly=n_w_load)
            s_w, d_w, t_w = gd.build_weekly_date_matrices    (DEBW, n_weekly=n_w_load)

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
            suffix = label[1:]    # e.g. "MAH" from "2MAH"
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

    # Keep the first trading day of each ISO week.
    if USE_WEEKLY_SAMPLING:
        full = gd.filter_to_first_trading_day_per_iso_week(full)

    y    = full[[f"price_{c}" for c in labels]].to_numpy()
    mat  = full[[f"start_{c}" for c in labels]].to_numpy() / 365.0
    dlt  = full[[f"dur_{c}"   for c in labels]].to_numpy() / 365.0
    tra  = full[[f"tra_{c}"   for c in labels]].to_numpy()
    return y, mat, dlt, tra


def load_stage_a_data():
    return _load_panel(STAGE_A_LABELS, STAGE_A_INCLUDE, panel_name="Stage A")


def load_stage_b_data():
    return _load_panel(SUBSET_LABELS, STAGE_B_INCLUDE, panel_name="Stage B")


def run_seasonality_grid(t_years, maturity, delivery, y_obs,
                         annual_grid=ANNUAL_GRID):
    rows = []
    best = None
    for ah in annual_grid:
        info = ld.seasonality_bic(t_years, maturity, delivery, y_obs, ah)
        rows.append({"annual_h": ah,
                     "n_obs": info["n_obs"], "n_eff": info["n_eff"],
                     "k": info["k"],
                     "logL": info["logL"], "sigma2": info["sigma2"],
                     "BIC":  info["BIC"], "cond_S": info["cond_S"]})
        if best is None or info["BIC"] < best["BIC"]:
            best = info
        print(f"  [a={ah}] k={info['k']:2d}  "
              f"logL={info['logL']:.1f}  BIC={info['BIC']:.1f}  "
              f"cond(S)={info['cond_S']:.2e}")
    df = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    return df, best


def heuristic_init(m, N_poly, independent_poly=INDEPENDENT_POLY):
    kwargs = dict(
        theta   = np.full(m, 1.0),
        mu      = np.concatenate([[0.0], np.zeros(m - 1)]),
        lam     = np.concatenate([[0.0], np.zeros(m - 1)]),
        c       = np.full(m, 0.1),
        d       = np.zeros(m),
        rho     = np.eye(m),
        p_delta = 0.0,
        p_beta  = 0.05,
        p_gamma = 0.05 if N_poly >= 5 else 0.0,
        p_e     = 0.03,
    )
    if independent_poly:
        kwargs["independent_poly"] = True
        kwargs["p_beta_arr"]  = np.full(m, 0.05)
        if N_poly >= 5:
            kwargs["p_gamma_arr"] = np.full(m, 0.05)
            kwargs["p_K_arr"]     = np.zeros(m)
    params = ld.LinearDiffusionParams(**kwargs)
    return ld.pack_ld(params, N_poly=N_poly, fit_d=FIT_D)


def fit_ekf_model(y_resid, maturity, delivery, dt, m, N_poly,
                  de_maxiter=80, seed=SEED,
                  tau_ref=TAU_REF,
                  independent_poly=INDEPENDENT_POLY):
    # Inputs are at the calibration cadence; dt should be DT_EKF.
    bounds = ld.make_bounds(m, N_poly, fit_d=FIT_D,
                            independent_poly=independent_poly)
    extra  = (tau_ref, FIT_D, independent_poly)
    args   = (y_resid, maturity, delivery, dt, N_poly, m) + extra

    if de_maxiter is None:
        x0 = heuristic_init(m, N_poly)
    else:
        de = differential_evolution(
            ld.EKF_MLE, bounds=bounds, args=args,
            seed=seed, maxiter=de_maxiter, tol=1e-3,
            popsize=DE_POPSIZE, mutation=(0.5, 1), recombination=0.7,
            workers=1, polish=False,
        )
        x0 = de.x

    lb = minimize(
        fun=ld.EKF_MLE, x0=x0, args=args,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": LBFGS_MAXITER, "ftol": 1e-12, "gtol": 1e-8},
    )
    return lb


def run_ekf_grid(y_resid, maturity, delivery, dt,
                 m_grid=M_GRID, n_grid=N_POLY_GRID,
                 out_csv=None):
    rows  = []
    # n_obs feeds the BIC penalty.
    y_for_count = y_resid
    n_obs  = int(y_for_count.shape[0] * y_for_count.shape[1]
                 - np.isnan(y_for_count).sum())
    for m in m_grid:
        for N_poly in n_grid:
            de_maxiter = DE_MAXITER_MAP.get((m, N_poly), 60)
            print(f"\n-> fitting m={m}, N_poly={N_poly} "
                  f"(de_maxiter={de_maxiter})")
            t0 = time.time()
            result = None
            try:
                result  = fit_ekf_model(y_resid, maturity, delivery, dt,
                                        m=m, N_poly=N_poly,
                                        de_maxiter=de_maxiter)
                log_lik = -float(result.fun) if result.fun < 1e9 else -1e10
                success = bool(result.success)
            except Exception as exc:
                print(f"   FAILED: {exc}")
                log_lik, success = -1e10, False

            indep_tag = "_indep" if INDEPENDENT_POLY else ""
            lam_tag   = "_lam"   if FIT_LAM         else ""
            if result is not None:
                params_fname = (f"params_{PERIOD_TAG}_ou_m{m}_N{N_poly}"
                                f"{indep_tag}{lam_tag}.npy")
                np.save(os.path.join(OUT_DIR, params_fname), result.x)
                print(f"   Saved params -> {params_fname}")

            k   = ld.num_params_ld(m, N_poly,
                                   fit_d=FIT_D,
                                   independent_poly=INDEPENDENT_POLY)
            # N=1: p_delta aliases mu and p_beta is unused; drop them so k matches LLR.
            if N_poly == 1:
                k -= 1
                k -= m if INDEPENDENT_POLY else 1
            bic = k * np.log(n_obs) - 2 * log_lik
            elapsed = time.time() - t0

            rows.append({"m": m, "N_poly": N_poly, "n_obs": n_obs, "k": k,
                         "logL": log_lik, "BIC": bic,
                         "success": success, "elapsed_s": elapsed})
            print(f"   logL={log_lik:.2f}  k={k}  BIC={bic:.2f}  "
                  f"(in {elapsed:.1f}s)")

            if out_csv is not None:
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    return pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)


def main():
    print(f"Loading Stage A panel {list(STAGE_A_LABELS)} ...")
    y_stagea, mat_stagea, del_stagea, tra_stagea = load_stage_a_data()
    print(f"  n_days          = {y_stagea.shape[0]}")
    print(f"  n_contracts     = {y_stagea.shape[1]}  -> "
          f"{tuple(STAGE_A_LABELS)}")

    print(f"\nLoading Stage B panel {list(SUBSET_LABELS)} ...")
    y_matrix, maturity, delivery, trading = load_stage_b_data()
    print(f"  n_days_subset   = {y_matrix.shape[0]}")
    print(f"  n_contracts     = {y_matrix.shape[1]}")

    # Same cut on both panels keeps Stage A OLS and Stage B MLE aligned.
    if START_DATE is not None:
        y_stagea, mat_stagea, del_stagea, tra_stagea, idx_a = \
            slice_panel_after_date(START_DATE, y_stagea, mat_stagea,
                                    del_stagea, tra_stagea)
        y_matrix, maturity, delivery, trading, idx_b = \
            slice_panel_after_date(START_DATE, y_matrix, maturity,
                                    delivery, trading)
        print(f"\nRestricting historical sample to dates >= {START_DATE}:")
        print(f"  Stage A: dropped {idx_a} rows; new size = {y_stagea.shape[0]}")
        print(f"  Stage B: dropped {idx_b} rows; new size = {y_matrix.shape[0]}")

    # Shared price scale = Stage A mean, so seas_beta can be reused on Stage B.
    price_scale  = float(y_stagea.mean())
    y_stagea_norm = y_stagea / price_scale
    y_norm        = y_matrix / price_scale
    print(f"  price_scale (Stage A mean) = {price_scale:.4f} EUR/MWh")
    print(f"  Stage B mean / Stage A mean = "
          f"{y_matrix.mean() / price_scale:.4f}  "
          f"(deviation absorbed by mu/lam in Stage B)")

    print(f"\n=== Stage A: seasonality BIC grid "
          f"({list(STAGE_A_LABELS)}) ===")
    seas_df, best_seas = run_seasonality_grid(
        tra_stagea[:, 0], mat_stagea, del_stagea, y_stagea_norm,
        annual_grid=ANNUAL_GRID,
    )
    seas_csv = os.path.join(OUT_DIR,
                             f"bic_seasonality_ou_{PERIOD_TAG}.csv")
    seas_df.to_csv(seas_csv, index=False)
    ah_best = int(best_seas["annual_h"])
    print(f"\nBest seasonality: annual_h={ah_best} "
          f"(BIC={best_seas['BIC']:.1f})")
    print(f"Saved {seas_csv}")

    # Rebuild the design matrix on the Stage B panel and re-evaluate the same g(t).
    seas_beta = best_seas["beta"]
    print("  seas_beta =", dict(zip(
    ["c", "m", "a_1", "b_1", "a_2", "b_2"], seas_beta.tolist())))
    _, S_sub, _ = ld.build_seasonality_matrix(
        trading[:, 0], maturity, delivery, y_norm,
        annual_h=ah_best,
    )
    n_t, n_c   = maturity.shape
    g_bar      = (S_sub @ seas_beta).reshape(n_t, n_c)
    y_resid    = y_norm - g_bar
    print(f"  Stage B residual: mean={y_resid.mean():+.5f}  "
          f"std={y_resid.std():.5f}")

    print(f"\n=== Stage B: EKF BIC grid ({list(SUBSET_LABELS)} residuals) ===")
    print(f"  USE_WEEKLY_SAMPLING={USE_WEEKLY_SAMPLING}  "
          f"DT_EKF={DT_EKF:.6f} years  "
          f"({'weekly ISO-Mon' if USE_WEEKLY_SAMPLING else 'daily'})")
    ekf_csv = os.path.join(OUT_DIR, f"bic_ekf_ou_{PERIOD_TAG}.csv")
    ekf_df  = run_ekf_grid(y_resid, maturity, delivery, DT_EKF,
                            m_grid=M_GRID, n_grid=N_POLY_GRID,
                            out_csv=ekf_csv)
    print("\nFinal EKF BIC table:")
    print(ekf_df)
    print(f"Saved {ekf_csv}")


if __name__ == "__main__":
    main()
