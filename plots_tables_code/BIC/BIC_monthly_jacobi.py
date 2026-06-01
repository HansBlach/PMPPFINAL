"""BIC grid search for the Jacobi PMPP model - sister to BIC_monthly_OU.py."""

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
import kalman_filter_jacobi as ld


DATA_YEARS    = range(2023, 2026)
START_DATE    = "2023-05-01"
DATA_PATH_FMT = ("/Users/hansblachfalkenberg/Desktop/Unimat-4/speciale/DE/"
                 "PowerFutureHistory_Phelix-DE_{year}.xlsx")
OUT_DIR       = os.path.dirname(os.path.abspath(__file__))

N_MONTHLY     = 3
N_QUARTERLY   = 4
N_YEARLY      = 3

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
M_GRID        = (1,2)
# Even N_poly forces the trailing alpha to 0 (deg 2k, not 2k+1); N_poly=1 is
# the identity map.
N_POLY_GRID   = (1,2,3,4,5)


PER_FACTOR_C  = False

# Spot envelope target. p_delta/c_tilde bounds are derived from these and the
# fitted seasonality so the spot corridor lands in this range (EEX band).
SPOT_LO_EUR   = -500.0
SPOT_HI_EUR   = +4000.0

# Weekly sampling keeps the first trading day of each ISO week; False is daily.
USE_WEEKLY_SAMPLING = True
DT          = 1 / 252.0
DT_WEEKLY   = 7 / 365.0
DT_EKF      = DT_WEEKLY if USE_WEEKLY_SAMPLING else DT
SEED        = 42

TAU_REF       = ld.TAU_REF_DEFAULT


SIGMA_INIT = 0.35

DE_MAXITER_MAP = {
    (1, 2): 200, (2, 2): 200, (3, 2): 200,
    (1, 3): 200, (2, 3): 200, (3, 3): 200,
    (1, 4): 200, (2, 4): 200, (3, 4): 200,
    (1, 5): 200, (2, 5): 200, (3, 5): 200,
}
DE_POPSIZE    = 10
LBFGS_MAXITER = 500

STAGE_A_INCLUDE = {
    "weekly": {
        "enabled": True,
        "1WAH": True, "2WAH": True, "3WAH": True, "4WAH": True,
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
SUBSET_LABELS  = _resolve_panel_labels(STAGE_B_INCLUDE)


def _period_tag(include):
    """Tag of enabled period classes (e.g. 'weekly_monthly') for output filenames."""
    enabled = [cls for cls in ("weekly", "monthly", "quarterly", "yearly")
                if include.get(cls, {}).get("enabled", False)]
    return "_".join(enabled) if enabled else "empty"


# Derived from Stage B, which is what the EKF fits.
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


def _load_panel(labels, include, panel_name="panel"):
    if not labels:
        raise ValueError(
            f"Empty {panel_name}: the include config selected zero contracts. "
            f"Enable at least one (class, maturity) pair.")

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


def heuristic_init(m, N_poly, per_factor_c=PER_FACTOR_C,
                    sigma_init=SIGMA_INIT):

    kappa_seeds = {
        1: np.array([1.5]),
        2: np.array([0.4, 3.5]),
        3: np.array([0.25, 1.0, 4.5]),
    }
    kappa = kappa_seeds[m]
    theta = np.full(m, 0.5)
    lam   = np.zeros(m)
    sigma = np.full(m, float(sigma_init))

    target = 1.0 + ld.AB_MARGIN
    sigma_q_max = np.sqrt(2.0 * kappa * np.minimum(theta, 1.0 - theta)
                            / target)
    sigma = np.minimum(sigma, 0.9 * sigma_q_max)

    k = ld.k_from_N(N_poly)
    alpha_tilde = np.zeros((m, k))
    beta_tilde  = np.zeros((m, k))
    c_size      = m if per_factor_c else 1
    c_tilde     = np.zeros(c_size)

    params = ld.jacobiParams(
        kappa=kappa, theta=theta, lam=lam, sigma=sigma, p_e=0.03,
        p_delta=0.0,
        alpha_tilde=alpha_tilde, beta_tilde=beta_tilde,
        c_tilde=c_tilde, per_factor_c=per_factor_c,
    )
    return ld.pack_Jacobi(params, N_poly=N_poly)


def fit_ekf_model(y_resid, maturity, delivery, dt, m, N_poly,
                   de_maxiter=80, seed=SEED,
                   tau_ref=TAU_REF, per_factor_c=PER_FACTOR_C,
                   spot_envelope=None):
    # Inputs are at the calibration cadence (see _load_panel).
    bounds = ld.make_bounds(m, N_poly, per_factor_c=per_factor_c,
                              spot_envelope=spot_envelope)

    extra = (tau_ref, per_factor_c)
    args  = (y_resid, maturity, delivery, dt, N_poly, m) + extra

    if de_maxiter is None:
        x0 = heuristic_init(m, N_poly, per_factor_c=per_factor_c)
    else:
        de = differential_evolution(
            ld.EKF_MLE, bounds=bounds, args=args,
            seed=seed, maxiter=de_maxiter, tol=1e-3,
            popsize=DE_POPSIZE, mutation=(0.5, 1), recombination=0.7,
            workers=1, polish=False,
        )
        x0 = de.x

    f_start = float(ld.EKF_MLE(x0, *args))

    lb = minimize(
        fun=ld.EKF_MLE, x0=x0, args=args,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": LBFGS_MAXITER, "ftol": 1e-12, "gtol": 1e-8},
    )

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


def run_ekf_grid(y_resid, maturity, delivery, dt,
                  m_grid=M_GRID, n_grid=N_POLY_GRID,
                  out_csv=None, per_factor_c=PER_FACTOR_C,
                  spot_envelope=None):
    rows  = []
    # n_obs feeds the BIC penalty; data is already at the calibration cadence.
    y_for_count = y_resid
    n_obs  = int(y_for_count.shape[0] * y_for_count.shape[1]
                 - np.isnan(y_for_count).sum())
    mode_tag = "perfac" if per_factor_c else "global"
    print(f"  EKF grid amplitude mode: {mode_tag} "
          f"(per_factor_c={per_factor_c})")
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
                                        de_maxiter=de_maxiter,
                                        per_factor_c=per_factor_c,
                                        spot_envelope=spot_envelope)
                log_lik = -float(result.fun) if result.fun < 1e9 else -1e10
                success = bool(result.success)
            except Exception as exc:
                print(f"   FAILED: {exc}")
                log_lik, success = -1e10, False

            if result is not None:

                params_fname = (
                    f"params_{PERIOD_TAG}_jacobi_m{m}_N{N_poly}_"
                    f"{mode_tag}_lamratio.npy")
                np.save(os.path.join(OUT_DIR, params_fname), result.x)
                print(f"   Saved params -> {params_fname}")

            k_full = ld.num_params_ld(m, N_poly, per_factor_c=per_factor_c)

            if spot_envelope is not None:
                c_size = m if per_factor_c else 1
                n_pinned = 1 + c_size                      # p_delta + c_tilde
            else:
                n_pinned = 0
            k   = k_full - n_pinned
            bic = k * np.log(n_obs) - 2 * log_lik
            elapsed = time.time() - t0

            rows.append({"m": m, "N_poly": N_poly,
                         "per_factor_c": per_factor_c,
                         "n_obs": n_obs,
                         "k_full": k_full, "k_pinned": n_pinned, "k": k,
                         "logL": log_lik, "BIC": bic,
                         "success": success, "elapsed_s": elapsed})
            if n_pinned > 0:
                print(f"   logL={log_lik:.2f}  k={k} "
                      f"(full={k_full}, pinned={n_pinned})  "
                      f"BIC={bic:.2f}  (in {elapsed:.1f}s)")
            else:
                print(f"   logL={log_lik:.2f}  k={k}  BIC={bic:.2f}  "
                      f"(in {elapsed:.1f}s)")

            if out_csv is not None:
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    return pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)


def _print_spot_bound_diagnostic(t_years, seas_beta, annual_h, price_scale,
                                   spot_lo_eur=-500.0, spot_hi_eur=+4000.0,
                                   m_grid=M_GRID, per_factor_c=PER_FACTOR_C):
    """Return the spot seasonality range (g_S_min, g_S_max) over the trading
    window; (None, None) if the basis width does not match seas_beta."""
    t = np.asarray(t_years, dtype=float)
    cols = [np.ones_like(t), t]
    for k in range(1, int(annual_h) + 1):
        omega = 2.0 * np.pi * k
        cols.append(np.cos(omega * t))
        cols.append(np.sin(omega * t))
    S_spot = np.column_stack(cols)
    beta = np.asarray(seas_beta, dtype=float)
    if S_spot.shape[1] != beta.shape[0]:
        print(f"  [spot-bound diagnostic skipped: basis width "
              f"{S_spot.shape[1]} != seas_beta length {beta.shape[0]}; "
              f"check annual_h]")
        return None, None
    g_S     = S_spot @ beta
    g_S_min = float(g_S.min())
    g_S_max = float(g_S.max())

    print(f"  spot g_S(t) range = [{g_S_min:.4f}, {g_S_max:.4f}]  "
          f"(raw [{price_scale*g_S_min:.1f}, {price_scale*g_S_max:.1f}] EUR/MWh)")
    return g_S_min, g_S_max


def main():
    print(f"Loading Stage A panel {list(STAGE_A_LABELS)} ...")
    y_stagea, mat_stagea, del_stagea, tra_stagea = load_stage_a_data()
    print(f"  n_days       = {y_stagea.shape[0]}")
    print(f"  n_contracts  = {y_stagea.shape[1]}  -> {tuple(STAGE_A_LABELS)}")

    print(f"\nLoading Stage B panel {list(SUBSET_LABELS)} ...")
    y_matrix, maturity, delivery, trading = load_stage_b_data()
    print(f"  n_days       = {y_matrix.shape[0]}")
    print(f"  n_contracts  = {y_matrix.shape[1]}")

    # Optional date restriction.
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

    price_scale  = float(y_stagea.mean())
    y_stagea_norm = y_stagea / price_scale
    y_norm        = y_matrix / price_scale
    print(f"  price_scale (Stage A mean) = {price_scale:.4f} EUR/MWh")

    # Stage A seasonality grid.
    print(f"\n=== Stage A: seasonality BIC grid ({list(STAGE_A_LABELS)}) ===")
    seas_df, best_seas = run_seasonality_grid(
        tra_stagea[:, 0], mat_stagea, del_stagea, y_stagea_norm,
        annual_grid=ANNUAL_GRID,
    )
    seas_csv = os.path.join(OUT_DIR,
                             f"bic_seasonality_jacobi_{PERIOD_TAG}.csv")
    seas_df.to_csv(seas_csv, index=False)
    ah_best = int(best_seas["annual_h"])
    print(f"\nBest seasonality: annual_h={ah_best} "
          f"(BIC={best_seas['BIC']:.1f})")
    print(f"Saved {seas_csv}")

    # Apply seasonality to Stage B.
    seas_beta = best_seas["beta"]
    _, S_sub, _ = ld.build_seasonality_matrix(
        trading[:, 0], maturity, delivery, y_norm,
        annual_h=ah_best,
    )
    n_t, n_c = maturity.shape
    g_bar    = (S_sub @ seas_beta).reshape(n_t, n_c)
    y_resid  = y_norm - g_bar
    print(f"  Stage B residual: mean={y_resid.mean():+.5f}  "
          f"std={y_resid.std():.5f}")

    # g_S range feeds the spot_envelope dict, which derives the (p_delta,
    # c_tilde) bounds in make_bounds from price_scale and the fitted seasonality.
    g_S_min, g_S_max = _print_spot_bound_diagnostic(
        t_years=tra_stagea[:, 0],
        seas_beta=seas_beta,
        annual_h=ah_best,
        price_scale=price_scale,
        spot_lo_eur=SPOT_LO_EUR, spot_hi_eur=SPOT_HI_EUR,
        m_grid=M_GRID, per_factor_c=PER_FACTOR_C,
    )
    if g_S_min is not None and g_S_max is not None:
        spot_envelope = {
            "price_scale": price_scale,
            "g_S_min":     g_S_min,
            "g_S_max":     g_S_max,
            "spot_lo_eur": SPOT_LO_EUR,
            "spot_hi_eur": SPOT_HI_EUR,
        }
        print(f"  -> spot envelope passed to make_bounds: "
              f"price_scale={price_scale:.4f}, "
              f"g_S=[{g_S_min:.4f}, {g_S_max:.4f}], "
              f"spot target=[{SPOT_LO_EUR:+.0f}, {SPOT_HI_EUR:+.0f}] EUR/MWh")
    else:
        spot_envelope = None
        print(f"  -> spot envelope unavailable; make_bounds will fall back "
              f"to its static pinning.")

    # Stage B EKF grid.
    mode_tag = "perfac" if PER_FACTOR_C else "global"
    print(f"\n=== Stage B: EKF BIC grid ({list(SUBSET_LABELS)} residuals, "
          f"amplitude mode = {mode_tag}) ===")
    print(f"  USE_WEEKLY_SAMPLING={USE_WEEKLY_SAMPLING}  "
          f"DT_EKF={DT_EKF:.6f} years")
    ekf_csv = os.path.join(
        OUT_DIR, f"bic_ekf_jacobi_{PERIOD_TAG}_{mode_tag}.csv")
    ekf_df  = run_ekf_grid(y_resid, maturity, delivery, DT_EKF,
                            m_grid=M_GRID, n_grid=N_POLY_GRID,
                            out_csv=ekf_csv,
                            per_factor_c=PER_FACTOR_C,
                            spot_envelope=spot_envelope)
    print("\nFinal EKF BIC table:")
    print(ekf_df)
    print(f"Saved {ekf_csv}")



if __name__ == "__main__":
    main()
