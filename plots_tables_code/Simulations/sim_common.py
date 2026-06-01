"""Shared helpers for the simulate_paths_* scripts (OU, Jacobi, two-market).

"""
import numpy as np


def trading_days_to_dt(t_decimal):
    """Decimal-year array -> numpy datetime64[D]."""
    if np.ndim(t_decimal) > 1:
        t_decimal = t_decimal[:, 0]
    years = np.floor(t_decimal).astype(int)
    days  = ((t_decimal - years) * 365.0).round().astype(int)
    return np.array([np.datetime64(f"{y}-01-01") + np.timedelta64(d, "D")
                     for y, d in zip(years, days)])


def detect_cycle_len(maturity_col, fallback=21):
    """Trading-day roll cycle inferred from maturity jumps; fallback if too few rolls."""
    diffs = np.diff(np.asarray(maturity_col, dtype=float))
    roll_idx = np.where(diffs > 0)[0]
    if len(roll_idx) < 2:
        return int(fallback)
    return int(np.median(np.diff(roll_idx)))


def build_per_col_rolling_schedule(maturity_hist, delivery_hist, n_steps,
                                    cycle_lens):
    """Tile the last cycle_lens[c] historical maturities per column to keep the roll cadence."""
    n_c = maturity_hist.shape[1]
    fut_mat = np.empty((n_steps, n_c))
    fut_del = np.empty((n_steps, n_c))
    for c in range(n_c):
        cl = max(1, int(cycle_lens[c]))
        mat_cyc = np.asarray(maturity_hist[-cl:, c], dtype=float)
        del_cyc = np.asarray(delivery_hist[-cl:, c], dtype=float)
        idx = np.arange(n_steps) % cl
        fut_mat[:, c] = mat_cyc[idx]
        fut_del[:, c] = del_cyc[idx]
    return fut_mat, fut_del


def ks_2samp(obs, sim):
    """Two-sample Kolmogorov-Smirnov via empirical CDFs on the union grid."""
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


# Back-compatible aliases for the names the scripts used previously.
_trading_to_dt = trading_days_to_dt
_detect_cycle_len = detect_cycle_len
_build_per_col_rolling_schedule = build_per_col_rolling_schedule
_ks_2samp = ks_2samp
