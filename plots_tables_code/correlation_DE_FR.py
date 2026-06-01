"""DE/FR power-futures correlation analysis on the BIC_monthly_twomarkets sample.

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

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

import BIC_monthly_twomarkets as btm


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def decimal_year_to_timestamp(dec_year: float) -> pd.Timestamp:
    """Inverse of BIC_monthly_twomarkets.date_to_decimal_year."""
    year = int(np.floor(dec_year))
    doy = (dec_year - year) * 365.0
    start = pd.Timestamp(year=year, month=1, day=1)
    return start + pd.Timedelta(days=float(doy))


def _datetime_index_from_decyear(arr) -> pd.DatetimeIndex:
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return pd.DatetimeIndex(
        [decimal_year_to_timestamp(t).normalize() for t in arr],
        name="trading_day",
    )


def load_joint_panel() -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Return (prices_DE, prices_FR, labels). Each price frame is indexed by
    trading day and has one column per maturity label (1MAH, 2MAH, ...)."""

    y1, y2, _mat1, _mat2, _del1, _del2, trading = btm.load_stage_b_data()

    # Apply the same START_DATE slice the BIC pipeline uses.
    if btm.START_DATE is not None:
        y1, y2, _, _, _, _, trading, idx_b = btm.slice_panel_after_date(
            btm.START_DATE, y1, y2, _mat1, _mat2, _del1, _del2, trading)
        print(f"Sliced {idx_b} rows before {btm.START_DATE}; "
              f"remaining {y1.shape[0]} weekly observations.")

    labels = list(btm.SUBSET_LABELS)
    idx = _datetime_index_from_decyear(trading)

    px_DE = pd.DataFrame(y1, index=idx, columns=[f"DE_{c}" for c in labels])
    px_FR = pd.DataFrame(y2, index=idx, columns=[f"FR_{c}" for c in labels])
    return px_DE, px_FR, labels


def load_deseasonalised_residuals() -> Tuple[pd.DataFrame, pd.DataFrame, list,
                                              dict]:
    """Replicate BIC_monthly_twomarkets.main() 
    """
    tm = btm.tm  # alias the Kalman_filter_TwoMarket module

    (y_a_DE, mat_a_DE, del_a_DE, tra_a_DE), \
    (y_a_FR, mat_a_FR, del_a_FR, tra_a_FR) = btm.load_stage_a_data()

    y1, y2, mat_1, mat_2, del_1, del_2, trading = btm.load_stage_b_data()

    if btm.START_DATE is not None:
        y_a_DE, mat_a_DE, del_a_DE, tra_a_DE, _ = \
            btm.slice_panel_after_date(btm.START_DATE,
                                        y_a_DE, mat_a_DE, del_a_DE, tra_a_DE)
        y_a_FR, mat_a_FR, del_a_FR, tra_a_FR, _ = \
            btm.slice_panel_after_date(btm.START_DATE,
                                        y_a_FR, mat_a_FR, del_a_FR, tra_a_FR)
        y1, y2, mat_1, mat_2, del_1, del_2, trading, _ = \
            btm.slice_panel_after_date(btm.START_DATE,
                                        y1, y2, mat_1, mat_2,
                                        del_1, del_2, trading)

    # Per-market price normalisation (matches main()).
    price_scale_DE = float(y_a_DE.mean())
    price_scale_FR = float(y_a_FR.mean())
    y1_a_norm = y_a_DE / price_scale_DE
    y2_a_norm = y_a_FR / price_scale_FR
    y1_norm   = y1     / price_scale_DE
    y2_norm   = y2     / price_scale_FR

    # Stage A seasonality grid (per market). Quiet wrapper -- run_seasonality_grid
    # prints a small report; we want both the table and the chosen info.
    print("  Stage A seasonality BIC (DE):")
    seas_df_DE, best_DE = btm.run_seasonality_grid(
        tra_a_DE[:, 0], mat_a_DE, del_a_DE, y1_a_norm,
        annual_grid=btm.ANNUAL_GRID, label="DE")
    print("  Stage A seasonality BIC (FR):")
    seas_df_FR, best_FR = btm.run_seasonality_grid(
        tra_a_FR[:, 0], mat_a_FR, del_a_FR, y2_a_norm,
        annual_grid=btm.ANNUAL_GRID, label="FR")

    # Apply each market's seasonality beta to the joint Stage B panel.
    n_t, n_c = mat_1.shape
    _, S_DE, _ = tm.build_seasonality_matrix(
        trading[:, 0], mat_1, del_1, y1_norm,
        annual_h=int(best_DE["annual_h"]))
    _, S_FR, _ = tm.build_seasonality_matrix(
        trading[:, 0], mat_2, del_2, y2_norm,
        annual_h=int(best_FR["annual_h"]))
    g_bar_DE = (S_DE @ best_DE["beta"]).reshape(n_t, n_c)
    g_bar_FR = (S_FR @ best_FR["beta"]).reshape(n_t, n_c)
    resid_DE = y1_norm - g_bar_DE
    resid_FR = y2_norm - g_bar_FR

    labels = list(btm.SUBSET_LABELS)
    idx = _datetime_index_from_decyear(trading)

    df_DE = pd.DataFrame(resid_DE, index=idx,
                          columns=[f"DE_{c}" for c in labels])
    df_FR = pd.DataFrame(resid_FR, index=idx,
                          columns=[f"FR_{c}" for c in labels])
    info = {
        "annual_h_DE":   int(best_DE["annual_h"]),
        "annual_h_FR":   int(best_FR["annual_h"]),
        "price_scale_DE": price_scale_DE,
        "price_scale_FR": price_scale_FR,
        "bic_DE":         seas_df_DE,
        "bic_FR":         seas_df_FR,
    }
    return df_DE, df_FR, labels, info


def matched_correlations(px_DE: pd.DataFrame, px_FR: pd.DataFrame,
                          labels: list) -> pd.DataFrame:
    """Per-maturity correlation: levels, log returns, and Spearman on log
    returns. Reports n_obs after dropping NaNs/inf produced by the log-diff."""
    rows = []
    log_DE = np.log(px_DE.where(px_DE > 0))
    log_FR = np.log(px_FR.where(px_FR > 0))
    ret_DE = log_DE.diff()
    ret_FR = log_FR.diff()

    for lab in labels:
        col_de = f"DE_{lab}"
        col_fr = f"FR_{lab}"

        lvl = pd.concat([px_DE[col_de], px_FR[col_fr]], axis=1).dropna()
        ret = pd.concat([ret_DE[col_de], ret_FR[col_fr]], axis=1).dropna()
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()

        rows.append({
            "maturity":          lab,
            "n_levels":          int(len(lvl)),
            "n_returns":         int(len(ret)),
            "pearson_levels":    float(lvl.corr().iloc[0, 1]),
            "pearson_logret":    float(ret.corr(method="pearson").iloc[0, 1]),
            "spearman_logret":   float(ret.corr(method="spearman").iloc[0, 1]),
            "std_logret_DE":     float(ret.iloc[:, 0].std()),
            "std_logret_FR":     float(ret.iloc[:, 1].std()),
        })

    return pd.DataFrame(rows)


def rolling_correlations(px_DE: pd.DataFrame, px_FR: pd.DataFrame,
                          labels: list, window: int) -> pd.DataFrame:
    """Rolling Pearson correlation of log returns, one column per maturity."""
    log_DE = np.log(px_DE.where(px_DE > 0))
    log_FR = np.log(px_FR.where(px_FR > 0))
    ret_DE = log_DE.diff()
    ret_FR = log_FR.diff()

    out = {}
    for lab in labels:
        a = ret_DE[f"DE_{lab}"]
        b = ret_FR[f"FR_{lab}"]
        out[f"rho_{lab}"] = a.rolling(window, min_periods=max(4, window // 2)).corr(b)

    df = pd.DataFrame(out, index=ret_DE.index)
    return df


def cross_maturity_matrix(px_DE: pd.DataFrame,
                           px_FR: pd.DataFrame) -> pd.DataFrame:
    """8x8 correlation matrix of log returns across all DE and FR maturities."""
    px = pd.concat([px_DE, px_FR], axis=1)
    ret = np.log(px.where(px > 0)).diff().replace([np.inf, -np.inf], np.nan)
    return ret.corr(method="pearson")


def matched_correlations_residual(res_DE: pd.DataFrame, res_FR: pd.DataFrame,
                                    labels: list) -> pd.DataFrame:
    """Per-maturity correlation on deseasonalised residuals: levels (direct)
    and first differences (innovation view)."""
    rows = []
    d_DE = res_DE.diff()
    d_FR = res_FR.diff()

    for lab in labels:
        col_de = f"DE_{lab}"
        col_fr = f"FR_{lab}"

        lvl = pd.concat([res_DE[col_de], res_FR[col_fr]], axis=1).dropna()
        diff = pd.concat([d_DE[col_de], d_FR[col_fr]], axis=1).dropna()

        rows.append({
            "maturity":            lab,
            "n_levels":            int(len(lvl)),
            "n_diffs":             int(len(diff)),
            "pearson_resid_level": float(lvl.corr().iloc[0, 1]),
            "pearson_resid_diff":  float(diff.corr(method="pearson").iloc[0, 1]),
            "spearman_resid_diff": float(diff.corr(method="spearman").iloc[0, 1]),
            "std_resid_DE":        float(res_DE[col_de].std()),
            "std_resid_FR":        float(res_FR[col_fr].std()),
        })

    return pd.DataFrame(rows)


def rolling_correlations_residual(res_DE: pd.DataFrame, res_FR: pd.DataFrame,
                                     labels: list, window: int) -> pd.DataFrame:
    """Rolling Pearson correlation of residual first differences, one column
    per matched maturity."""
    d_DE = res_DE.diff()
    d_FR = res_FR.diff()

    out = {}
    for lab in labels:
        a = d_DE[f"DE_{lab}"]
        b = d_FR[f"FR_{lab}"]
        out[f"rho_{lab}"] = a.rolling(window,
                                       min_periods=max(4, window // 2)).corr(b)
    return pd.DataFrame(out, index=d_DE.index)


def cross_maturity_matrix_residual(res_DE: pd.DataFrame,
                                     res_FR: pd.DataFrame) -> pd.DataFrame:
    """8x8 correlation matrix on deseasonalised residual levels."""
    res = pd.concat([res_DE, res_FR], axis=1)
    return res.corr(method="pearson")


def maybe_plot_rolling(roll: pd.DataFrame, out_path: str,
                        ylabel: str = "Rolling Pearson correlation (log returns)",
                        title:  str = "DE vs FR power futures -- rolling correlation by maturity"
                        ) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (matplotlib not available, skipping plot: {exc})")
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in roll.columns:
        ax.plot(roll.index, roll[col], label=col, linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=12,
                        help="Rolling-correlation window in samples "
                             "(weekly cadence -> 12 ~ 3 months). Default 12.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not write corr_DE_FR_rolling.png.")
    args = parser.parse_args()

    print(f"Loading joint DE/FR Stage B panel "
          f"({list(btm.SUBSET_LABELS)} per market) ...")
    px_DE, px_FR, labels = load_joint_panel()
    print(f"  panel: {len(px_DE)} weekly observations "
          f"{px_DE.index.min().date()} -> {px_DE.index.max().date()}")
    print(f"  DE columns: {list(px_DE.columns)}")
    print(f"  FR columns: {list(px_FR.columns)}")

    print("\n=== 1) Matched-maturity correlations ===")
    matched = matched_correlations(px_DE, px_FR, labels)
    matched_path = os.path.join(OUT_DIR, "corr_DE_FR_matched.csv")
    matched.to_csv(matched_path, index=False)
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(matched.to_string(index=False))
    print(f"Saved {matched_path}")

    print(f"\n=== 2) Rolling correlations (window={args.window} samples) ===")
    roll = rolling_correlations(px_DE, px_FR, labels, window=args.window)
    roll_path = os.path.join(OUT_DIR, "corr_DE_FR_rolling.csv")
    roll.to_csv(roll_path)
    last_obs = roll.dropna(how="all").tail(5)
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print("Last few rows:")
        print(last_obs.to_string())
    print(f"  mean over sample: "
          f"{ {c: float(roll[c].mean()) for c in roll.columns} }")
    print(f"Saved {roll_path}")

    if not args.no_plot:
        plot_path = os.path.join(OUT_DIR, "corr_DE_FR_rolling.png")
        if maybe_plot_rolling(roll, plot_path):
            print(f"Saved {plot_path}")

    print("\n=== 3) Cross-maturity correlation matrix (log returns) ===")
    cross = cross_maturity_matrix(px_DE, px_FR)
    cross_path = os.path.join(OUT_DIR, "corr_DE_FR_crossmaturity.csv")
    cross.to_csv(cross_path)
    with pd.option_context("display.float_format", "{:+.3f}".format):
        print(cross.to_string())
    print(f"Saved {cross_path}")

    # ------------------------------------------------------------------
    # Deseasonalised pipeline
    # ------------------------------------------------------------------
    print("\n=== 4) Building deseasonalised residuals (Stage A fit -> Stage B) ===")
    res_DE, res_FR, _labels, info = load_deseasonalised_residuals()
    print(f"  annual_h chosen: DE={info['annual_h_DE']}, "
          f"FR={info['annual_h_FR']}")
    print(f"  price scale:     DE={info['price_scale_DE']:.4f} EUR/MWh, "
          f"FR={info['price_scale_FR']:.4f} EUR/MWh")
    print(f"  residual DE: mean={res_DE.values.mean():+.5f}  "
          f"std={res_DE.values.std():.5f}")
    print(f"  residual FR: mean={res_FR.values.mean():+.5f}  "
          f"std={res_FR.values.std():.5f}")

    print("\n=== 5) Matched-maturity correlations on residuals ===")
    matched_res = matched_correlations_residual(res_DE, res_FR, labels)
    matched_res_path = os.path.join(OUT_DIR,
                                      "corr_DE_FR_matched_deseasonalised.csv")
    matched_res.to_csv(matched_res_path, index=False)
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(matched_res.to_string(index=False))
    print(f"Saved {matched_res_path}")

    # Side-by-side raw vs. deseasonalised summary.
    summary = matched[["maturity", "pearson_logret", "spearman_logret"]] \
        .merge(matched_res[["maturity", "pearson_resid_level",
                              "pearson_resid_diff", "spearman_resid_diff"]],
                on="maturity")
    summary = summary.rename(columns={
        "pearson_logret":       "raw_pearson_logret",
        "spearman_logret":      "raw_spearman_logret",
        "pearson_resid_level":  "deseas_pearson_level",
        "pearson_resid_diff":   "deseas_pearson_diff",
        "spearman_resid_diff":  "deseas_spearman_diff",
    })
    summary_path = os.path.join(OUT_DIR, "corr_DE_FR_matched_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\nSide-by-side raw vs. deseasonalised:")
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(summary.to_string(index=False))
    print(f"Saved {summary_path}")

    print(f"\n=== 6) Rolling correlations on residual first differences "
          f"(window={args.window} samples) ===")
    roll_res = rolling_correlations_residual(res_DE, res_FR, labels,
                                                window=args.window)
    roll_res_path = os.path.join(OUT_DIR,
                                   "corr_DE_FR_rolling_deseasonalised.csv")
    roll_res.to_csv(roll_res_path)
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print("Last few rows:")
        print(roll_res.dropna(how="all").tail(5).to_string())
    print(f"  mean over sample: "
          f"{ {c: float(roll_res[c].mean()) for c in roll_res.columns} }")
    print(f"Saved {roll_res_path}")

    if not args.no_plot:
        plot_path = os.path.join(OUT_DIR,
                                   "corr_DE_FR_rolling_deseasonalised.png")
        if maybe_plot_rolling(
                roll_res, plot_path,
                ylabel="Rolling Pearson (residual first differences)",
                title="DE vs FR -- rolling correlation, "
                       "deseasonalised residuals"):
            print(f"Saved {plot_path}")

    print("\n=== 7) Cross-maturity correlation matrix (residual levels) ===")
    cross_res = cross_maturity_matrix_residual(res_DE, res_FR)
    cross_res_path = os.path.join(OUT_DIR,
                                    "corr_DE_FR_crossmaturity_deseasonalised.csv")
    cross_res.to_csv(cross_res_path)
    with pd.option_context("display.float_format", "{:+.3f}".format):
        print(cross_res.to_string())
    print(f"Saved {cross_res_path}")


if __name__ == "__main__":
    main()
