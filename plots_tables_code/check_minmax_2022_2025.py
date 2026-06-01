"""
Quick min/max scan of the 1MAH-4MAH observed Stage B panel between 2022 and 2025.

Reuses the BIC_monthly_OU loader so we get exactly the same data the EKF sees
(after STAGE_B_INCLUDE / START_DATE / thinning are applied, modulo we skip
the thinning here to scan every raw trading day).

    python check_minmax_2022_2025.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

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
import BIC_monthly_OU as drv


LABELS = ("1WAH", "2WAH", "3WAH", "4WAH")


def main():
    # Bypass drv.START_DATE so 2022 data is included. load_stage_b_data
    # otherwise truncates whatever range you pass in to [START_DATE, end].
    _prev_start = drv.START_DATE
    drv.START_DATE = None
    try:
        y, mat, dlt, tra = drv.load_stage_b_data()
    finally:
        drv.START_DATE = _prev_start
    # trading[:, 0] is decimal-year; restrict to [2022, 2026).
    t = tra[:, 0]
    mask = (t >= 2022.0) & (t < 2026.0)
    y = y[mask]
    t = t[mask]

    panel_labels = list(drv.SUBSET_LABELS)
    print(f"Stage B panel: {panel_labels}")
    print(f"Window: 2022-01-01 .. 2025-12-31   ({len(t)} trading days)")
    print(f"Date range covered: {t.min():.4f} .. {t.max():.4f}\n")

    print(f"{'contract':>8s}  {'min EUR/MWh':>12s}  {'max EUR/MWh':>12s}  "
          f"{'mean':>10s}  {'std':>10s}  {'n_valid':>8s}")
    for c, name in enumerate(panel_labels):
        if name not in LABELS:
            continue
        col = y[:, c]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            print(f"{name:>8s}  {'(no data)':>12s}")
            continue
        print(f"{name:>8s}  {valid.min():>12.3f}  {valid.max():>12.3f}  "
              f"{valid.mean():>10.3f}  {valid.std():>10.3f}  {len(valid):>8d}")

    print()
    pooled = y[:, [panel_labels.index(l) for l in LABELS if l in panel_labels]]
    pooled = pooled[~np.isnan(pooled)]
    print(f"Pooled across 1-4MAH:  min={pooled.min():.3f}   "
          f"max={pooled.max():.3f}   mean={pooled.mean():.3f}   "
          f"std={pooled.std():.3f}   n={len(pooled)}")


if __name__ == "__main__":
    main()
