import os
import sys

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

import pandas as pd
import numpy as np
import GetData as gd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

monthly_list, quarterly_list, yearly_list = [], [], []

for i in range(2022, 2026):
    path = f"/Users/hansblachfalkenberg/Desktop/Unimat-4/speciale/DE/PowerFutureHistory_Phelix-DE_{i}.xlsx"
    DEBM = gd.get_data(path, "DEBM")
    DEBQ = gd.get_data(path, "DEBQ")
    DEBY = gd.get_data(path, "DEBY")
    m, q, y = gd.build_settlement_matrix(DEBM, DEBQ, DEBY, n_monthly=3, n_quarterly=4, n_yearly=3)
    monthly_list.append(m)
    quarterly_list.append(q)
    yearly_list.append(y)

monthly  = pd.concat(monthly_list,  axis=0).reset_index(drop=True)
quarterly = pd.concat(quarterly_list, axis=0).reset_index(drop=True)
yearly   = pd.concat(yearly_list,   axis=0).reset_index(drop=True)


# PCA with no roll over dates and pct
def diff_no_rollover(df):
    """
    Fill missing prices, diff, then remove the first trading day of every
    month (roll-over) so monthly, quarterly, and yearly stay the same length.
    """
    df = df.sort_values("Trading Day").reset_index(drop=True)
    price_cols = [c for c in df.columns if c != "Trading Day"]
    is_roll = df["Trading Day"].dt.month != df["Trading Day"].dt.month.shift(1)
    df[price_cols] = (df[price_cols]
                      .ffill()
                      .interpolate(axis=1))
    df[price_cols] = np.log(df[price_cols] / df[price_cols].shift(1))
    df.loc[is_roll, price_cols] = np.nan
    return df.dropna().reset_index(drop=True)

monthly_diff   = diff_no_rollover(monthly)
quarterly_diff = diff_no_rollover(quarterly)
yearly_diff    = diff_no_rollover(yearly)

m = monthly_diff.drop(columns = ["Trading Day"])
q = quarterly_diff.drop(columns = ["Trading Day"])
y = yearly_diff.drop(columns = ["Trading Day"])
PCA_matrix = pd.concat([m,q,y], axis = 1).reset_index(drop=True)

# --- 3. Standardize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(PCA_matrix)

# --- 4. Fit PCA ---
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# --- 5. Explained variance ---
ev = pca.explained_variance_ratio_
print("Explained variance per component:")
for i, v in enumerate(ev):
    print(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {np.cumsum(ev)[i]*100:.1f}%)")

print(pca.components_[0].round(3))
print(PCA_matrix[['3MAH', '1QAH']].corr()) 
maturities = ["1MAH", "2MAH", "3MAH", "1QAH", "2QAH", "3QAH", "4QAH", "1YAH", "2YAH", "3YAH"]
# Loadings: how each maturity contributes to each PC
loadings = pd.DataFrame(
    pca.components_,                          # shape: (10, 10)
    columns=maturities,
    index=[f'PC{i+1}' for i in range(10)]
)

# Plot the first 3 PCs — classic "level, slope, curvature"
fig, ax = plt.subplots(figsize=(9, 5))
for i in range(3):
    ax.plot(maturities, pca.components_[i], marker='o', label=f'PC{i+1} ({ev[i]*100:.1f}%)')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('PCA Loadings — Futures Curve')
ax.set_xlabel('Maturity')
ax.set_ylabel('Loading')
ax.legend()
plt.grid(True)
plt.show()

#Do time series
pca = PCA(n_components=3)
principal_factors = pca.fit_transform(X_scaled)  # shape: (n_days, 3)

# Each column is one principal factor time series
pca = PCA(n_components=3)
principal_factors = pca.fit_transform(X_scaled)  # shape: (n_days, 3)

# Each column is one principal factor time series
pf = pd.DataFrame(
    principal_factors,
    index=PCA_matrix.index,   # carries over the index from your cleaned data
    columns=['PF1', 'PF2', 'PF3']
)

dates = pf.index

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(dates, pf[f'PF{i+1}'], linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel(f'PF{i+1}')
    ax.grid(True)
    
    # Add explained variance in title
    ev = pca.explained_variance_ratio_[i] * 100
    labels = ['Level', 'Slope', 'Prompt vs Deferred']
    ax.set_title(f'PF{i+1} — {labels[i]} ({ev:.1f}%)')

axes[-1].set_xlabel('Date')
fig.suptitle('Principal Factors — Futures Curve', fontsize=13)
plt.tight_layout()
plt.show()

print(PCA_matrix.diff().describe())
print((PCA_matrix.diff() == 0).sum())        # count of zero changes per maturity
print(PCA_matrix.isnull().sum())             # NaN counts
print(PCA_matrix)

