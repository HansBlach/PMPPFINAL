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

#PCA without removing roll over dates
m = monthly.drop(columns = ["Trading Day"])
q = quarterly.drop(columns = ["Trading Day"])
y = yearly.drop(columns = ["Trading Day"])
PCA_matrix = pd.concat([m,q,y], axis = 1).reset_index(drop=True)
print(PCA_matrix.head())

print(PCA_matrix.isnull().sum())  
print(PCA_matrix.isnull().sum(axis=1))
PCA_matrix = (PCA_matrix
     .ffill()                
     .interpolate(axis=1)    
     .diff()                 
     .dropna()               
)

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
#plt.show()