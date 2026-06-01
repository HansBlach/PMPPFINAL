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
start_list_m, start_list_q, start_list_y = [], [], []  # days from Trading Day to Delivery Start, per year
dur_list_m,   dur_list_q,   dur_list_y   = [], [], []  # days from Delivery Start to Delivery End, per year

for i in range(2022, 2026):
    path = f"/Users/hansblachfalkenberg/Desktop/Unimat-4/speciale/DE/PowerFutureHistory_Phelix-DE_{i}.xlsx"
    DEBM = gd.get_data(path, "DEBM")          # reads the monthly sheet from the Excel file into a DataFrame
    DEBQ = gd.get_data(path, "DEBQ")          # reads the quarterly sheet from the Excel file into a DataFrame
    DEBY = gd.get_data(path, "DEBY")          # reads the yearly sheet from the Excel file into a DataFrame
    m, q, y = gd.build_settlement_matrix(DEBM, DEBQ, DEBY, n_monthly=3, n_quarterly=4, n_yearly=3)
    # ^ pivots the three raw sheets into wide matrices: one column per maturity offset,
    #   one row per Trading Day (3 monthly, 4 quarterly, 3 yearly maturities)
    monthly_list.append(m)
    quarterly_list.append(q)
    yearly_list.append(y)

    (ms, qs, ys), (md, qd, yd) = gd.build_date_matrices(DEBM, DEBQ, DEBY, n_monthly=3, n_quarterly=4, n_yearly=3)
    # ^ same pivot structure but values are calendar-day differences instead of settlement prices:
    #   (ms/qs/ys) = days from Trading Day to Delivery Start for each maturity
    #   (md/qd/yd) = days from Delivery Start to Delivery End for each maturity
    start_list_m.append(ms); start_list_q.append(qs); start_list_y.append(ys)
    dur_list_m.append(md);   dur_list_q.append(qd);   dur_list_y.append(yd)

monthly  = pd.concat(monthly_list,  axis=0).reset_index(drop=True)  # stacks the per-year monthly matrices into one continuous DataFrame
quarterly = pd.concat(quarterly_list, axis=0).reset_index(drop=True) # stacks the per-year quarterly matrices into one continuous DataFrame
yearly   = pd.concat(yearly_list,   axis=0).reset_index(drop=True)  # stacks the per-year yearly matrices into one continuous DataFrame

start_monthly   = pd.concat(start_list_m, axis=0).reset_index(drop=True)  # stacks per-year days-to-start monthly matrices
start_quarterly = pd.concat(start_list_q, axis=0).reset_index(drop=True)  # stacks per-year days-to-start quarterly matrices
start_yearly    = pd.concat(start_list_y, axis=0).reset_index(drop=True)  # stacks per-year days-to-start yearly matrices

dur_monthly   = pd.concat(dur_list_m, axis=0).reset_index(drop=True)  # stacks per-year duration monthly matrices
dur_quarterly = pd.concat(dur_list_q, axis=0).reset_index(drop=True)  # stacks per-year duration quarterly matrices
dur_yearly    = pd.concat(dur_list_y, axis=0).reset_index(drop=True)  # stacks per-year duration yearly matrices


# PCA with no roll over dates
def diff_no_rollover(df):
    """
    Fill missing prices, diff, then remove the first trading day of every
    month (roll-over) so monthly, quarterly, and yearly stay the same length.
    """
    df = df.sort_values("Trading Day").reset_index(drop=True)  # sorts rows chronologically by Trading Day
    price_cols = [c for c in df.columns if c != "Trading Day"]
    is_roll = df["Trading Day"].dt.month != df["Trading Day"].dt.month.shift(1)  # shift(1) compares each row's month to the previous row's month; True on the first trading day of a new month (the roll-over day)
    df[price_cols] = (df[price_cols]
                      .ffill()            # forward-fills any missing prices using the last known value
                      .interpolate(axis=1)  # fills any remaining NaNs by linearly interpolating across maturities (columns) within each row
                      .diff())            # takes the day-over-day price difference (today minus yesterday) for each maturity
    df.loc[is_roll, price_cols] = np.nan  # sets roll-over day price changes to NaN so the artificial jump from contract expiry is excluded
    return df.dropna().reset_index(drop=True)  # drops all rows that contain NaN (roll-over days and the first row where diff() has no prior value)

monthly_diff   = diff_no_rollover(monthly)
quarterly_diff = diff_no_rollover(quarterly)
yearly_diff    = diff_no_rollover(yearly)

m = monthly_diff.drop(columns = ["Trading Day"])   # removes the Trading Day column, keeping only the price-change columns
q = quarterly_diff.drop(columns = ["Trading Day"])  # removes the Trading Day column, keeping only the price-change columns
y = yearly_diff.drop(columns = ["Trading Day"])    # removes the Trading Day column, keeping only the price-change columns

# PCA_matrix is a (n_days × 10) DataFrame of daily price changes across all 10 maturities.
# Each row is one trading day; each column is one maturity (1MAH, 2MAH, 3MAH, 1QAH, 2QAH, 3QAH, 4QAH, 1YAH, 2YAH, 3YAH).
# Roll-over days are excluded so every change reflects genuine market moves, not contract expiry jumps.
# This matrix is the input to PCA, which will decompose it into orthogonal factors (level, slope, curvature).
PCA_matrix = pd.concat([m,q,y], axis = 1).reset_index(drop=True)  # concatenates the three maturity groups side-by-side into one wide matrix

def remove_rollover(df):
    """
    Remove roll-over days (first trading day of each new month) without differencing.
    Used for the date-difference matrices where the raw values are meaningful as-is.
    """
    df = df.sort_values("Trading Day").reset_index(drop=True)  # sorts rows chronologically
    value_cols = [c for c in df.columns if c != "Trading Day"]
    is_roll = df["Trading Day"].dt.month != df["Trading Day"].dt.month.shift(1)  # True on the first day of each new month
    df.loc[is_roll, value_cols] = np.nan  # marks roll-over days as NaN so they are excluded
    return df.dropna().reset_index(drop=True)  # drops roll-over rows and any rows with missing values

# --- Build days-to-start matrix ---
# days_to_start_matrix is a (n_days × 10) DataFrame of raw calendar days from Trading Day to Delivery Start.
# Each column is one maturity (1MAH … 3YAH); roll-over days are excluded to keep the index aligned with PCA_matrix.
# The value tells you how far ahead each contract's delivery window begins on any given trading day.
start_monthly_clean   = remove_rollover(start_monthly)
start_quarterly_clean = remove_rollover(start_quarterly)
start_yearly_clean    = remove_rollover(start_yearly)

sm = start_monthly_clean.drop(columns=["Trading Day"])    # days-to-start for monthly maturities
sq = start_quarterly_clean.drop(columns=["Trading Day"])  # days-to-start for quarterly maturities
sy = start_yearly_clean.drop(columns=["Trading Day"])     # days-to-start for yearly maturities

maturity_matrix = pd.concat([sm, sq, sy], axis=1).reset_index(drop=True)  # wide matrix: one column per maturity

# --- Build duration matrix ---
# duration_matrix is a (n_days × 10) DataFrame of raw calendar days from Delivery Start to Delivery End.
# Each column is one maturity; roll-over days are excluded to keep the index aligned with PCA_matrix.
# The value tells you how long each contract's delivery window lasts (~30 days monthly, ~90 quarterly, ~365 yearly).
dur_monthly_clean   = remove_rollover(dur_monthly)
dur_quarterly_clean = remove_rollover(dur_quarterly)
dur_yearly_clean    = remove_rollover(dur_yearly)

dm = dur_monthly_clean.drop(columns=["Trading Day"])    # delivery duration for monthly maturities
dq = dur_quarterly_clean.drop(columns=["Trading Day"])  # delivery duration for quarterly maturities
dy = dur_yearly_clean.drop(columns=["Trading Day"])     # delivery duration for yearly maturities

delivery_period_matrix = pd.concat([dm, dq, dy], axis=1).reset_index(drop=True)  # wide matrix: one column per maturity

print(PCA_matrix)
# --- 3. Standardize ---
scaler = StandardScaler()                    # creates a scaler that will subtract the mean and divide by std dev for each column
X_scaled = scaler.fit_transform(PCA_matrix) # fits the scaler to PCA_matrix (computes mean and std) and transforms it, so every maturity has mean 0 and std 1

# --- 4. Fit PCA ---
pca = PCA()                          # initialises PCA; no n_components specified so all 10 components are kept
X_pca = pca.fit_transform(X_scaled) # fits PCA to the scaled data (finds the principal components) and projects the data onto them

# --- 5. Explained variance ---
ev = pca.explained_variance_ratio_   # array of length 10: the fraction of total variance explained by each principal component
print("Explained variance per component:")
for i, v in enumerate(ev):
    print(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {np.cumsum(ev)[i]*100:.1f}%)")
    # np.cumsum(ev) computes the running total of explained variance so we can see how many PCs are needed to capture most of the variance

print(pca.components_[0].round(3))          # prints the loadings of PC1 (the "level" factor) rounded to 3 decimal places
print(PCA_matrix[['3MAH', '1QAH']].corr()) # computes the Pearson correlation matrix between the 3MAH and 1QAH price-change series
maturities = ["1MAH", "2MAH", "3MAH", "1QAH", "2QAH", "3QAH", "4QAH", "1YAH", "2YAH", "3YAH"]
# Loadings: how each maturity contributes to each PC
loadings = pd.DataFrame(
    pca.components_,                          # shape: (10, 10) — each row is one PC, each column is one maturity's loading
    columns=maturities,
    index=[f'PC{i+1}' for i in range(10)]
)

# Plot the first 3 PCs — classic "level, slope, curvature"
fig, ax = plt.subplots(figsize=(9, 5))  # creates a figure and a single set of axes with the given size (inches)
for i in range(3):
    ax.plot(maturities, pca.components_[i], marker='o', label=f'PC{i+1} ({ev[i]*100:.1f}%)')
    # plots the loadings of PC i+1 across maturities; marker='o' draws a dot at each maturity point

ax.axhline(0, color='black', linewidth=0.8)  # draws a horizontal reference line at y=0
ax.set_title('PCA Loadings — Futures Curve')
ax.set_xlabel('Maturity')
ax.set_ylabel('Loading')
ax.legend()         # displays the legend with PC labels and explained variance percentages
plt.grid(True)      # overlays a grid on the plot for readability
plt.show()          # renders and displays the plot

#Do time series
pca = PCA(n_components=3)                        # initialises PCA keeping only the top 3 components (level, slope, curvature)
principal_factors = pca.fit_transform(X_scaled)  # fits PCA and projects the data; result shape is (n_days, 3)

# Each column is one principal factor time series
pca = PCA(n_components=3)                        # re-initialises PCA with 3 components (same as above — duplicate fit)
principal_factors = pca.fit_transform(X_scaled)  # fits and transforms again; result shape is (n_days, 3)

# Each column is one principal factor time series
pf = pd.DataFrame(
    principal_factors,
    index=PCA_matrix.index,    # uses the same row index as PCA_matrix so dates are preserved
    columns=['PF1', 'PF2', 'PF3']
)

dates = pf.index  # extracts the index (trading day integers) to use as the x-axis in the plots

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
# creates a figure with 3 stacked subplots sharing the same x-axis; sharex=True keeps zoom/pan in sync

for i, ax in enumerate(axes):
    ax.plot(dates, pf[f'PF{i+1}'], linewidth=0.8)   # plots the time series of principal factor i+1
    ax.axhline(0, color='black', linewidth=0.8)       # draws a horizontal reference line at y=0
    ax.set_ylabel(f'PF{i+1}')
    ax.grid(True)   # overlays a grid for readability

    # Add explained variance in title
    ev = pca.explained_variance_ratio_[i] * 100  # fraction of variance explained by this component, converted to percentage
    labels = ['Level', 'Slope', 'Prompt vs Deferred']
    ax.set_title(f'PF{i+1} — {labels[i]} ({ev:.1f}%)')

axes[-1].set_xlabel('Date')
fig.suptitle('Principal Factors — Futures Curve', fontsize=13)  # adds a centred title above all subplots
plt.tight_layout()  # automatically adjusts subplot spacing so labels and titles do not overlap
plt.show()          # renders and displays the figure

print(PCA_matrix.diff().describe())          # computes second-order differences (diff of diff) and prints summary statistics (mean, std, min, max, etc.)
print((PCA_matrix.diff() == 0).sum())        # count of zero changes per maturity
print(PCA_matrix.isnull().sum())             # NaN counts
print(PCA_matrix)
