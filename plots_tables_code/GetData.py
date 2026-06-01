import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def to_decimal_year(dates):
    """Convert dates to decimal year: year + day_of_year / 365."""
    dates = pd.to_datetime(dates)
    return dates.dt.year + (dates.dt.dayofyear - 1) / 365


def filter_to_first_trading_day_per_iso_week(df, trading_day_col="Trading Day"):
    df = df.copy()
    df[trading_day_col] = pd.to_datetime(df[trading_day_col])
    iso = df[trading_day_col].dt.isocalendar()
    df["_iso_year"] = iso["year"].astype(int)
    df["_iso_week"] = iso["week"].astype(int)
    df = (
        df.sort_values(trading_day_col)
          .groupby(["_iso_year", "_iso_week"], as_index=False, sort=True)
          .first()
          .drop(columns=["_iso_year", "_iso_week"])
          .reset_index(drop=True)
    )
    return df


def get_data(file_path, sheet_name=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
    return df


def filter_monthly_maturity(df, offset):
    """
    Filter rows where Delivery Start is `offset` months ahead of Trading Day.

    offset=1 → front month (1MAH)
    offset=2 → second month ahead (2MAH)
    ...etc
    """
    df = df.copy()
    df["Trading Day"] = pd.to_datetime(df["Trading Day"])
    df["Delivery Start"] = pd.to_datetime(df["Delivery Start"])

    month_offset = (
        (df["Delivery Start"].dt.year - df["Trading Day"].dt.year) * 12
        + (df["Delivery Start"].dt.month - df["Trading Day"].dt.month)
    )

    return df[month_offset == offset].reset_index(drop=True)


def filter_quarterly_maturity(df, offset):
    """
    Filter rows where Delivery Start is `offset` quarters ahead of Trading Day.

    """
    df = df.copy()
    df["Trading Day"] = pd.to_datetime(df["Trading Day"])
    df["Delivery Start"] = pd.to_datetime(df["Delivery Start"])

    trading_quarter = (df["Trading Day"].dt.month - 1) // 3 + 1
    delivery_quarter = (df["Delivery Start"].dt.month - 1) // 3 + 1

    quarter_offset = (
        (df["Delivery Start"].dt.year - df["Trading Day"].dt.year) * 4
        + (delivery_quarter - trading_quarter)
    )

    return df[quarter_offset == offset].reset_index(drop=True)


def filter_yearly_maturity(df, offset):
    """
    Filter rows where Delivery Start is `offset` years ahead of Trading Day.
    """
    df = df.copy()
    df["Trading Day"] = pd.to_datetime(df["Trading Day"])
    df["Delivery Start"] = pd.to_datetime(df["Delivery Start"])

    year_offset = df["Delivery Start"].dt.year - df["Trading Day"].dt.year

    return df[year_offset == offset].reset_index(drop=True)

def build_settlement_matrix(monthly_df, quarterly_df, yearly_df,
                            n_monthly=None, n_quarterly=None, n_yearly=None):
    """
    Build a settlement price matrix for each contract type.

    """

    def _pivot(df, offset_series, label_suffix, n_offsets):
        df = df.copy()
        df["Trading Day"] = pd.to_datetime(df["Trading Day"])
        df["_offset"] = offset_series
        df = df[df["_offset"] >= 1]
        if n_offsets is not None:
            df = df[df["_offset"] <= n_offsets]
        matrix = (
            df.pivot_table(
                index="Trading Day",
                columns="_offset",
                values="Settlement Price",
                aggfunc="mean",
            )
            .rename(columns=lambda n: f"{int(n)}{label_suffix}")
            .reset_index()
        )
        matrix.columns.name = None
        return matrix

    # Monthly: offset in calendar months
    m = monthly_df.copy()
    m["Trading Day"] = pd.to_datetime(m["Trading Day"])
    m["Delivery Start"] = pd.to_datetime(m["Delivery Start"])
    month_offset = (
        (m["Delivery Start"].dt.year - m["Trading Day"].dt.year) * 12
        + (m["Delivery Start"].dt.month - m["Trading Day"].dt.month)
    )
    monthly_matrix = _pivot(m, month_offset, "MAH", n_monthly)

    # Quarterly: offset in quarters
    q = quarterly_df.copy()
    q["Trading Day"] = pd.to_datetime(q["Trading Day"])
    q["Delivery Start"] = pd.to_datetime(q["Delivery Start"])
    trading_qtr = (q["Trading Day"].dt.month - 1) // 3 + 1
    delivery_qtr = (q["Delivery Start"].dt.month - 1) // 3 + 1
    quarter_offset = (
        (q["Delivery Start"].dt.year - q["Trading Day"].dt.year) * 4
        + (delivery_qtr - trading_qtr)
    )
    quarterly_matrix = _pivot(q, quarter_offset, "QAH", n_quarterly)

    # Yearly: offset in years
    y = yearly_df.copy()
    y["Trading Day"] = pd.to_datetime(y["Trading Day"])
    y["Delivery Start"] = pd.to_datetime(y["Delivery Start"])
    year_offset = y["Delivery Start"].dt.year - y["Trading Day"].dt.year
    yearly_matrix = _pivot(y, year_offset, "YAH", n_yearly)

    return monthly_matrix, quarterly_matrix, yearly_matrix


def build_weekly_settlement_matrix(weekly_df, n_weekly=4):
    """
    Build a settlement price matrix for weekly contracts from the DEB1-5 sheet.

    """
    w = weekly_df.copy()
    w["Trading Day"]    = pd.to_datetime(w["Trading Day"])
    w["Delivery Start"] = pd.to_datetime(w["Delivery Start"])

    trading_monday  = w["Trading Day"]    - pd.to_timedelta(w["Trading Day"].dt.weekday,    unit="D")
    delivery_monday = w["Delivery Start"] - pd.to_timedelta(w["Delivery Start"].dt.weekday, unit="D")
    week_offset = ((delivery_monday - trading_monday).dt.days // 7).astype(int)

    w["_offset"] = week_offset
    w = w[w["_offset"] >= 1]
    if n_weekly is not None:
        w = w[w["_offset"] <= n_weekly]

    matrix = (
        w.pivot_table(
            index="Trading Day",
            columns="_offset",
            values="Settlement Price",
            aggfunc="mean",
        )
        .rename(columns=lambda n: f"{int(n)}WAH")
        .reset_index()
    )
    matrix.columns.name = None
    return matrix


def build_weekly_date_matrices(weekly_df, n_weekly=4):
    """
    Build maturity (days to delivery start), duration (delivery window length),
    and decimal-year matrices for weekly contracts from the DEB1-5 sheet.

    """
    w = weekly_df.copy()
    w["Trading Day"]    = pd.to_datetime(w["Trading Day"])
    w["Delivery Start"] = pd.to_datetime(w["Delivery Start"])
    w["Delivery End"]   = pd.to_datetime(w["Delivery End"])

    trading_monday  = w["Trading Day"]    - pd.to_timedelta(w["Trading Day"].dt.weekday,    unit="D")
    delivery_monday = w["Delivery Start"] - pd.to_timedelta(w["Delivery Start"].dt.weekday, unit="D")
    week_offset = ((delivery_monday - trading_monday).dt.days // 7).astype(int)

    w["_offset"]      = week_offset
    w["days_to_start"] = (w["Delivery Start"] - w["Trading Day"]).dt.days
    w["duration"]      = (w["Delivery End"]   - w["Delivery Start"]).dt.days
    w["decimal_year"]  = to_decimal_year(w["Trading Day"]).values

    w = w[w["_offset"] >= 1]
    if n_weekly is not None:
        w = w[w["_offset"] <= n_weekly]

    def _piv(col, suffix):
        mat = (
            w.pivot_table(
                index="Trading Day",
                columns="_offset",
                values=col,
                aggfunc="mean",
            )
            .rename(columns=lambda n: f"{int(n)}{suffix}")
            .reset_index()
        )
        mat.columns.name = None
        return mat

    start_mat   = _piv("days_to_start", "WAH")
    dur_mat     = _piv("duration",      "WAH")
    trading_mat = _piv("decimal_year",  "WAH")
    return start_mat, dur_mat, trading_mat


def build_date_matrices(monthly_df, quarterly_df, yearly_df,
                        n_monthly=None, n_quarterly=None, n_yearly=None):
    """
    Build two sets of date-difference matrices with the same structure as
    build_settlement_matrix (Trading Day index, one column per maturity offset).

    Matrix 1 — days_to_start

    Matrix 2 — duration
    """

    def _pivot_date(df, offset_series, label_suffix, n_offsets, value_col):
        # identical pivot logic to build_settlement_matrix but for a date column
        df = df.copy()
        df["_offset"] = offset_series
        df = df[df["_offset"] >= 1]
        if n_offsets is not None:
            df = df[df["_offset"] <= n_offsets]
        matrix = (
            df.pivot_table(
                index="Trading Day",
                columns="_offset",
                values=value_col,
                aggfunc="mean",
            )
            .rename(columns=lambda n: f"{int(n)}{label_suffix}")
            .reset_index()
        )
        matrix.columns.name = None
        return matrix

    # --- Monthly ---
    m = monthly_df.copy()
    m["Trading Day"]    = pd.to_datetime(m["Trading Day"])
    m["Delivery Start"] = pd.to_datetime(m["Delivery Start"])
    m["Delivery End"]   = pd.to_datetime(m["Delivery End"])
    month_offset = (
        (m["Delivery Start"].dt.year  - m["Trading Day"].dt.year)  * 12
        + (m["Delivery Start"].dt.month - m["Trading Day"].dt.month)
    )
    m["days_to_start"] = (m["Delivery Start"] - m["Trading Day"]).dt.days
    m["duration"]      = (m["Delivery End"]   - m["Delivery Start"]).dt.days
    m["decimal_year"]  = to_decimal_year(m["Trading Day"]).values
    m_start = _pivot_date(m, month_offset, "MAH", n_monthly, "days_to_start")
    m_dur   = _pivot_date(m, month_offset, "MAH", n_monthly, "duration")
    m_tra   = _pivot_date(m, month_offset, "MAH", n_monthly, "decimal_year")



    # --- Quarterly ---
    q = quarterly_df.copy()
    q["Trading Day"]    = pd.to_datetime(q["Trading Day"])
    q["Delivery Start"] = pd.to_datetime(q["Delivery Start"])
    q["Delivery End"]   = pd.to_datetime(q["Delivery End"])
    trading_qtr  = (q["Trading Day"].dt.month    - 1) // 3 + 1
    delivery_qtr = (q["Delivery Start"].dt.month - 1) // 3 + 1
    quarter_offset = (
        (q["Delivery Start"].dt.year - q["Trading Day"].dt.year) * 4
        + (delivery_qtr - trading_qtr)
    )
    q["days_to_start"] = (q["Delivery Start"] - q["Trading Day"]).dt.days
    q["duration"]      = (q["Delivery End"]   - q["Delivery Start"]).dt.days
    q["decimal_year"]  = to_decimal_year(q["Trading Day"]).values
    q_start = _pivot_date(q, quarter_offset, "QAH", n_quarterly, "days_to_start")
    q_dur   = _pivot_date(q, quarter_offset, "QAH", n_quarterly, "duration")
    q_tra   = _pivot_date(q, quarter_offset, "QAH", n_quarterly, "decimal_year")

    # --- Yearly ---
    y = yearly_df.copy()
    y["Trading Day"]    = pd.to_datetime(y["Trading Day"])
    y["Delivery Start"] = pd.to_datetime(y["Delivery Start"])
    y["Delivery End"]   = pd.to_datetime(y["Delivery End"])
    year_offset = y["Delivery Start"].dt.year - y["Trading Day"].dt.year
    y["days_to_start"] = (y["Delivery Start"] - y["Trading Day"]).dt.days
    y["duration"]      = (y["Delivery End"]   - y["Delivery Start"]).dt.days
    y["decimal_year"]  = to_decimal_year(y["Trading Day"]).values
    y_start = _pivot_date(y, year_offset, "YAH", n_yearly, "days_to_start")
    y_dur   = _pivot_date(y, year_offset, "YAH", n_yearly, "duration")
    y_tra   = _pivot_date(y, year_offset, "YAH", n_yearly, "decimal_year")

    return (m_start, q_start, y_start), (m_dur, q_dur, y_dur), (m_tra, q_tra, y_tra)


