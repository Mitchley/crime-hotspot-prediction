"""Load, clean, engineer, label, and EDA the SAPS crime data."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
sns.set_theme(style="whitegrid", context="talk")


# ---------- Station aliases (merge duplicate labels) ----------
STATION_ALIASES = {
    "beville south africa": "beville",
    "beville south":        "beville",
    # add more as you spot them in the raw data
}


# ---------- Load + clean ----------
def _read_raw():
    return pd.read_csv(
        C.RAW_CSV,
        na_values=["NA", "N/A", "", " ", "-"],
        keep_default_na=True,
        low_memory=False,
    )


def _split_year(df):
    y = df["year"].astype(str).str.split("/", expand=True)
    df["year_start"] = pd.to_numeric(y[0], errors="coerce").astype("Int64")
    df["year_end"] = pd.to_numeric(y[1], errors="coerce").astype("Int64")
    return df.drop(columns=["year"])


def _clean_meta(df):
    df["station"] = df["station"].astype(str).str.strip().str.lower()
    for c in ("loc_mn", "dc_mn"):
        df[c] = (df[c].astype(str).str.strip().str.lower()
                       .replace("nan", np.nan))
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    return df


def _filter_cape_town(df):
    mask = df["dc_mn"].fillna("").str.contains(C.DC_FILTER, case=False, na=False)
    n_before = len(df)
    df = df[mask].copy()
    print(f"[filter] Cape Town: {len(df):,} / {n_before:,} rows kept")
    return df


def _clean_crime_cols(df):
    for c in C.CRIME_COLUMNS:
        if c not in df:
            df[c] = 0
        df[c] = (pd.to_numeric(df[c], errors="coerce")
                 .clip(lower=0).fillna(0).astype("int64"))
    return df


def _drop_bad_coords(df):
    m = df["longitude"].between(16, 33) & df["latitude"].between(-35, -22)
    return df[m | df["longitude"].isna()].copy()


def _impute_coords(df):
    df = df.sort_values(["station", "year_start"]).copy()
    for c in ("longitude", "latitude"):
        df[c] = df.groupby("station")[c].transform(lambda s: s.fillna(s.mean()))
        df[c] = df.groupby("dc_mn")[c].transform(lambda s: s.fillna(s.mean()))
    return df


def _merge_duplicate_stations(df):
    """
    Merge rows that refer to the same physical station but were labeled
    differently in the raw data (e.g. 'beville' vs 'beville south africa').
    Sums crime counts, averages coordinates.
    """
    df = df.copy()

    # Canonicalize station name
    df["station"] = (
        df["station"].astype(str).str.strip().str.lower()
          .replace(STATION_ALIASES)
          .str.replace(r"\s+south\s+africa\s*$", "", regex=True)
          .str.replace(r"\s+saps\s*$", "", regex=True)
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
    )

    group_cols = ["station", "year_start"]
    agg = {c: "sum" for c in C.CRIME_COLUMNS}
    agg.update({"longitude": "mean", "latitude": "mean",
                "loc_mn": "first", "dc_mn": "first"})
    if "year_end" in df.columns:
        agg["year_end"] = "first"

    before = len(df)
    df = df.groupby(group_cols, as_index=False).agg(agg)
    after = len(df)
    if after < before:
        print(f"[dedup] merged {before - after} duplicate station-year rows "
              f"({before:,} -> {after:,})")
    return df


# ---------- Feature engineering ----------
def _slope(y):
    """Least-squares slope of y vs 0..n-1 (fast, no polyfit)."""
    n = len(y)
    x = np.arange(n)
    xm = x.mean()
    return ((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum()


def _add_lag_features(df):
    df = df.sort_values(["station", "year_start"]).copy()
    g = df.groupby("station", sort=False)
    new_cols = {}

    for c in C.CRIME_COLUMNS:
        series = g[c]
        for k in C.LAGS:
            new_cols[f"{c}_lag{k}"] = series.shift(k)
        new_cols[f"{c}_roll{C.ROLLING_WINDOW}"] = series.transform(
            lambda s: s.shift(1).rolling(C.ROLLING_WINDOW,
                                         min_periods=1).mean())
        new_cols[f"{c}_trend{C.TREND_WINDOW}"] = series.transform(
            lambda s: s.shift(1).rolling(C.TREND_WINDOW,
                                         min_periods=C.TREND_WINDOW)
                       .apply(_slope, raw=True))

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _add_hotspot_labels(df):
    for c in C.CRIME_COLUMNS:
        thr = (df.groupby("year_start")[c]
                 .transform(lambda s: s.quantile(C.HOTSPOT_QUANTILE)))
        df[f"{c}_hotspot"] = (df[c] >= thr).astype(int)
    return df


# ---------- Public API ----------
def build_dataset():
    df = (_read_raw()
          .pipe(_split_year)
          .pipe(_clean_meta)
          .pipe(_filter_cape_town)
          .pipe(_clean_crime_cols)
          .pipe(_drop_bad_coords)
          .pipe(_impute_coords)
          .pipe(_merge_duplicate_stations)
          .pipe(_add_lag_features)
          .pipe(_add_hotspot_labels))
    return (df.dropna(subset=["longitude", "latitude", "year_start"])
              .reset_index(drop=True))


def add_dc_dummies(df):
    d = pd.get_dummies(df["dc_mn"].fillna("unknown"), prefix="dc", dtype=int)
    return pd.concat([df, d], axis=1)


def get_feature_columns(df):
    lags = [c for c in df
            if any(c.endswith(f"_lag{k}") for k in C.LAGS)
            or c.endswith(f"_roll{C.ROLLING_WINDOW}")
            or c.endswith(f"_trend{C.TREND_WINDOW}")]
    dummies = [c for c in df if c.startswith("dc_") and c != "dc_mn"]
    return lags + ["longitude", "latitude"] + dummies


def save_features(df):
    df.to_parquet(C.FEATURES_PARQUET, index=False)


def load_features():
    return pd.read_parquet(C.FEATURES_PARQUET)


# ---------- EDA (3 charts only) ----------
def _save(fig, name):
    p = C.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def _eda_crime_totals(df):
    """Which crimes are most common."""
    fig, ax = plt.subplots(figsize=(10, 12))
    df[C.CRIME_COLUMNS].sum().sort_values().plot.barh(ax=ax,
                                                       color="steelblue")
    ax.set_title("Total incidents per crime (City of Cape Town)")
    ax.set_xlabel("Reported incidents (2005–2024)")
    _save(fig, "01_crime_totals.png")


def _eda_crime_trends(df, top=8):
    """How the top crimes have changed over time."""
    cols = df[C.CRIME_COLUMNS].sum().nlargest(top).index
    fig, ax = plt.subplots(figsize=(14, 7))
    df.groupby("year_start")[cols].sum().plot(ax=ax, marker="o", linewidth=2)
    ax.set_title(f"Crime trends — top {top} crimes")
    ax.set_ylabel("Reported incidents")
    ax.set_xlabel("Year")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, "02_crime_trends.png")


def _eda_geo_density(df):
    """Where crime concentrates geographically."""
    y = int(df["year_start"].max())
    s = df[df["year_start"] == y].copy()
    s["total"] = s[C.CRIME_COLUMNS].sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(s["longitude"], s["latitude"], c=s["total"], s=30,
                    cmap="YlOrRd", alpha=0.8)
    ax.set_title(f"Crime density by station ({y})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(sc, ax=ax, label="Total incidents")
    _save(fig, "03_geo_density.png")


def run_eda(df=None):
    if df is None:
        df = build_dataset()
    print("Running EDA (3 charts)...")
    _eda_crime_totals(df)
    _eda_crime_trends(df)
    _eda_geo_density(df)
    print(f"EDA figures -> {C.FIGURES_DIR}")


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()

    if a.eda:
        run_eda()
    if a.build or not (a.eda or a.build):
        d = add_dc_dummies(build_dataset())
        save_features(d)
        print(f"Saved {len(d):,} rows × {d.shape[1]} cols")