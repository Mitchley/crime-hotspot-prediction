"""Load trained models and produce hotspot probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data_preprocessor import get_feature_columns
from .model_trainer import load_models


class HotspotPredictor:
    def __init__(self):
        self.models = load_models()
        self.feature_cols: list[str] | None = None

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols is None:
            self.feature_cols = get_feature_columns(df)
        return df[self.feature_cols].fillna(0)

    def predict_station_year(self, df, station, year) -> dict:
        row = df[(df["station"] == station) & (df["year_start"] == year)]
        if row.empty:
            return {}
        X = self._prep(row)
        return {c: float(m.predict_proba(X)[:, 1][0])
                for c, m in self.models.items()}

    def predict_all_latest(self, df, year=None) -> pd.DataFrame:
        if year is None:
            year = int(df["year_start"].max())
        sub = df[df["year_start"] == year].copy()
        X = self._prep(sub)
        for c, m in self.models.items():
            sub[f"{c}_prob"] = m.predict_proba(X)[:, 1]
        return sub

    def top_stations(self, df, crime, year=None, k=10) -> pd.DataFrame:
        preds = self.predict_all_latest(df, year)
        col = f"{crime}_prob"
        return (preds[["station", "loc_mn", "dc_mn", "longitude", "latitude", col]]
                .sort_values(col, ascending=False).head(k).reset_index(drop=True))

    # ------------------------------------------------------------------
    # NEW: forward-looking forecast for the year AFTER the last one
    # ------------------------------------------------------------------
    def forecast_next_year(self, df: pd.DataFrame):
        """
        Forecast the next year using only data up to the latest available year.

        Approach (naive persistence):
          - For each station, take the latest year's crime counts as the
            assumed counts for next_year.
          - Recompute lag/rolling/trend features on the extended timeline.
          - Feed those features into each per-crime model.

        Returns
        -------
        (forecast_df, next_year)
            forecast_df: one row per station with {crime}_prob columns.
            next_year:   int, the forecast year.
        """
        latest = int(df["year_start"].max())
        next_year = latest + 1

        latest_rows = df[df["year_start"] == latest].copy()
        if latest_rows.empty:
            raise ValueError("No rows for the latest year in the dataset.")

        # Synthetic next-year rows: same crime counts as latest year
        synth = latest_rows.copy()
        synth["year_start"] = next_year

        # Stack and recompute lag/rolling/trend features on the extended timeline
        combined = pd.concat([df, synth], ignore_index=True)
        combined = combined.sort_values(["station", "year_start"])

        g = combined.groupby("station", sort=False)
        for c in C.CRIME_COLUMNS:
            if c not in combined.columns:
                continue
            for k in C.LAGS:
                combined[f"{c}_lag{k}"] = g[c].shift(k)
            combined[f"{c}_roll{C.ROLLING_WINDOW}"] = g[c].transform(
                lambda s: s.shift(1).rolling(
                    C.ROLLING_WINDOW, min_periods=1).mean()
            )
            combined[f"{c}_trend{C.TREND_WINDOW}"] = g[c].transform(
                lambda s: s.shift(1).rolling(
                    C.TREND_WINDOW, min_periods=C.TREND_WINDOW
                ).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0]
                    if not np.isnan(x).any() else np.nan,
                    raw=True,
                )
            )

        # Keep only the synthetic next-year rows
        forecast = combined[combined["year_start"] == next_year].copy()

        # Predict using the trained per-crime models
        X = self._prep(forecast)
        for c, m in self.models.items():
            forecast[f"{c}_prob"] = m.predict_proba(X)[:, 1]

        return forecast, next_year