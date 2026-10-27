"""Build features, compare 4 algorithms, refit with the winner."""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.data_preprocessor import build_dataset, add_dc_dummies, save_features
from src.model_trainer import (
    compare_models, pick_global_winner, save_models, train_final,
)


def _write_markdown(metrics: pd.DataFrame) -> None:
    path = C.REPORTS_DIR / "model_comparison.md"
    leaderboard = (metrics.dropna(subset=["roc_auc"])
                   .groupby("algorithm")[["accuracy", "precision", "recall",
                                          "f1", "roc_auc"]]
                   .mean().sort_values("roc_auc", ascending=False).round(4))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Algorithm comparison (mean across crimes)\n\n")
        f.write(leaderboard.to_markdown())
        f.write("\n\n## Per-crime detail\n\n")
        f.write(metrics.sort_values(["crime", "algorithm"]).round(4)
                .to_markdown(index=False))
    print(f"Wrote report -> {path}")


def main():
    print("== Phase 1: features ==")
    df = add_dc_dummies(build_dataset())
    save_features(df)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}\n")

    print("== Phase 2: 4-way comparison ==")
    comparison = compare_models(df)
    comparison.to_csv(C.COMPARISON_FILE, index=False)
    print(f"Saved comparison -> {C.COMPARISON_FILE}")
    _write_markdown(comparison)

    winner = pick_global_winner(comparison)

    print("\n== Phase 3: refit with winner ==")
    final_models, final_metrics = train_final(df, winner)
    print("\n=== Final metrics ===")
    print(final_metrics.round(4).to_string(index=False))
    save_models(final_models, final_metrics)


if __name__ == "__main__":
    main()