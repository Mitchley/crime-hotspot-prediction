"""Train, compare, and finalize hotspot classifiers for every crime type."""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from . import config as C
from .data_preprocessor import get_feature_columns


# ----------------------------------------------------------------------
# Optional LSTM classifier (kept for compatibility)
# ----------------------------------------------------------------------
class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embed_dim=64, lstm_units=48, dropout=0.3,
                 epochs=25, batch_size=128, learning_rate=1e-3, verbose=0,
                 random_state=42):
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.random_state = random_state
        self.model_ = None
        self.n_features_ = None

    def _build(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers

        tf.random.set_seed(self.random_state)
        inp = layers.Input(shape=(1, self.n_features_))
        x = layers.Dense(self.embed_dim, activation="relu")(inp)
        x = layers.LSTM(self.lstm_units, return_sequences=False)(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = models.Model(inp, out)
        m.compile(optimizer=optimizers.Adam(self.learning_rate),
                  loss="binary_crossentropy", metrics=["accuracy"])
        return m

    def fit(self, X, y):
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.n_features_ = X.shape[1]
        X3 = X.reshape(-1, 1, self.n_features_)

        classes = np.unique(y)
        if len(classes) > 1:
            cw = compute_class_weight("balanced", classes=classes, y=y)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
        else:
            class_weight = None

        self.model_ = self._build()
        self.model_.fit(X3, y, epochs=self.epochs, batch_size=self.batch_size,
                        verbose=self.verbose, class_weight=class_weight)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype="float32").reshape(-1, 1, self.n_features_)
        p = self.model_.predict(X, verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ----------------------------------------------------------------------
# Model factory
# ----------------------------------------------------------------------
def make_model(name: str):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000,
                                       class_weight="balanced",
                                       random_state=42)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )
    if name == "xgboost":
        return XGBClassifier(**C.XGB_PARAMS)
    if name == "lstm":
        return LSTMClassifier(**C.LSTM_PARAMS)
    raise ValueError(f"Unknown model: {name}")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _year_mask(df, years):
    return df["year_start"].between(*years)


def _usable_crimes(df):
    usable = []
    for c in C.CRIME_COLUMNS:
        t = f"{c}_hotspot"
        if t not in df.columns:
            continue
        if df[t].sum() < C.MIN_NONZERO:
            print(f"[skip] {c:25s} only {int(df[t].sum())} hotspots")
            continue
        usable.append(c)
    return usable


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------
def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    fcols = get_feature_columns(df)
    train_mask = _year_mask(df, C.TRAIN_YEARS)
    test_mask = _year_mask(df, C.TEST_YEARS)

    X_train = df.loc[train_mask, fcols].fillna(0)
    X_test = df.loc[test_mask, fcols].fillna(0)

    crimes = _usable_crimes(df)
    n_algos = len(C.CANDIDATE_MODELS)
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON  |  {n_algos} algorithms  ×  {len(crimes)} crimes")
    print(f"{'='*70}")
    print(f"Train: {C.TRAIN_YEARS[0]}–{C.TRAIN_YEARS[1]}   "
          f"Test: {C.TEST_YEARS[0]}–{C.TEST_YEARS[1]}")
    print(f"{'='*70}\n")

    rows = []
    for crime in crimes:
        target = f"{crime}_hotspot"
        y_train = df.loc[train_mask, target].values
        y_test = df.loc[test_mask, target].values
        if len(np.unique(y_train)) < 2:
            print(f"[skip] {crime}: single-class training data")
            continue

        print(f"{crime}")
        for algo in C.CANDIDATE_MODELS:
            try:
                m = make_model(algo)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                y_prob = m.predict_proba(X_test)[:, 1]
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except ValueError:
                    auc = np.nan

                rows.append(dict(
                    crime=crime, algorithm=algo,
                    accuracy=accuracy_score(y_test, y_pred),
                    precision=precision_score(y_test, y_pred, zero_division=0),
                    recall=recall_score(y_test, y_pred, zero_division=0),
                    f1=f1_score(y_test, y_pred, zero_division=0),
                    roc_auc=auc,
                ))
                print(f"    {algo:14s}  AUC = {auc:.3f}")
            except Exception as e:
                print(f"    {algo:14s}  FAILED: {e}")
                rows.append(dict(crime=crime, algorithm=algo,
                                 accuracy=np.nan, precision=np.nan,
                                 recall=np.nan, f1=np.nan, roc_auc=np.nan))
        print()
    return pd.DataFrame(rows)


def pick_global_winner(comparison: pd.DataFrame) -> str:
    summary = (comparison.dropna(subset=["roc_auc"])
               .groupby("algorithm")[["accuracy", "precision", "recall",
                                      "f1", "roc_auc"]]
               .mean()
               .sort_values("roc_auc", ascending=False))

    print(f"\n{'='*70}")
    print("ALGORITHM LEADERBOARD  (mean across all crimes)")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Algorithm':<16}{'Accuracy':>10}{'Precision':>11}"
          f"{'Recall':>9}{'F1':>8}{'AUC':>9}")
    print(f"{'-'*70}")
    for i, (name, row) in enumerate(summary.iterrows(), start=1):
        marker = "  ← BEST" if i == 1 else ""
        print(f"{i:<6}{name:<16}{row['accuracy']:>10.3f}"
              f"{row['precision']:>11.3f}{row['recall']:>9.3f}"
              f"{row['f1']:>8.3f}{row['roc_auc']:>9.3f}{marker}")
    print(f"{'='*70}\n")

    winner = summary.index[0]
    print(f">>> Global winner: {winner.upper()}")
    print(f"    (chosen by highest mean ROC-AUC across "
          f"{len(summary)} algorithms)\n")
    return winner


# ----------------------------------------------------------------------
# Final training
# ----------------------------------------------------------------------
def train_final(df, algorithm):
    fcols = get_feature_columns(df)
    train_mask = _year_mask(df, C.TRAIN_YEARS)
    test_mask = _year_mask(df, C.TEST_YEARS)

    X_train = df.loc[train_mask, fcols].fillna(0)
    X_test = df.loc[test_mask, fcols].fillna(0)

    models, rows = {}, []
    print(f"\n{'='*70}")
    print(f"FINAL TRAINING  |  algorithm = {algorithm}")
    print(f"{'='*70}\n")

    for crime in _usable_crimes(df):
        target = f"{crime}_hotspot"
        y_train = df.loc[train_mask, target].values
        y_test = df.loc[test_mask, target].values
        if len(np.unique(y_train)) < 2:
            continue

        m = make_model(algorithm)
        m.fit(X_train, y_train)
        models[crime] = m

        y_pred = m.predict(X_test)
        try:
            auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        except ValueError:
            auc = np.nan

        rows.append(dict(crime=crime, algorithm=algorithm,
                         accuracy=accuracy_score(y_test, y_pred),
                         precision=precision_score(y_test, y_pred,
                                                  zero_division=0),
                         recall=recall_score(y_test, y_pred, zero_division=0),
                         f1=f1_score(y_test, y_pred, zero_division=0),
                         roc_auc=auc))
        print(f"  [ok] {crime:25s} AUC = {auc:.3f}")

    return models, pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def save_models(models, metrics):
    joblib.dump(models, C.MODELS_FILE)
    metrics.to_csv(C.METRICS_FILE, index=False)
    print(f"\nSaved {len(models)} final models -> {C.MODELS_FILE}")
    print(f"Saved final metrics       -> {C.METRICS_FILE}")


def load_models():
    return joblib.load(C.MODELS_FILE)