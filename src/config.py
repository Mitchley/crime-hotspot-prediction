"""Project-wide configuration: paths, column lists, model settings."""
from pathlib import Path

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_CSV = DATA_RAW / "sapacr-2005-2025-v1.3.csv"
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
FORECAST_PARQUET = DATA_PROCESSED / "forecast_next_year.parquet"
MODELS_FILE = MODELS_DIR / "hotspot_models.pkl"
COMPARISON_FILE = MODELS_DIR / "comparison.csv"
METRICS_FILE = MODELS_DIR / "metrics.csv"

for _p in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------- Delimitation ----------------
DC_FILTER = "city of cape town"

# ---------------- Columns ----------------
META_COLUMNS = ["year", "station", "loc_mn", "dc_mn", "longitude", "latitude"]

CRIME_COLUMNS = [
    "other_theft", "arson", "assault_gbh", "attempted_murder",
    "attempted_sexoff", "bank_robbery", "burglary_nonres", "burglary_res",
    "carjacking", "commercial_crime", "common_assault", "common_robbery",
    "contact_sexoff", "dui", "drug_crime", "illegal_firearms", "kidnapping",
    "malicious_damage", "murder", "rape", "robbery_nonres", "robbery_res",
    "cash_transit_robbery", "aggr_robbery", "sexual_assault",
    "police_detected_sexoff", "shoplifting", "stock_theft",
    "vehicle_theft", "theft_from_vehicle", "truck_hijacking",
]

# ---------------- Hotspot definition ----------------
HOTSPOT_QUANTILE = 0.80
MIN_NONZERO = 50

# ---------------- Feature engineering ----------------
LAGS = [1, 2, 3]
ROLLING_WINDOW = 3
TREND_WINDOW = 5

# ---------------- Train / test split ----------------
TRAIN_YEARS = (2008, 2019)
TEST_YEARS = (2020, 2024)

# ---------------- XGBoost ----------------
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

# ---------------- LSTM ----------------
LSTM_PARAMS = dict(
    embed_dim=64,
    lstm_units=48,
    dropout=0.3,
    epochs=25,
    batch_size=128,
    learning_rate=1e-3,
    verbose=0,
)

# ---------------- Model comparison ----------------
CANDIDATE_MODELS = ["logreg", "random_forest", "xgboost"]