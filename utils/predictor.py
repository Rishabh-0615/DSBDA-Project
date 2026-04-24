import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


DEFAULT_TARGET = "sales"
DEFAULT_RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "region",
    "category",
    "sub_category",
    "segment",
    "ship_mode",
    "state",
]
NUMERICAL_FEATURES = ["order_month", "order_year", "ship_lag_days"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


class ModelTrainingError(Exception):
    pass


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_training_columns(df):
    required = {"sales", "order_date", "ship_date"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ModelTrainingError(
            "Dataset missing required columns for model training: " + ", ".join(missing)
        )


def _safe_string_col(df, col):
    if col not in df.columns:
        df[col] = "Unknown"
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()


def _build_feature_frame(df):
    working = df.copy()

    for col in ["order_date", "ship_date"]:
        working[col] = pd.to_datetime(working[col], errors="coerce")

    # Feature engineering from temporal columns.
    working["order_month"] = working["order_date"].dt.month
    working["order_year"] = working["order_date"].dt.year
    working["ship_lag_days"] = (working["ship_date"] - working["order_date"]).dt.days

    for col in CATEGORICAL_FEATURES:
        _safe_string_col(working, col)

    for col in NUMERICAL_FEATURES:
        if col not in working.columns:
            working[col] = np.nan

    x = working[ALL_FEATURES]
    y = pd.to_numeric(working[DEFAULT_TARGET], errors="coerce")

    train_df = x.copy()
    train_df[DEFAULT_TARGET] = y
    train_df = train_df.dropna(subset=[DEFAULT_TARGET])

    if len(train_df) < 50:
        raise ModelTrainingError("Not enough valid rows to train model (need at least 50 rows).")

    x_clean = train_df[ALL_FEATURES]
    y_clean = train_df[DEFAULT_TARGET]
    return x_clean, y_clean


def _candidate_models():
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE),
    }

    if XGBRegressor is not None:
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=DEFAULT_RANDOM_STATE,
        )

    return models


def _build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = float((np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))).mean() * 100)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "mape": float(mape),
    }


def train_and_save_model(df, model_path, metadata_path):
    _validate_training_columns(df)
    x, y = _build_feature_frame(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=DEFAULT_RANDOM_STATE,
    )

    best_name = None
    best_pipeline = None
    best_metrics = None
    leaderboard = {}

    for model_name, model in _candidate_models().items():
        pipeline = _build_pipeline(model)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics = _evaluate(y_test.to_numpy(), y_pred)
        leaderboard[model_name] = metrics

        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_name = model_name
            best_pipeline = pipeline
            best_metrics = metrics

    if best_pipeline is None:
        raise ModelTrainingError("Failed to train any model.")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    bundle = {
        "pipeline": best_pipeline,
        "best_model_name": best_name,
        "features": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target": DEFAULT_TARGET,
    }
    joblib.dump(bundle, model_path)

    defaults = {
        "region": "Unknown",
        "category": "Unknown",
        "sub_category": "Unknown",
        "segment": "Unknown",
        "ship_mode": "Unknown",
        "state": "Unknown",
    }
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in x.columns and not x[cat_col].dropna().empty:
            defaults[cat_col] = str(x[cat_col].mode(dropna=True).iloc[0])

    metadata = {
        "trained_at": _utc_now_iso(),
        "best_model": best_name,
        "metrics": best_metrics,
        "model_leaderboard": leaderboard,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": ALL_FEATURES,
        "prediction_defaults": defaults,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_model(model_path):
    if not os.path.exists(model_path):
        return None

    try:
        return joblib.load(model_path)
    except Exception:
        return None


def load_model_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def ensure_required_model_artifacts(df, model_path, metadata_path):
    model_bundle = load_model(model_path)
    metadata = load_model_metadata(metadata_path)

    if model_bundle is not None and metadata is not None:
        return model_bundle, metadata

    metadata = train_and_save_model(df, model_path, metadata_path)
    model_bundle = load_model(model_path)
    return model_bundle, metadata


def _parse_date_parts(order_date, ship_date):
    order_ts = pd.to_datetime(order_date, errors="coerce")
    ship_ts = pd.to_datetime(ship_date, errors="coerce")

    if pd.isna(order_ts):
        order_ts = pd.Timestamp(datetime.utcnow().date())
    if pd.isna(ship_ts):
        ship_ts = order_ts

    return {
        "order_month": int(order_ts.month),
        "order_year": int(order_ts.year),
        "ship_lag_days": int((ship_ts - order_ts).days),
    }


def predict(model_bundle, payload, metadata=None):
    if model_bundle is None or "pipeline" not in model_bundle:
        raise ModelTrainingError("Model is not available for prediction.")

    defaults = (metadata or {}).get("prediction_defaults", {}) if metadata else {}

    region = str(payload.get("region") or defaults.get("region") or "Unknown").strip()
    category = str(payload.get("category") or defaults.get("category") or "Unknown").strip()
    sub_category = str(payload.get("sub_category") or defaults.get("sub_category") or "Unknown").strip()
    segment = str(payload.get("segment") or defaults.get("segment") or "Unknown").strip()
    ship_mode = str(payload.get("ship_mode") or defaults.get("ship_mode") or "Unknown").strip()
    state = str(payload.get("state") or defaults.get("state") or "Unknown").strip()

    temporal = _parse_date_parts(payload.get("order_date"), payload.get("ship_date"))

    input_row = {
        "region": region,
        "category": category,
        "sub_category": sub_category,
        "segment": segment,
        "ship_mode": ship_mode,
        "state": state,
        "order_month": temporal["order_month"],
        "order_year": temporal["order_year"],
        "ship_lag_days": temporal["ship_lag_days"],
    }

    x_infer = pd.DataFrame([input_row], columns=ALL_FEATURES)
    pred = model_bundle["pipeline"].predict(x_infer)[0]

    return max(0.0, round(float(pred), 2))
