# SuperStore Sales Data Analysis & Prediction - Web Dashboard

## Overview
This project provides a complete sales analytics and prediction web application built with Flask.
It includes:
1. Data upload and preprocessing
2. KPI and chart-based analytics
3. Real model training and model selection
4. Persisted model inference for live sales prediction
5. Real evaluation metrics and model leaderboard

## What Is Fully Implemented
There are no placeholder prediction/metrics paths in the app now.

1. Uploading a CSV triggers preprocessing and model training.
2. The best-performing regression model is selected by RMSE.
3. Model and metadata are persisted to disk:
   - model/sales_model.pkl
   - model/model_metadata.json
4. Prediction page sends real feature inputs to the trained pipeline.
5. Model Metrics page shows real MAE, MSE, RMSE, MAPE, R2, train/test counts, and per-model leaderboard.

## Model Pipeline
Implemented in utils/predictor.py.

1. Feature engineering:
   - order_month
   - order_year
   - ship_lag_days
2. Categorical features:
   - region, category, sub_category, segment, ship_mode, state
3. Preprocessing:
   - SimpleImputer for missing values
   - OneHotEncoder(handle_unknown='ignore') for categorical columns
4. Candidate models:
   - LinearRegression
   - RandomForestRegressor
   - GradientBoostingRegressor
   - XGBRegressor (optional, only when available)
5. Train/test split: 80/20 (random_state=42)
6. Selection criterion: lowest RMSE

## Folder Notes
1. app.py: Flask routes and orchestration for upload, analytics, prediction, and metrics.
2. utils/data_loader.py: schema normalization and preprocessing.
3. utils/analytics.py: KPI and chart aggregation endpoints.
4. utils/predictor.py: training, persistence, metadata, and inference.
5. templates/: pages for dashboard, analysis, prediction, and model metrics.

## Required Dataset Columns
For end-to-end prediction/training flow, dataset should contain:
1. Sales
2. Order Date
3. Ship Date
4. Region
5. Category

Additional columns improve prediction context:
1. Sub-Category
2. Segment
3. Ship Mode
4. State

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

4. Open browser:

```text
http://localhost:5000/
```

## How To Test
1. Open Upload page and upload data/superstore.csv.
2. Verify Dashboard KPIs and charts are populated.
3. Open Sales Prediction page:
   - Choose region/category/sub-category/segment/ship mode/state
   - Enter order date and ship date
   - Click Generate Prediction
   - Confirm predicted sales value appears
4. Open Model Metrics page:
   - Verify MAE, MSE, RMSE, MAPE, R2 are real numbers
   - Verify Best Model and Trained At are populated
   - Verify leaderboard table has multiple models
5. Verify artifacts are created:
   - model/sales_model.pkl
   - model/model_metadata.json

## API Endpoints
1. /api/chart-data: Dashboard/analysis chart payloads.
2. /predict_sales: Live prediction endpoint.
3. /api/model-info: Model metadata JSON.

## Deploy On Render
1. Push this repo to GitHub.
2. In Render, create a new Web Service and connect the GitHub repository.
3. Use these settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Render will use `render.yaml` and `Procfile` if present.
5. After deploy, open the live URL and upload `data/superstore.csv` if the dashboard starts empty.

## Local Test Before Deploy
1. Run `python app.py` locally.
2. Open `http://localhost:5000/`.
3. Upload the dataset.
4. Verify dashboard, analysis, prediction, and model metrics pages.
