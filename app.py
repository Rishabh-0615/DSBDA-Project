import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from utils.data_loader import load_data, preprocess_data
from utils.analytics import generate_kpis, get_sales_by_region, get_sales_by_category, get_profit_by_segment, get_monthly_sales_trend
from utils.predictor import (
    ModelTrainingError,
    ensure_required_model_artifacts,
    load_model_metadata,
    predict,
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state to store the loaded dataframe (in a real app, use a database or caching mechanism)
app.config['DF'] = None

MODEL_PATH = 'model/sales_model.pkl'
METADATA_PATH = 'model/model_metadata.json'


def _load_existing_df_if_available():
    df = app.config['DF']
    if df is not None:
        return df

    data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'superstore.csv')
    if not os.path.exists(data_path):
        return None

    df = load_data(data_path)
    df = preprocess_data(df)
    app.config['DF'] = df
    return df


def _ensure_model_ready(df):
    try:
        return ensure_required_model_artifacts(df, MODEL_PATH, METADATA_PATH)
    except Exception as exc:
        app.logger.exception('Model preparation failed: %s', exc)
        return None, None


def _prediction_options(df, metadata):
    defaults = (metadata or {}).get('prediction_defaults', {})

    def values_for(col):
        if col in df.columns:
            vals = sorted(v for v in df[col].dropna().astype(str).str.strip().unique().tolist() if v)
            if vals:
                return vals
        if defaults.get(col):
            return [defaults[col]]
        return ['Unknown']

    return {
        'regions': values_for('region'),
        'categories': values_for('category'),
        'sub_categories': values_for('sub_category'),
        'segments': values_for('segment'),
        'ship_modes': values_for('ship_mode'),
        'states': values_for('state'),
    }


def _analysis_filter_options(df):
    def values_for(col):
        if col in df.columns:
            vals = sorted(v for v in df[col].dropna().astype(str).str.strip().unique().tolist() if v)
            if vals:
                return vals
        return []

    return {
        'regions': values_for('region'),
        'categories': values_for('category'),
    }


def _apply_filters(df, region=None, category=None, start_month=None, end_month=None):
    filtered = df.copy()

    if region and region.lower() != 'all' and 'region' in filtered.columns:
        filtered = filtered[filtered['region'].astype(str) == region]

    if category and category.lower() != 'all' and 'category' in filtered.columns:
        filtered = filtered[filtered['category'].astype(str) == category]

    if 'order_date' in filtered.columns:
        if start_month:
            start_ts = pd.to_datetime(f"{start_month}-01", errors='coerce')
            if pd.notna(start_ts):
                filtered = filtered[filtered['order_date'] >= start_ts]

        if end_month:
            end_ts = pd.to_datetime(f"{end_month}-01", errors='coerce')
            if pd.notna(end_ts):
                end_ts = end_ts + pd.offsets.MonthEnd(0)
                filtered = filtered[filtered['order_date'] <= end_ts]

    return filtered

@app.route('/')
def dashboard():
    df = _load_existing_df_if_available()
    if df is None:
        return redirect(url_for('upload'))

    _ensure_model_ready(df)
            
    if df is not None:
        kpis = generate_kpis(df)
        return render_template('dashboard.html', kpis=kpis)
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            filename = 'superstore.csv' # Always save as superstore.csv for simplicity
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess
            df = load_data(filepath)
            df = preprocess_data(df)
            if df is None or df.empty:
                return render_template('upload.html', error='Uploaded CSV could not be processed.')

            app.config['DF'] = df

            # Re-train model and regenerate metrics for new dataset.
            _ensure_model_ready(df)
            
            return redirect(url_for('dashboard'))

    return render_template('upload.html', error=None)

@app.route('/analysis')
def analysis():
    df = _load_existing_df_if_available()
    if df is None:
        return redirect(url_for('upload'))
    filter_options = _analysis_filter_options(df)
    return render_template('analysis.html', filter_options=filter_options)

@app.route('/api/chart-data')
def chart_data():
    df = _load_existing_df_if_available()
    if df is None:
        return jsonify({"error": "No data available"}), 400

    region = request.args.get('region')
    category = request.args.get('category')
    start_month = request.args.get('start_month')
    end_month = request.args.get('end_month')
    df = _apply_filters(df, region=region, category=category, start_month=start_month, end_month=end_month)
        
    region_sales = get_sales_by_region(df)
    category_sales = get_sales_by_category(df)
    segment_profit = get_profit_by_segment(df)
    monthly_trend = get_monthly_sales_trend(df)
    
    return jsonify({
        'region_sales': region_sales,
        'category_sales': category_sales,
        'segment_profit': segment_profit,
        'monthly_trend': monthly_trend
    })

@app.route('/predict')
def prediction():
    df = _load_existing_df_if_available()
    if df is None:
        return redirect(url_for('upload'))

    model_bundle, metadata = _ensure_model_ready(df)
    if model_bundle is None:
        fallback_options = {
            'regions': ['Unknown'],
            'categories': ['Unknown'],
            'sub_categories': ['Unknown'],
            'segments': ['Unknown'],
            'ship_modes': ['Unknown'],
            'states': ['Unknown'],
        }
        return render_template('prediction.html', options=fallback_options, model_error='Model is not available yet.')

    options = _prediction_options(df, metadata)
    return render_template('prediction.html', options=options, model_error=None)

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    data = request.json or {}
    df = _load_existing_df_if_available()
    if df is None:
        return jsonify({'error': 'No dataset loaded. Upload CSV first.'}), 400

    model_bundle, metadata = _ensure_model_ready(df)
    if model_bundle is None:
        return jsonify({'error': 'Model is not available.'}), 500

    try:
        predicted_sales = predict(model_bundle, data, metadata=metadata)
    except ModelTrainingError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify({'error': 'Prediction failed due to internal error.'}), 500
    
    return jsonify({'predicted_sales': predicted_sales})

@app.route('/model')
def model_metrics():
    df = _load_existing_df_if_available()
    if df is None:
        return redirect(url_for('upload'))

    _ensure_model_ready(df)
    metadata = load_model_metadata(METADATA_PATH)
    if metadata is None:
        metrics = None
        leaderboard = {}
        trained_at = None
        best_model = None
        train_rows = 0
        test_rows = 0
    else:
        metrics = metadata.get('metrics', {})
        leaderboard = metadata.get('model_leaderboard', {})
        trained_at = metadata.get('trained_at')
        best_model = metadata.get('best_model')
        train_rows = metadata.get('train_rows', 0)
        test_rows = metadata.get('test_rows', 0)

    return render_template(
        'model_metrics.html',
        metrics=metrics,
        leaderboard=leaderboard,
        trained_at=trained_at,
        best_model=best_model,
        train_rows=train_rows,
        test_rows=test_rows,
    )
    
    
@app.route('/api/model-info')
def model_info():
    metadata = load_model_metadata(METADATA_PATH)
    if metadata is None:
        return jsonify({'error': 'Model metadata not available.'}), 404

    return jsonify(metadata)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
