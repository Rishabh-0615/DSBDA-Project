import pandas as pd

def load_data(filepath):
    try:
        # Assuming latin1 encoding as is common with this dataset or utf-8
        df = pd.read_csv(filepath, encoding='latin1')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    if df is None:
        return None
    
    # Standardize column names
    df.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df.columns]
    
    # Convert dates
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    if 'ship_date' in df.columns:
        df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
    
    # Convert numerical columns
    for col in ['sales', 'profit', 'quantity', 'discount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing sales rows to keep analytics/prediction stable.
    if 'sales' in df.columns:
        df = df.dropna(subset=['sales'])

    # Keep categorical columns consistent for downstream one-hot encoding.
    for col in ['region', 'category', 'sub_category', 'segment', 'ship_mode', 'state', 'city']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        
    return df
