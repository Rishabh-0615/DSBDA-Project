import pandas as pd

def generate_kpis(df):
    if df is None or df.empty:
        return {'total_sales': 0, 'total_profit': 0, 'total_orders': 0, 'avg_sales': 0}
        
    total_sales = df['sales'].sum() if 'sales' in df.columns else 0
    total_profit = df['profit'].sum() if 'profit' in df.columns else 0
    total_orders = len(df)
    avg_sales = df['sales'].mean() if 'sales' in df.columns else 0
    
    return {
        'total_sales': round(total_sales, 2),
        'total_profit': round(total_profit, 2),
        'total_orders': total_orders,
        'avg_sales': round(avg_sales, 2)
    }

def get_sales_by_region(df):
    if 'region' not in df.columns or 'sales' not in df.columns:
        return {'labels': [], 'values': []}
    
    grouped = df.groupby('region')['sales'].sum().reset_index()
    return {
        'labels': grouped['region'].tolist(),
        'values': grouped['sales'].tolist()
    }

def get_sales_by_category(df):
    if 'category' not in df.columns or 'sales' not in df.columns:
        return {'labels': [], 'values': []}
        
    grouped = df.groupby('category')['sales'].sum().reset_index()
    return {
        'labels': grouped['category'].tolist(),
        'values': grouped['sales'].tolist()
    }

def get_profit_by_segment(df):
    if 'segment' not in df.columns or 'profit' not in df.columns:
        return {'labels': [], 'values': []}
        
    grouped = df.groupby('segment')['profit'].sum().reset_index()
    return {
        'labels': grouped['segment'].tolist(),
        'values': grouped['profit'].tolist()
    }

def get_monthly_sales_trend(df):
    if 'order_date' not in df.columns or 'sales' not in df.columns:
        return {'labels': [], 'values': []}
        
    df_copy = df.copy()
    try:
        df_copy = df_copy.dropna(subset=['order_date'])
        df_copy['month_year'] = df_copy['order_date'].dt.strftime('%Y-%m')
        grouped = df_copy.groupby('month_year')['sales'].sum().reset_index().sort_values('month_year')
        return {
            'labels': grouped['month_year'].tolist(),
            'values': grouped['sales'].tolist()
        }
    except Exception:
        return {'labels': [], 'values': []}
