"""
=============================================================
  Walmart Sales Forecasting — Power BI Export Script
  Run this AFTER your notebook (walmart_full_pipeline.ipynb)

  NOTE: We load the already-cleaned files the notebook saved.
  No re-cleaning needed — notebook already did all of that.

  Notebook saves:
    data/processed/walmart_master.csv      <- Step 3 output
    data/processed/daily_aggregated.csv    <- Step 5 output
    models/xgboost_walmart.pkl             <- Step 10 output
    models/encoder_category.pkl            <- Step 10 output
    models/encoder_store.pkl               <- Step 10 output
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Output folder
os.makedirs('powerbi_exports', exist_ok=True)
print("📁 Output folder: powerbi_exports/\n")


# ============================================================
# LOAD — already cleaned files from notebook
# No re-cleaning, no re-merging, just load and use
# ============================================================
print("📂 Loading processed files from notebook ...")

df        = pd.read_csv('data/processed/walmart_master.csv',   parse_dates=['date'])
daily_agg = pd.read_csv('data/processed/daily_aggregated.csv', parse_dates=['date'])

print(f"   ✅ walmart_master.csv    -> {len(df):,} rows")
print(f"   ✅ daily_aggregated.csv  -> {len(daily_agg):,} rows\n")


# ============================================================
# EXPORT 1 — walmart_sales.csv
# ============================================================
print("⏳ Building Export 1: walmart_sales.csv ...")

if 'day_name' not in df.columns:
    df['day_name'] = df['date'].dt.day_name()

wanted_cols = [
    'transaction_id', 'date', 'hour', 'day_name', 'dayofweek', 'week',
    'store_id', 'city', 'region', 'latitude', 'longitude',
    'category', 'subcategory', 'sku', 'product_name',
    'quantity', 'unit_price', 'unit_cost', 'revenue', 'margin', 'margin_pct',
    'supplier'
]
available_cols = [c for c in wanted_cols if c in df.columns]
missing_cols   = [c for c in wanted_cols if c not in df.columns]
if missing_cols:
    print(f"   ⚠️  Columns not in master file (skipped): {missing_cols}")

sales_export = df[available_cols].copy()
sales_export.to_csv('powerbi_exports/walmart_sales.csv', index=False)
print(f"   ✅ walmart_sales.csv  ->  {len(sales_export):,} rows, {len(available_cols)} columns")


# ============================================================
# EXPORT 2 — store_locations.csv
# ============================================================
print("⏳ Building Export 2: store_locations.csv ...")

store_rev  = df.groupby('store_id')['revenue'].sum().reset_index(name='total_revenue')
store_txns = df.groupby('store_id')['transaction_id'].count().reset_index(name='total_transactions')

store_cols = [c for c in ['store_id','store_name','city','region','latitude','longitude'] if c in df.columns]
store_dim  = df[store_cols].drop_duplicates('store_id').copy()
store_dim  = store_dim.merge(store_rev,  on='store_id', how='left')
store_dim  = store_dim.merge(store_txns, on='store_id', how='left')
store_dim['avg_transaction'] = (store_dim['total_revenue'] / store_dim['total_transactions']).round(2)

store_dim.to_csv('powerbi_exports/store_locations.csv', index=False)
print(f"   ✅ store_locations.csv  ->  {len(store_dim)} stores")


# ============================================================
# EXPORT 3 — daily_aggregated.csv
# Already built by notebook Step 5 — just copy it over
# ============================================================
print("⏳ Building Export 3: daily_aggregated.csv ...")

if 'day_name' not in daily_agg.columns:
    daily_agg['day_name'] = daily_agg['date'].dt.day_name()

daily_agg.to_csv('powerbi_exports/daily_aggregated.csv', index=False)
print(f"   ✅ daily_aggregated.csv  ->  {len(daily_agg):,} rows")


# ============================================================
# EXPORT 4 — xgboost_predictions.csv
# ============================================================
print("⏳ Building Export 4: xgboost_predictions.csv ...")

try:
    xgb      = joblib.load('models/xgboost_walmart.pkl')
    le_cat_m = joblib.load('models/encoder_category.pkl')
    le_sto_m = joblib.load('models/encoder_store.pkl')

    FEATURES = [
        'store_enc', 'category_enc',
        'dayofweek', 'day', 'week',
        'is_weekend', 'month_start', 'month_end',
        'num_transactions', 'avg_price',
        'lag_1d', 'lag_7d', 'roll_3d'
    ]

    model_df = daily_agg.dropna(subset=FEATURES + ['total_revenue']).copy()
    model_df['category_enc'] = le_cat_m.transform(model_df['category'])
    model_df['store_enc']    = le_sto_m.transform(model_df['store_id'])

    split   = model_df['date'].max() - pd.Timedelta(days=7)
    test_df = model_df[model_df['date'] > split].copy()

    test_df['predicted_revenue'] = xgb.predict(test_df[FEATURES])
    test_df['error']             = test_df['total_revenue'] - test_df['predicted_revenue']
    test_df['abs_error']         = test_df['error'].abs()
    test_df['pct_error']         = (test_df['abs_error'] / (test_df['total_revenue'] + 1e-5) * 100).round(2)
    test_df['model']             = 'XGBoost'

    pred_export = test_df[[
        'date', 'store_id', 'category',
        'total_revenue', 'predicted_revenue',
        'error', 'abs_error', 'pct_error', 'model'
    ]]
    pred_export.to_csv('powerbi_exports/xgboost_predictions.csv', index=False)
    print(f"   ✅ xgboost_predictions.csv  ->  {len(pred_export):,} rows")

except FileNotFoundError:
    print("   ⚠️  models/ not found — run notebook Step 10 first, then re-run this script")


# ============================================================
# EXPORT 5 — prophet_forecast.csv
# ============================================================
print("⏳ Building Export 5: prophet_forecast.csv ...")

try:
    from prophet import Prophet

    store_daily = (
        daily_agg.groupby('date')['total_revenue']
        .sum().reset_index()
        .rename(columns={'date': 'ds', 'total_revenue': 'y'})
        .sort_values('ds')
    )

    split_date = store_daily['ds'].max() - pd.Timedelta(days=7)
    train_p    = store_daily[store_daily['ds'] <= split_date]

    model_p = Prophet(
        yearly_seasonality=False, weekly_seasonality=True,
        daily_seasonality=False, changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative'
    )
    model_p.fit(train_p)
    future   = model_p.make_future_dataframe(periods=14, freq='D')
    forecast = model_p.predict(future)

    prophet_export = forecast[['ds','yhat','yhat_lower','yhat_upper','trend','weekly']].copy()
    prophet_export.columns = ['date','forecast','forecast_lower','forecast_upper','trend','weekly_effect']
    prophet_export['is_future'] = (prophet_export['date'] > store_daily['ds'].max()).astype(int)
    prophet_export = prophet_export.merge(
        store_daily.rename(columns={'ds':'date','y':'actual_revenue'}),
        on='date', how='left'
    )
    prophet_export['model'] = 'Prophet'
    prophet_export.to_csv('powerbi_exports/prophet_forecast.csv', index=False)
    print(f"   ✅ prophet_forecast.csv  ->  {len(prophet_export)} rows ({prophet_export['is_future'].sum()} future days)")

except Exception as e:
    print(f"   ⚠️  Prophet export skipped: {e}")


# ============================================================
# EXPORT 6 — model_metrics.csv
# ============================================================
print("⏳ Building Export 6: model_metrics.csv ...")

try:
    pred_df = pd.read_csv('powerbi_exports/xgboost_predictions.csv')
    mae_x   = mean_absolute_error(pred_df['total_revenue'], pred_df['predicted_revenue'])
    rmse_x  = np.sqrt(mean_squared_error(pred_df['total_revenue'], pred_df['predicted_revenue']))
    mape_x  = pred_df['pct_error'].mean()

    prop_df   = pd.read_csv('powerbi_exports/prophet_forecast.csv')
    prop_test = prop_df[prop_df['actual_revenue'].notna() & (prop_df['is_future'] == 0)].tail(7)
    mae_p     = mean_absolute_error(prop_test['actual_revenue'], prop_test['forecast'])
    rmse_p    = np.sqrt(mean_squared_error(prop_test['actual_revenue'], prop_test['forecast']))
    mape_p    = np.mean(np.abs((prop_test['actual_revenue'] - prop_test['forecast']) /
                               (prop_test['actual_revenue'] + 1e-5))) * 100

    metrics = pd.DataFrame({
        'Model':        ['XGBoost', 'Prophet'],
        'MAE':          [round(mae_x, 2),      round(mae_p, 2)],
        'RMSE':         [round(rmse_x, 2),     round(rmse_p, 2)],
        'MAPE_pct':     [round(mape_x, 2),     round(mape_p, 2)],
        'Accuracy_pct': [round(100-mape_x, 2), round(100-mape_p, 2)],
        'Better_model': ['XGBoost' if mae_x < mae_p else 'Prophet'] * 2
    })
    metrics.to_csv('powerbi_exports/model_metrics.csv', index=False)
    print(f"   ✅ model_metrics.csv  ->  2 rows")
    print(f"\n   XGBoost -> MAE: ${mae_x:,.0f} | RMSE: ${rmse_x:,.0f} | MAPE: {mape_x:.1f}% | Accuracy: {100-mape_x:.1f}%")
    print(f"   Prophet -> MAE: ${mae_p:,.0f} | RMSE: ${rmse_p:,.0f} | MAPE: {mape_p:.1f}% | Accuracy: {100-mape_p:.1f}%")

except Exception as e:
    print(f"   ⚠️  Metrics export skipped: {e}")


# ============================================================
print("\n" + "="*55)
print("✅ ALL EXPORTS COMPLETE — powerbi_exports/ folder")
print("="*55)
print("""
FILES TO LOAD IN POWER BI:
  walmart_sales.csv          -> Revenue Overview, EDA pages
  store_locations.csv        -> Store Map page
  daily_aggregated.csv       -> Daily trend charts
  xgboost_predictions.csv    -> Forecast Accuracy page
  prophet_forecast.csv       -> Revenue Forecast page
  model_metrics.csv          -> KPI cards, Model Comparison

POWER BI RELATIONSHIPS:
  walmart_sales.store_id        -> store_locations.store_id
  walmart_sales.date            -> daily_aggregated.date
  xgboost_predictions.store_id  -> store_locations.store_id
  xgboost_predictions.date      -> daily_aggregated.date
""")
