import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
import logging

# Suppress excessive Prophet logs to keep the console clean
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.WARNING)

# ==========================================
# Database Connection and Extraction
# ==========================================
engine = create_engine('postgresql://postgres:MGolRhrRWkSXY6M2@db.znprnevdjfwpgxetgmjl.supabase.co:5432/postgres')

query_migration = """
    SELECT f.startdate, g.countryname AS receiving_country
    FROM f_mobility f
    JOIN d_geography g ON f.receivinggeography = g.geographyid
    WHERE f.startdate IS NOT NULL
"""
df_migration = pd.read_sql(query_migration, engine)

# ==========================================
# Pre-processing
# ==========================================
df_migration['startdate'] = pd.to_datetime(df_migration['startdate'], format='%Y%m%d')

# Remove 2020 and 2021 (atypical years due to the pandemic)
df_migration = df_migration[~(df_migration['startdate'].dt.year.isin([2020, 2021]))]

# Group by start month and destination country
df_grouped = df_migration.groupby([pd.Grouper(key='startdate', freq='MS'), 'receiving_country']).size().reset_index(name='total_students')

# ==========================================
# Modeling and Forecasting with Prophet
# ==========================================
predictions_list = []

# Manually create target dates for 2023 and 2024 (January to December)
target_dates = pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS')
future_dataframe = pd.DataFrame({'ds': target_dates})

print("Starting time series forecasting...")

for country in df_grouped['receiving_country'].unique():
    df_country = df_grouped[df_grouped['receiving_country'] == country][['startdate', 'total_students']]
    df_country.columns = ['ds', 'y']
    
    # Require at least 24 months of historical data to capture annual seasonality
    if len(df_country) >= 24:
        model = Prophet(yearly_seasonality=True)
        model.fit(df_country)
        
        prediction = model.predict(future_dataframe)
        
        future_prediction = prediction[['ds', 'yhat']].copy()
        
        # Limit lower bound to 0 and round (it makes no sense to predict "half a student" or negative values)
        future_prediction['yhat'] = future_prediction['yhat'].clip(lower=0).round()
        future_prediction['receiving_country'] = country
        
        predictions_list.append(future_prediction)
    else:
        print(f"Insufficient data to forecast {country} (Minimum: 24 months. Found: {len(df_country)})")

# ==========================================
# Save Results
# ==========================================
if predictions_list:
    df_final_predictions = pd.concat(predictions_list)
    
    table_name = 'erasmus_predictions_23_24'
    print(f"\nSaving results to table '{table_name}'...")
    df_final_predictions.to_sql(table_name, engine, if_exists='replace', index=False)
    print("Time Series process completed successfully!")
else:
    print("No forecasts were generated.")
