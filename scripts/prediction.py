import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import logging

# Suppress Prophet spammy logs
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.WARNING)

engine = create_engine('postgresql://postgres:MGolRhrRWkSXY6M2@db.znprnevdjfwpgxetgmjl.supabase.co:5432/postgres')

# Migration Prediction
query_migration = """
    SELECT f.startdate, g.countryname AS receiving_country
    FROM f_mobility f
    JOIN d_geography g ON f.receivinggeography = g.geographyid
    WHERE f.startdate IS NOT NULL
"""
df_migration = pd.read_sql(query_migration, engine)

df_migration['startdate'] = pd.to_datetime(df_migration['startdate'], format='%Y%m%d')

# Remove 2020 and 2021
df_migration = df_migration[~(df_migration['startdate'].dt.year.isin([2020, 2021]))]

df_grouped = df_migration.groupby([pd.Grouper(key='startdate', freq='MS'), 'receiving_country']).size().reset_index(name='total_students')

predictions_list = []

# Create manually target dates for 2023 and 2024 (January to December)
target_dates = pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS')
future_dataframe = pd.DataFrame({'ds': target_dates})

for country in df_grouped['receiving_country'].unique():
    df_country = df_grouped[df_grouped['receiving_country'] == country][['startdate', 'total_students']]
    df_country.columns = ['ds', 'y']
    
    if len(df_country) >= 24:
        model = Prophet(yearly_seasonality=True)
        model.fit(df_country)
        
        prediction = model.predict(future_dataframe)
        
        future_prediction = prediction[['ds', 'yhat']].copy()
        
        future_prediction['yhat'] = future_prediction['yhat'].clip(lower=0).round()
        
        future_prediction['receiving_country'] = country
        predictions_list.append(future_prediction)
    else:
        print(f"Insufficient data to predict {country} (Minimum: 24 months)")

if predictions_list:
    df_final_predictions = pd.concat(predictions_list)
    df_final_predictions.to_sql('erasmus_predictions_23_24', engine, if_exists='replace', index=False)
    print("Time series predictions completed.")

# ==========================================
# Data Mining (Opportunity Classification)
# ==========================================
query_mining = """
    SELECT 
        d.age, 
        d.gender, 
        e.educationlevel, 
        g.countryname AS sending_country, 
        f.mobilitydurationdays, 
        f.feweropportunitiesflag 
    FROM f_mobility f
    JOIN d_demographics d ON f.demographicsid = d.demographicsid
    JOIN d_education e ON f.educationid = e.educationid
    JOIN d_geography g ON f.sendinggeography = g.geographyid
"""
df_mining = pd.read_sql(query_mining, engine)

df_mining = df_mining.dropna(subset=['feweropportunitiesflag'])

features = ['age', 'gender', 'educationlevel', 'sending_country', 'mobilitydurationdays']
X = df_mining[features]
y = df_mining['feweropportunitiesflag'].astype(int)

categorical_features = ['gender', 'educationlevel', 'sending_country']
numeric_features = ['age', 'mobilitydurationdays']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fills missing numbers with the median
    ('passthrough', 'passthrough')
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Fills missing text with 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Preprocessing data...")
X_train_processed = preprocessor.fit_transform(X_train)

print("Training Random Forest Model...")
# O verbose=2 mostra o progresso do treino na consola e o n_jobs=-1 acelera usando todos os cores
classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', verbose=2, n_jobs=-1)

# Treino direto (mais otimizado que o loop)
classifier.fit(X_train_processed, y_train)

# Desativar verbose para não encher a consola de mensagens durante a avaliação e predição
classifier.verbose = 0

pipeline_ml = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

print("\n--- Model Evaluation Report ---")
y_pred = pipeline_ml.predict(X_test)
print(classification_report(y_test, y_pred))
print("-------------------------------\n")

df_mining['prob_fewer_opportunities'] = pipeline_ml.predict_proba(X)[:, 1]

df_mining.to_sql('ml_results_opportunities', engine, if_exists='replace', index=False)
print("Data Mining (Random Forest) completed.")