import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Database Connection and Extraction
# ==========================================
engine = create_engine('postgresql://postgres:MGolRhrRWkSXY6M2@db.znprnevdjfwpgxetgmjl.supabase.co:5432/postgres')

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
print("Extracting data from database...")
df_mining = pd.read_sql(query_mining, engine)

df_mining = df_mining.dropna(subset=['feweropportunitiesflag'])

features = ['age', 'gender', 'educationlevel', 'sending_country', 'mobilitydurationdays']
X = df_mining[features]
y = df_mining['feweropportunitiesflag'].astype(int)

# ==========================================
# Pre-processing Setup
# ==========================================
numeric_features = ['age', 'mobilitydurationdays']
categorical_features = ['gender', 'educationlevel', 'sending_country']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ==========================================
# TVT Split 80/10/10
# ==========================================
print("Splitting data into Train, Validation, and Test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ==========================================
# Pre-processing Execution
# ==========================================
print("Preprocessing data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

models = {
    "Random Forest (Black-box)": RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1),
    "Decision Tree (Interpretable)": DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=6),
    "Logistic Regression (Interpretable)": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
}

# ==========================================
# Model Evaluation (Validation Set)
# ==========================================
print("\n--- Model Evaluation on Validation Set (10%) ---")

best_model_name = ""
best_f1 = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    model.fit(X_train_processed, y_train)
    
    y_val_pred = model.predict(X_val_processed)
    
    f1 = f1_score(y_val, y_val_pred, average='macro')
    
    print(f"  -> F1-Macro (Validation): {f1:.4f} (Time: {time.time() - start_time:.1f}s)")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

# ==========================================
# Final Evaluation (Test Set)
# ==========================================
print(f"\n--- Selected Model: {best_model_name} ---")
final_model = models[best_model_name]

print("\n--- Classification Report on Test Set (10%) ---")
y_test_pred = final_model.predict(X_test_processed)
print(classification_report(y_test, y_test_pred))

# ==========================================
# Interpretability
# ==========================================
if hasattr(final_model, 'feature_importances_'):
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n--- Top 10 Most Important Features ---")
    for i in range(min(10, len(importances))):
        print(f"{i+1}. {all_feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
else:
    print("\n[The selected model does not support direct feature_importances_ (e.g., Logistic Regression)]")

# ==========================================
# Generate Predictions and Save to Database
# ==========================================
print("\nApplying the final pipeline to the entire dataset to extract probabilities...")

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', final_model)
])

full_pipeline.fit(X_train, y_train)

df_mining['prob_fewer_opportunities'] = full_pipeline.predict_proba(X)[:, 1]

table_name = 'ml_results_opportunities'
print(f"Saving results to table '{table_name}'...")
df_mining.to_sql(table_name, engine, if_exists='replace', index=False)

print("Classification process completed successfully!")