# import pandas as pd
# import numpy as np
# from catboost import CatBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sentence_transformers import SentenceTransformer
# import pickle

# MODEL_PATH = "catboost_fraud_model.pkl"

# def train_fraud_model(df):
#     # Feature engineering (dates, ratios, embeddings, catégoriques)
#     df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], dayfirst=True)
#     df['incident_date'] = pd.to_datetime(df['incident_date'], dayfirst=True)
#     df['days_since_policy_bind'] = (df['incident_date'] - df['policy_bind_date']).dt.days
#     df['month_incident'] = df['incident_date'].dt.month
#     df['day_of_week_incident'] = df['incident_date'].dt.dayofweek

#     df['claim_per_month'] = df['total_claim_amount'] / (df['months_as_customer'] + 1)
#     df['capital_gain_ratio'] = df['capital-gains'] / (df['total_claim_amount'] + 1)

#     model_text = SentenceTransformer('all-MiniLM-L6-v2')
#     incident_embeddings = np.array([model_text.encode(text) for text in df['incident_type'].astype(str)])
#     for i in range(incident_embeddings.shape[1]):
#         df[f'incident_emb_{i}'] = incident_embeddings[:,i]

#     cat_features = ['policy_state', 'insured_sex', 'insured_education_level', 'auto_make', 'auto_model']
#     y = df['fraud_reported'].map({'Y':1,'N':0})

#     feature_columns = [
#         'months_as_customer', 'age', 'capital-gains', 'capital-loss',
#         'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
#         'number_of_vehicles_involved', 'days_since_policy_bind', 'month_incident',
#         'day_of_week_incident', 'claim_per_month', 'capital_gain_ratio'
#     ]
#     feature_columns += [f'incident_emb_{i}' for i in range(incident_embeddings.shape[1])]
#     feature_columns += cat_features

#     clf = RandomForestClassifier(
#         iterations=500,
#         learning_rate=0.1,
#         depth=6,
#         eval_metric='AUC',
#         verbose=100,
#         class_weights=[1, (y==0).sum()/(y==1).sum()]
#     )
#     clf.fit(df[feature_columns], y, cat_features=cat_features)

#     # Sauvegarder le modèle
#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump((clf, feature_columns, model_text), f)

#     return clf, feature_columns, model_text

# def load_model():
#     try:
#         with open(MODEL_PATH, "rb") as f:
#             clf, feature_columns, model_text = pickle.load(f)
#     except FileNotFoundError:
#         df = pd.read_csv("insurance_claims.csv")
#         clf, feature_columns, model_text = train_fraud_model(df)
#     return clf, feature_columns, model_text
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "fraud_model.pkl"

def train_fraud_model(df):
    # --- Feature engineering ---
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], dayfirst=True)
    df['incident_date'] = pd.to_datetime(df['incident_date'], dayfirst=True)
    df['days_since_policy_bind'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df['month_incident'] = df['incident_date'].dt.month
    df['day_of_week_incident'] = df['incident_date'].dt.dayofweek
    df['claim_per_month'] = df['total_claim_amount'] / (df['months_as_customer'] + 1)
    df['capital_gain_ratio'] = df['capital-gains'] / (df['total_claim_amount'] + 1)

    # --- Embeddings textuelles ---
    model_text = SentenceTransformer('all-MiniLM-L6-v2')
    incident_embeddings = np.array([model_text.encode(text) for text in df['incident_type'].astype(str)])
    for i in range(incident_embeddings.shape[1]):
        df[f'incident_emb_{i}'] = incident_embeddings[:,i]

    # --- Cat features ---
    cat_features = ['policy_state', 'insured_sex', 'insured_education_level', 'auto_make', 'auto_model']
    y = df['fraud_reported'].map({'Y':1,'N':0})

    # --- Features final ---
    feature_columns = [
        'months_as_customer', 'age', 'capital-gains', 'capital-loss',
        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
        'number_of_vehicles_involved', 'days_since_policy_bind', 'month_incident',
        'day_of_week_incident', 'claim_per_month', 'capital_gain_ratio'
    ]
    feature_columns += [f'incident_emb_{i}' for i in range(incident_embeddings.shape[1])]
    feature_columns += cat_features

    # --- Modèle CatBoost ---
    clf = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        eval_metric='AUC',
        verbose=100,
        class_weights=[1, (y==0).sum()/(y==1).sum()]
    )

    clf.fit(df[feature_columns], y, cat_features=cat_features)

    # --- Sauvegarder ---
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((clf, feature_columns, model_text), f)

    return clf, feature_columns, model_text

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            clf, feature_columns, model_text = pickle.load(f)
    except FileNotFoundError:
        df = pd.read_csv("../insurance_claims.csv")
        clf, feature_columns, model_text = train_fraud_model(df)
    return clf, feature_columns, model_text

# -----------------------------
# 0️⃣ Charger les données et modèle
# -----------------------------
df = pd.read_csv("../insurance_claims.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

clf, feature_columns, model_text = load_model()

# -----------------------------
# 1️⃣ Préparer le test set
# -----------------------------
X_test = test_df.copy()
X_test['policy_bind_date'] = pd.to_datetime(X_test['policy_bind_date'], dayfirst=True)
X_test['incident_date'] = pd.to_datetime(X_test['incident_date'], dayfirst=True)
X_test['days_since_policy_bind'] = (X_test['incident_date'] - X_test['policy_bind_date']).dt.days
X_test['month_incident'] = X_test['incident_date'].dt.month
X_test['day_of_week_incident'] = X_test['incident_date'].dt.dayofweek
X_test['claim_per_month'] = X_test['total_claim_amount'] / (X_test['months_as_customer'] + 1)
X_test['capital_gain_ratio'] = X_test['capital-gains'] / (X_test['total_claim_amount'] + 1)

# Embeddings incident_type pour le test
incident_embeddings = np.array([model_text.encode(text) for text in X_test['incident_type'].astype(str)])
for i in range(incident_embeddings.shape[1]):
    X_test[f'incident_emb_{i}'] = incident_embeddings[:,i]

# Ajouter les colonnes manquantes si nécessaire
for col in feature_columns:
    if col not in X_test.columns:
        X_test[col] = 0

y_test = X_test['fraud_reported'].map({'Y':1,'N':0})

# -----------------------------
# 2️⃣ Prédiction
# -----------------------------
y_pred_proba = clf.predict_proba(X_test[feature_columns])[:,1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# -----------------------------
# 3️⃣ Évaluation
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Statistiques globales ===")
print(f"Accuracy : {accuracy:.3f}")
print(f"Précision : {precision:.3f}")
print(f"Rappel : {recall:.3f}")
print(f"F1-score : {f1:.3f}")
print("\n=== Matrice de confusion ===")
print(cm)
