# test_fraud_model.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
import pickle

MODEL_PATH = "../fraud_model.pkl"
DATA_PATH = "../insurance_claims.csv"

# -----------------------------
# Charger ou entraîner le modèle
# -----------------------------
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            clf, feature_columns, model_text = pickle.load(f)
    except FileNotFoundError:
        df = pd.read_csv(DATA_PATH)
        clf, feature_columns, model_text = train_fraud_model(df)
    return clf, feature_columns, model_text

# -----------------------------
# Préparer les features
# -----------------------------
def prepare_features(df, model_text):
    df = df.copy()
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], dayfirst=True)
    df['incident_date'] = pd.to_datetime(df['incident_date'], dayfirst=True)
    df['days_since_policy_bind'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df['month_incident'] = df['incident_date'].dt.month
    df['day_of_week_incident'] = df['incident_date'].dt.dayofweek
    df['claim_per_month'] = df['total_claim_amount'] / (df['months_as_customer'] + 1)
    df['capital_gain_ratio'] = df['capital-gains'] / (df['total_claim_amount'] + 1)

    # Embeddings
    incident_embeddings = np.array([model_text.encode(str(text)) for text in df['incident_type']])
    for i in range(incident_embeddings.shape[1]):
        df[f'incident_emb_{i}'] = incident_embeddings[:, i]

    return df

# -----------------------------
# Tester le modèle
# -----------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def test_model():
    # Charger dataset et modèle
    df = pd.read_csv(DATA_PATH)
    clf, feature_columns, model_text = load_model()

    # Prendre 20% aléatoire
    test_df = df.sample(frac=0.2, random_state=42)
    test_df = prepare_features(test_df, model_text)

    # Label réel
    y_true = test_df['fraud_reported'].map({'Y':1,'N':0}).values

    # Prédiction
    y_pred = clf.predict(test_df[feature_columns])
    y_proba = clf.predict_proba(test_df[feature_columns])[:,1]

    # Affichage des résultats
    results = test_df[['policy_number']].copy()
    results['fraud_pred'] = y_pred
    results['fraud_proba'] = y_proba
    results['fraud_true'] = y_true

    print("=== Résultats sur 20% du dataset ===")
    print(results.head(20))

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Statistiques globales ===")
    print(f"Total tests : {len(results)}")
    print(f"Précision : {prec:.3f}")
    print(f"Rappel : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"Accuracy : {acc:.3f}")

    print("\n=== Matrice de confusion ===")
    print(cm)

if __name__ == "__main__":
    test_model()
