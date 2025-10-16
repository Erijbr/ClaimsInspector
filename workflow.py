import chromadb
from chromadb.config import Settings
import pandas as pd
import sqlite3
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
from typing_extensions import Annotated as TypedAnnotated
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# -----------------------------
# Chemins et initialisation
# -----------------------------
MODEL_PATH = "./Agents/fraud_model.pkl"
DB_PATH = "claims.db"
DATA_PATH = "insurance_claims.csv"
CHROMA_PATH = "./chroma_db"

# Charger modèle ML et NLP
with open(MODEL_PATH, "rb") as f:
    clf, feature_columns, model_text = pickle.load(f)
print("✅ Modèle chargé depuis", MODEL_PATH)

# -----------------------------
# ChromaDB avec PersistentClient
# -----------------------------
print(f"\n🔧 Initialisation ChromaDB dans {CHROMA_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

existing_collections = [c.name for c in chroma_client.list_collections()]
print(f"📚 Collections existantes: {existing_collections}")

collection_name = "claims_embeddings"
try:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"✅ Collection '{collection_name}' récupérée - {collection.count()} documents")
except Exception as e:
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Embeddings des sinistres d'assurance"}
    )
    print(f"✅ Collection '{collection_name}' créée")

print(f"📊 Nombre de documents au départ: {collection.count()}\n")

# Charger les données
df = pd.read_csv(DATA_PATH)
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

# -----------------------------
# Wrappers et Agents
# -----------------------------
def wrap_agent(agent_func):
    def wrapped(state):
        return agent_func(state)
    return wrapped

def claim_intake_agent(inputs):
    print("\n[1/7] 📥 Claim Intake Agent")
    df_sample = test_df.sample(n=10).copy()
    df_sample["state"] = "EN_ATTENTE"
    
    conn = sqlite3.connect(DB_PATH)
    df_sample.to_sql("claims", conn, if_exists="append", index=False)
    conn.close()
    
    print(f"   ✓ {len(df_sample)} sinistres ingérés")
    return {"claims": df_sample.to_dict(orient="records")}

def policy_validation_agent(inputs):
    print("\n[2/7] 🔍 Policy Validation Agent")
    claims = inputs["claims"]
    validated = 0
    rejected = 0
    
    for c in claims:
        if not c.get("policy_number"):
            c["status"] = "ERROR"
            c["state"] = "REJETÉ"
            rejected += 1
        else:
            c["status"] = "VALIDATED"
            c["state"] = "EN_COURS"
            validated += 1
    
    print(f"   ✓ Validés: {validated}, Rejetés: {rejected}")
    return {"claims": claims}

def fraud_scoring_agent(inputs):
    print("\n[3/7] 🤖 Fraud Scoring Agent")
    claims = inputs["claims"]
    scored = 0
    
    for claim in claims:
        if claim.get("status") != "VALIDATED":
            continue
            
        X_claim = pd.DataFrame([claim])
        X_claim['policy_bind_date'] = pd.to_datetime(X_claim['policy_bind_date'], errors='coerce')
        X_claim['incident_date'] = pd.to_datetime(X_claim['incident_date'], errors='coerce')
        X_claim['days_since_policy_bind'] = (X_claim['incident_date'] - X_claim['policy_bind_date']).dt.days
        X_claim['month_incident'] = X_claim['incident_date'].dt.month
        X_claim['day_of_week_incident'] = X_claim['incident_date'].dt.dayofweek
        X_claim['claim_per_month'] = X_claim['total_claim_amount'] / (X_claim['months_as_customer'] + 1)
        X_claim['capital_gain_ratio'] = X_claim['capital-gains'] / (X_claim['total_claim_amount'] + 1)

        embedding = model_text.encode([X_claim.iloc[0]['incident_type']])[0]
        for i in range(len(embedding)):
            X_claim[f'incident_emb_{i}'] = embedding[i]

        for col in feature_columns:
            if col not in X_claim.columns:
                X_claim[col] = 0

        claim['fraud_score'] = float(clf.predict_proba(X_claim[feature_columns])[:,1][0])
        scored += 1
    
    print(f"   ✓ {scored} sinistres scorés")
    return {"claims": claims}

def nlp_vector_agent(inputs):
    print("\n[4/7] 🧠 NLP Vector Agent (k-NN)")
    claims = inputs["claims"]
    current_count = collection.count()
    k_neighbors = 5
    SIMILARITY_THRESHOLD = 0.75  # Seuil pour filtrer les voisins trop éloignés
    
    print(f"   📊 ChromaDB contient actuellement {current_count} documents")
    
    for claim in claims:
        text = f"{claim.get('incident_type','')} {claim.get('collision_type','')} " \
               f"{claim.get('incident_severity','')} {claim.get('authorities_contacted','')}"
        embedding = model_text.encode([text])[0].astype('float32')

        if current_count == 0:
            claim['nlp_score'] = 0.5
            print(f"   ⚠️  Collection vide - Score neutre (0.5) pour {claim.get('policy_number')}")
        else:
            try:
                results = collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=min(k_neighbors, current_count),
                    include=['embeddings', 'metadatas', 'distances']
                )
                
                if results and results.get('embeddings') and len(results['embeddings']) > 0 and len(results['embeddings'][0]) > 0:
                    closest_embs = np.array(results['embeddings'][0])
                    metadatas = results['metadatas'][0]
                    
                    similarities = np.dot(closest_embs, embedding) / (
                        np.linalg.norm(closest_embs, axis=1) * np.linalg.norm(embedding) + 1e-10
                    )
                    similarities = np.clip(similarities, 0, 1)
                    
                    # Calculer le score pondéré
                    fraud_weight = 0.0
                    total_weight = 0.0
                    fraud_count = 0
                    not_fraud_count = 0
                    
                    for i, metadata in enumerate(metadatas):
                        similarity = similarities[i]
                        
                        # Filtrer les voisins trop éloignés
                        if similarity < SIMILARITY_THRESHOLD:
                            continue
                        
                        decision = metadata.get('decision', 'Not Fraud')
                        total_weight += similarity
                        
                        if 'Fraud' in decision and 'Not Fraud' not in decision:
                            fraud_weight += similarity
                            fraud_count += 1
                        else:
                            not_fraud_count += 1
                    
                    if total_weight > 0:
                        claim['nlp_score'] = float(fraud_weight / total_weight)
                    else:
                        claim['nlp_score'] = 0.5
                    
                    avg_similarity = float(np.mean(similarities))
                    print(f"   ✓ Policy {claim.get('policy_number')} - NLP score: {claim['nlp_score']:.3f}")
                    print(f"      └─ {fraud_count + not_fraud_count} voisins valides: {fraud_count} Fraud, {not_fraud_count} Not Fraud (sim moy: {avg_similarity:.3f})")
                    
                else:
                    claim['nlp_score'] = 0.5
                    print(f"   ⚠️  Pas de voisins trouvés pour {claim.get('policy_number')}")
                    
            except Exception as e:
                print(f"   ❌ Erreur query pour {claim.get('policy_number')}: {e}")
                claim['nlp_score'] = 0.5
    
    return {"claims": claims}

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé Groq
groq_api_key = os.getenv("GROQ_API_KEY")

# Créer l'instance ChatGroq
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)




def decision_explanation_agent(inputs):
    print("\n[5/7] ⚖️  Decision Explanation Agent")
    claims = inputs["claims"]
    fraud_count = 0
    legit_count = 0
    needs_human = 0
    
    for claim in claims:
        fraud_score = claim.get('fraud_score', 0.5)
        nlp_score = claim.get('nlp_score', 0.5)
        total_score = (fraud_score + nlp_score) / 2
        claim['total_score'] = total_score

        # Décision selon score
        if total_score > 0.7:
            claim['requires_human'] = False
            claim['decision'] = "Fraud - Auto"
            claim['state'] = "TRAITÉ_AUTO"
            fraud_count += 1
        elif total_score < 0.3:
            claim['requires_human'] = False
            claim['decision'] = "Not Fraud - Auto"
            claim['state'] = "TRAITÉ_AUTO"
            legit_count += 1
        else:
            claim['requires_human'] = True
            claim['decision'] = "Needs Human Approval"
            claim['state'] = "EN_ATTENTE_APPROBATION"
            needs_human += 1

        # Générer explication LLM
        # prompt = (
        #     f"Vous êtes un agent d'assurance analysant un sinistre. "
        #     f"Un modèle ML (CatBoost + NLP k-NN) a analysé ce sinistre.\n\n"
        #     f"Détails : Type={claim.get('incident_type')}, Collision={claim.get('collision_type')}, "
        #     f"Gravité={claim.get('incident_severity')}, Montant={claim.get('total_claim_amount')}\n"
        #     f"Score total : {total_score:.2f} (fraud_score={fraud_score:.2f}, nlp_score={nlp_score:.2f})\n"
        #     f"Décision : {claim['decision']}\n\n"
        #     f"Expliquez en 2-3 phrases pourquoi cette décision, sans mentionner les scores numériques. "
        #     f"Mentionnez les facteurs clés (type incident, gravité, historique similaire, etc.)."
        # )
        
        # try:
        #     claim['llm_explanation'] = llm.invoke(prompt).content
        # except Exception as e:
        #     claim['llm_explanation'] = f"Erreur LLM : {e}"
    
    print(f"   ✓ Fraudes auto: {fraud_count}, Légitimes auto: {legit_count}, À vérifier: {needs_human}")
    return {"claims": claims}

def ensure_columns_exist(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(claims)")
    existing_cols = [col[1] for col in cursor.fetchall()]
    new_cols = ['total_score', 'fraud_score', 'nlp_score', 'decision', 'requires_human', 'llm_explanation']
    
    for col in new_cols:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE claims ADD COLUMN {col} TEXT")
    conn.commit()

def update_chroma_db(claim):
    """Ajouter ou mettre à jour un sinistre dans ChromaDB"""
    text = f"{claim.get('incident_type','')} {claim.get('collision_type','')} " \
           f"{claim.get('incident_severity','')} {claim.get('authorities_contacted','')}"
    embedding = model_text.encode([text])[0].astype('float32')
    policy_number = str(claim.get("policy_number"))

    try:
        existing = collection.get(ids=[policy_number])
        if existing['ids']:
            collection.delete(ids=[policy_number])
    except Exception:
        pass

    try:
        collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{
                "policy_number": policy_number,
                "decision": str(claim.get("decision", "")),
                "fraud_score": str(claim.get("fraud_score", "")),
                "total_score": str(claim.get("total_score", "")),
                "incident_type": str(claim.get("incident_type", ""))
            }],
            ids=[policy_number]
        )
    except Exception as e:
        print(f"   ❌ Erreur ChromaDB pour {policy_number}: {e}")

def notification_agent(inputs):
    print("\n[6/7] 📢 Notification Agent")
    claims = inputs["claims"]
    
    conn = sqlite3.connect(DB_PATH)
    ensure_columns_exist(conn)
    cursor = conn.cursor()
    
    auto_added = 0
    pending_updated = 0
    
    for claim in claims:
        # Mettre à jour SQLite pour TOUS les sinistres
        cursor.execute("""
            UPDATE claims
            SET state=?, fraud_score=?, nlp_score=?, total_score=?, decision=?, requires_human=?, llm_explanation=?
            WHERE policy_number=?
        """, (
            claim.get('state'),
            str(claim.get('fraud_score', '')),
            str(claim.get('nlp_score', '')),
            str(claim.get('total_score', '')),
            str(claim.get('decision', '')),
            int(claim.get('requires_human', 0)),
            str(claim.get('llm_explanation', '')),
            claim.get('policy_number')
        ))
        
        # Ajouter dans ChromaDB SEULEMENT les décisions automatiques finales
        if claim.get('state') == "TRAITÉ_AUTO":
            update_chroma_db(claim)
            auto_added += 1
        else:
            pending_updated += 1
    
    conn.commit()
    conn.close()
    
    print(f"   ✓ SQLite: {auto_added + pending_updated} sinistres mis à jour")
    print(f"   ✓ ChromaDB: {auto_added} décisions auto ajoutées, {pending_updated} en attente validation")
    print(f"   💾 Total ChromaDB: {collection.count()} documents")
    
    return {"claims": claims}

def dashboard_agent(inputs):
    print("\n[7/7] 📊 Dashboard Agent")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM claims", conn)
    conn.close()
    print(f"   ✓ {len(df)} sinistres disponibles dans le dashboard")
    return inputs

def merge_claims(left, right):
    merged = {c.get('policy_number'): c for c in left} if left else {}
    for claim in right:
        policy_num = claim.get('policy_number')
        if policy_num in merged:
            merged[policy_num].update(claim)
        else:
            merged[policy_num] = claim
    return list(merged.values())

class ClaimState(TypedDict):
    claims: TypedAnnotated[list, merge_claims]

def create_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("intake", wrap_agent(claim_intake_agent))
    graph.add_node("validation", wrap_agent(policy_validation_agent))
    graph.add_node("fraud_scoring", wrap_agent(fraud_scoring_agent))
    graph.add_node("nlp_vector", wrap_agent(nlp_vector_agent))
    graph.add_node("decision", wrap_agent(decision_explanation_agent))
    graph.add_node("notification", wrap_agent(notification_agent))
    graph.add_node("dashboard", wrap_agent(dashboard_agent))

    graph.set_entry_point("intake")

    graph.add_edge("intake", "validation")
    graph.add_edge("validation", "fraud_scoring")
    graph.add_edge("validation", "nlp_vector")
    graph.add_edge("fraud_scoring", "decision")
    graph.add_edge("nlp_vector", "decision")
    graph.add_edge("decision", "notification")
    graph.add_edge("decision", "dashboard")
    graph.add_edge("notification", END)
    graph.add_edge("dashboard", END)

    return graph.compile()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 WORKFLOW MULTI-AGENT - DÉTECTION DE FRAUDE")
    print("=" * 60)
    
    app = create_graph()
    initial_state = {"claims": []}
    
    try:
        result = app.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("✅ WORKFLOW TERMINÉ AVEC SUCCÈS")
        print("=" * 60)
        print(f"📈 {len(result.get('claims', []))} sinistres traités")
        print(f"💾 Documents ChromaDB: {collection.count()}")
        
        auto_fraud = sum(1 for c in result.get('claims', []) if c.get('decision') == 'Fraud - Auto')
        auto_legit = sum(1 for c in result.get('claims', []) if c.get('decision') == 'Not Fraud - Auto')
        needs_human = sum(1 for c in result.get('claims', []) if c.get('requires_human'))
        
        print(f"🔍 Fraudes auto: {auto_fraud}, Légitimes auto: {auto_legit}, À vérifier: {needs_human}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()