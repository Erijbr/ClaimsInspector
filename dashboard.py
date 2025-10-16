# # # import sqlite3
# # # import pandas as pd
# # # import streamlit as st
# # # import time

# # # DB_PATH = "claims.db"

# # # st.set_page_config(page_title="Tableau de bord des sinistres", layout="wide")
# # # st.title("📊 Tableau de bord des sinistres")

# # # def load_data():
# # #     conn = sqlite3.connect(DB_PATH)
# # #     df = pd.read_sql("SELECT * FROM claims", conn)
# # #     conn.close()
# # #     return df

# # # def color_rows(row):
# # #     """Colorie toute la ligne selon l'état du sinistre"""
# # #     if row['state'] == "EN_ATTENTE_APPROBATION":
# # #         return ['background-color: orange; color: black'] * len(row)
# # #     elif row['state'] == "TERMINÉ":
# # #         return ['background-color: lightgreen; color: black'] * len(row)
# # #     elif row['state'] == "EN_ATTENTE":
# # #         return ['background-color: yellow; color: black'] * len(row)
# # #     else:
# # #         return [''] * len(row)

# # # # Auto-refresh toutes les X secondes


# # # placeholder = st.empty()

# # # while True:
# # #     df = load_data()
# # #     with placeholder.container():
# # #         st.dataframe(df.style.apply(color_rows, axis=1))
# # #     time.sleep(60)



# # import sqlite3
# # import pandas as pd
# # import streamlit as st
# # import time

# # DB_PATH = "claims.db"

# # st.set_page_config(page_title="Tableau de bord des sinistres", layout="wide")
# # st.title("📊 Tableau de bord des sinistres")

# # # -----------------------------
# # # Fonctions utilitaires
# # # -----------------------------
# # def load_data():
# #     conn = sqlite3.connect(DB_PATH)
# #     df = pd.read_sql("SELECT * FROM claims", conn)
# #     conn.close()
# #     return df

# # def update_claim_decision(policy_number, approve=True):
# #     conn = sqlite3.connect(DB_PATH)
# #     cursor = conn.cursor()
# #     cursor.execute("SELECT decision FROM claims WHERE policy_number=?", (policy_number,))
# #     row = cursor.fetchone()
# #     if not row:
# #         conn.close()
# #         return
# #     system_decision = row[0]

# #     if approve:
# #         final_decision = system_decision + " (validé humain)"
# #     else:
# #         final_decision = "Fraud (corrigé humain)" if system_decision != "Fraud" else "Not Fraud (corrigé humain)"

# #     cursor.execute("""
# #         UPDATE claims
# #         SET state = ?, decision = ?
# #         WHERE policy_number = ?
# #     """, ("TERMINÉ", final_decision, policy_number))
# #     conn.commit()
# #     conn.close()

# # def color_state(state):
# #     if state == "EN_ATTENTE_APPROBATION":
# #         return "🟧 EN ATTENTE"
# #     elif state == "TERMINÉ":
# #         return "🟩 TERMINÉ"
# #     elif state == "EN_ATTENTE":
# #         return "🟨 EN TRAITEMENT"
# #     else:
# #         return state

# # # -----------------------------
# # # Tableau avec expander
# # # -----------------------------
# # df = load_data()

# # for idx, row in df.iterrows():
# #     with st.expander(f"🆔 {row['policy_number']} — {row['incident_type']} — {color_state(row['state'])}"):
# #         st.write(f"**Décision système :** {row['decision']}")
# #         st.write(f"**État :** {row['state']}")
        
# #         # Afficher toutes les colonnes en détail
# #         st.write("**Détails complets :**")
# #         st.json(row.to_dict())

# #         # Boutons seulement si en attente d'approbation
# #         if row['state'] == "EN_ATTENTE_APPROBATION":
# #             col1, col2 = st.columns([1, 1])
# #             with col1:
# #                 if st.button("✅ Approuver", key=f"approve_{idx}"):
# #                     update_claim_decision(row['policy_number'], approve=True)
# #                     st.success(f"Sinistre {row['policy_number']} approuvé ✅")
# #                     time.sleep(1)
# #                     st.experimental_rerun()
# #             with col2:
# #                 if st.button("❌ Refuser", key=f"reject_{idx}"):
# #                     update_claim_decision(row['policy_number'], approve=False)
# #                     st.warning(f"Sinistre {row['policy_number']} rejeté ❌")
# #                     time.sleep(1)
# #                     st.experimental_rerun()

# # # -----------------------------
# # # Rafraîchir manuellement
# # # -----------------------------
# # st.markdown("---")
# # if st.button("🔄 Rafraîchir les données"):
# #     st.experimental_rerun()
# # # 


# # import sqlite3
# # import pandas as pd
# # import streamlit as st
# # import time

# # # Importer la fonction update_chroma_db
# # from workflow import update_chroma_db  # ⚠️ Remplace par le vrai fichier si nécessaire

# # DB_PATH = "claims.db"

# # st.set_page_config(page_title="Tableau de bord des sinistres", layout="wide")
# # st.title("📊 Tableau de bord des sinistres")

# # # -----------------------------
# # # Fonctions utilitaires
# # # -----------------------------
# # def load_data():
# #     conn = sqlite3.connect(DB_PATH)
# #     df = pd.read_sql("SELECT * FROM claims", conn)
# #     conn.close()
# #     return df

# # def update_claim_decision(policy_number, approve=True):
# #     """Met à jour la décision et le state dans SQLite, puis ChromaDB si TERMINÉ"""
# #     conn = sqlite3.connect(DB_PATH)
# #     cursor = conn.cursor()
# #     cursor.execute("SELECT * FROM claims WHERE policy_number=?", (policy_number,))
# #     row = cursor.fetchone()
# #     if not row:
# #         conn.close()
# #         return
# #     columns = [desc[0] for desc in cursor.description]
# #     claim = dict(zip(columns, row))

# #     system_decision = claim['decision']

# #     if approve:
# #         claim['decision'] = system_decision + " (validé humain)"
# #     else:
# #         claim['decision'] = "Fraud (corrigé humain)" if system_decision != "Fraud" else "Not Fraud (corrigé humain)"

# #     claim['state'] = "TERMINÉ"

# #     # Mettre à jour SQLite
# #     cursor.execute("""
# #         UPDATE claims
# #         SET state = ?, decision = ?
# #         WHERE policy_number = ?
# #     """, (claim['state'], claim['decision'], policy_number))
# #     conn.commit()
# #     conn.close()

# #     # Mettre à jour directement ChromaDB
# #     update_chroma_db(claim)


# # def color_state(state):
# #     if state == "EN_ATTENTE_APPROBATION":
# #         return "🟧 EN ATTENTE"
# #     elif state == "TERMINÉ":
# #         return "🟩 TERMINÉ"
# #     elif state == "EN_ATTENTE":
# #         return "🟨 EN TRAITEMENT"
# #     else:
# #         return state

# # # -----------------------------
# # # Tableau avec expander
# # # -----------------------------
# # df = load_data()

# # for idx, row in df.iterrows():
# #     with st.expander(f"🆔 {row['policy_number']} — {row['incident_type']} — {color_state(row['state'])}"):
# #         st.write(f"**Décision système :** {row['decision']}")
# #         st.write(f"**État :** {row['state']}")

# #         # Afficher toutes les colonnes en détail
# #         st.write("**Détails complets :**")
# #         st.json(row.to_dict())

# #         # Boutons seulement si en attente d'approbation
# #         if row['state'] == "EN_ATTENTE_APPROBATION":
# #             col1, col2 = st.columns([1, 1])
# #             with col1:
# #                 if st.button("✅ Approuver", key=f"approve_{idx}"):
# #                     update_claim_decision(row['policy_number'], approve=True)
# #                     st.success(f"Sinistre {row['policy_number']} approuvé ✅ et ajouté dans ChromaDB")
# #                     time.sleep(1)
# #                     st.experimental_rerun()
# #             with col2:
# #                 if st.button("❌ Refuser", key=f"reject_{idx}"):
# #                     update_claim_decision(row['policy_number'], approve=False)
# #                     st.warning(f"Sinistre {row['policy_number']} rejeté ❌ et ajouté dans ChromaDB")
# #                     time.sleep(1)
# #                     st.experimental_rerun()

# # # -----------------------------
# # # Rafraîchir manuellement
# # # -----------------------------
# # st.markdown("---")
# # if st.button("🔄 Rafraîchir les données"):
# #     st.experimental_rerun()



# import sqlite3
# import pandas as pd
# import streamlit as st
# import time
# from workflow import update_chroma_db  # ⚠️ à adapter si nécessaire

# DB_PATH = "claims.db"

# st.set_page_config(page_title="Tableau de bord des sinistres", layout="wide")
# st.title("📊 Tableau de bord des sinistres")

# # -----------------------------
# # Fonctions utilitaires
# # -----------------------------
# def load_data():
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql("SELECT * FROM claims", conn)
#     conn.close()
#     return df

# def update_claim_decision(policy_number, approve=True):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM claims WHERE policy_number=?", (policy_number,))
#     row = cursor.fetchone()
#     if not row:
#         conn.close()
#         return
#     columns = [desc[0] for desc in cursor.description]
#     claim = dict(zip(columns, row))

#     system_decision = claim['decision']

#     if approve:
#         claim['decision'] = system_decision + " (validé humain)"
#     else:
#         claim['decision'] = "Fraud (corrigé humain)" if system_decision != "Fraud" else "Not Fraud (corrigé humain)"

#     claim['state'] = "TERMINÉ"

#     cursor.execute("""
#         UPDATE claims
#         SET state = ?, decision = ?
#         WHERE policy_number = ?
#     """, (claim['state'], claim['decision'], policy_number))
#     conn.commit()
#     conn.close()

#     update_chroma_db(claim)


# def color_state_html(state):
#     if state == "EN_ATTENTE_APPROBATION":
#         return "<span style='background-color:#FFA500;color:white;padding:3px 8px;border-radius:6px;'>🟧 EN ATTENTE</span>"
#     elif state == "TERMINÉ":
#         return "<span style='background-color:#00C851;color:white;padding:3px 8px;border-radius:6px;'>🟩 TERMINÉ</span>"
#     elif state == "EN_ATTENTE":
#         return "<span style='background-color:#FFD700;color:black;padding:3px 8px;border-radius:6px;'>🟨 EN TRAITEMENT</span>"
#     else:
#         return state

# # -----------------------------
# # Affichage du tableau
# # -----------------------------
# df = load_data()

# # Ajout d’une colonne colorée pour l’état
# df['État'] = df['state'].apply(color_state_html)
# df['Décision système'] = df['decision']

# # Colonnes visibles dans le tableau
# colonnes_affichees = ['policy_number', 'incident_type', 'Décision système', 'État']
# df_display = df[colonnes_affichees].copy()

# # Rendu HTML pour les couleurs
# st.markdown("### 📋 Liste des sinistres")
# st.markdown(
#     df_display.to_html(escape=False, index=False),
#     unsafe_allow_html=True
# )

# # -----------------------------
# # Actions sur chaque ligne
# # -----------------------------
# st.markdown("---")
# st.subheader("⚙️ Actions manuelles")

# for idx, row in df.iterrows():
#     if row['state'] == "EN_ATTENTE_APPROBATION":
#         st.markdown(f"### 🆔 {row['policy_number']} — {row['incident_type']}")
#         col1, col2 = st.columns([1, 1])
#         with col1:
#             if st.button("✅ Approuver", key=f"approve_{idx}"):
#                 update_claim_decision(row['policy_number'], approve=True)
#                 st.success(f"Sinistre {row['policy_number']} approuvé ✅")
#                 time.sleep(1)
#                 st.experimental_rerun()
#         with col2:
#             if st.button("❌ Refuser", key=f"reject_{idx}"):
#                 update_claim_decision(row['policy_number'], approve=False)
#                 st.warning(f"Sinistre {row['policy_number']} rejeté ❌")
#                 time.sleep(1)
#                 st.experimental_rerun()

# # -----------------------------
# # Bouton de rafraîchissement
# # -----------------------------
# st.markdown("---")
# if st.button("🔄 Rafraîchir les données"):
#     st.experimental_rerun()
import sqlite3
import pandas as pd
import streamlit as st
import time
from workflow import update_chroma_db  # ⚠️ Adapter le chemin si besoin

DB_PATH = "claims.db"

st.set_page_config(page_title="Tableau de bord des sinistres", layout="wide")
st.title("📊 Tableau de bord des sinistres")

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM claims", conn)
    conn.close()
    return df

def update_claim_decision(policy_number, approve=True):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM claims WHERE policy_number=?", (policy_number,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return
    columns = [desc[0] for desc in cursor.description]
    claim = dict(zip(columns, row))

    system_decision = claim['decision']
    claim['decision'] = (
        system_decision + " (validé humain)" if approve
        else "Fraud (corrigé humain)" if system_decision != "Fraud"
        else "Not Fraud (corrigé humain)"
    )
    claim['state'] = "TERMINÉ"

    cursor.execute("""
        UPDATE claims
        SET state = ?, decision = ?
        WHERE policy_number = ?
    """, (claim['state'], claim['decision'], policy_number))
    conn.commit()
    conn.close()

    update_chroma_db(claim)

def color_state(state):
    colors = {
        "EN_ATTENTE_APPROBATION": "🟧 EN ATTENTE",
        "EN_ATTENTE": "🟨 EN TRAITEMENT",
        "TERMINÉ": "🟩 TERMINÉ",
        "TRAITÉ_AUTO": "🟦 TRAITÉ AUTO",
    }
    return colors.get(state, state)

# -----------------------------
# Chargement et affichage des données
# -----------------------------
df = load_data()
df["État"] = df["state"].apply(color_state)

# Afficher toutes les colonnes
st.markdown("### 🔍 Liste complète des sinistres")
st.dataframe(df, use_container_width=True)

# -----------------------------
# Sélection d'un sinistre
# -----------------------------
policy_numbers = df["policy_number"].tolist()
selected_policy = st.selectbox("Sélectionner un sinistre", [""] + policy_numbers)

if selected_policy:
    row = df[df["policy_number"] == selected_policy].iloc[0]
    st.markdown("---")
    st.markdown(f"### ⚙️ Action sur le sinistre **{row['policy_number']}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approuver"):
            update_claim_decision(row["policy_number"], approve=True)
            st.success(f"Sinistre {row['policy_number']} approuvé ✅")
            time.sleep(1)
            st.experimental_rerun()
    with col2:
        if st.button("❌ Refuser"):
            update_claim_decision(row["policy_number"], approve=False)
            st.warning(f"Sinistre {row['policy_number']} rejeté ❌")
            time.sleep(1)
            st.experimental_rerun()

# -----------------------------
# Rafraîchir les données
# -----------------------------
st.markdown("---")
if st.button("🔄 Rafraîchir les données"):
    st.experimental_rerun()
