

ClaimsInspector est une application Python pour la **détection de fraudes dans le domaine des assurances**, combinant des modèles d’IA, un workflow automatisé et un dashboard interactif.

---

## 📂 Structure du projet

```

code/
├─ Agents/                 # Modèles et données de fraude
│  ├─ catboost_info/       # Fichiers CatBoost
│  ├─ fraud_model.pkl      # Modèle entraîné
│  └─ fraud_model.py       # Code du modèle
├─ chroma_db/              # Base de données vectorielle
├─ dashboard.py            # Dashboard interactif Streamlit
├─ workflow.py             # Script principal du workflow IA
├─ requirements.txt        # Librairies Python nécessaires
├─ .env                    # Variables d’environnement (non versionné)
└─ .gitignore

````

---

## ⚙️ Installation

1. **Cloner le dépôt et se déplacer dans le dossier du projet :**

```bash
git clone https://github.com/Erijbr/ClaimsInspector.git
cd ClaimsInspector/code
````

2. **Créer un environnement virtuel :**

```bash
python -m venv venv
```

3. **Activer l'environnement :**

* **Windows :**

```powershell
venv\Scripts\activate
```

* **Linux/macOS :**

```bash
source venv/bin/activate
```

4. **Installer les dépendances :**

```bash
pip install -r requirements.txt
```

---

## 🔒 Configuration du `.env`

Créez un fichier `.env` à la racine du projet pour stocker vos clés API et variables sensibles.
Exemple :

```dotenv
GROQ_API_KEY=Votre_Groq_API_Key
AUTRE_VARIABLE=Valeur
```

> Le fichier `.env` n’est pas versionné pour protéger vos informations sensibles.

---

## 💻 Exécution du projet

### 1. Lancer le dashboard Streamlit

```bash
python3 -m streamlit run dashboard.py
```

Le dashboard sera accessible sur `http://localhost:8501`.

### 2. Exécuter le workflow principal

```bash
python3 workflow.py
```

* Ce script effectue le traitement des données, l’entraînement ou le chargement des modèles, et les prédictions.

---

## 🛠️ Dépendances principales

* pandas
* numpy
* scikit-learn
* catboost
* streamlit
* chromadb

Toutes les dépendances sont listées dans `requirements.txt`.



```

---

Si tu veux, je peux te faire **une version encore plus concise et « prête à copier »**, où tu n’as plus qu’à remplacer le `.env` et lancer les commandes Python.  

Veux‑tu que je fasse ça ?
```
