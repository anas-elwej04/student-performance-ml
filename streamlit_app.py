import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------
# 1. Chargement des données
# ---------------------------

@st.cache_data
def load_data():
    mat = pd.read_csv("student-mat.csv", sep=";")
    por = pd.read_csv("student-por.csv", sep=";")
    return mat, por

mat, por = load_data()

st.title("Prédiction de la réussite d'étudiants 🎓")
st.write("Dataset : Student Performance (UCI) – Math vs Portugais")

# Choix de la matière
dataset_choice = st.selectbox("Choisir la matière :", ["Mathématiques", "Portugais"])
df = mat if dataset_choice == "Mathématiques" else por

st.subheader(f"Aperçu des données – {dataset_choice}")
st.dataframe(df.head())

# ---------------------------
# 2. Préparation des données
# ---------------------------

# Création de la variable cible passed
df = df.copy()
df["passed"] = (df["G3"] >= 10).astype(int)

# Affichage du taux de réussite
success_rate = df["passed"].mean()
st.write(f"Taux de réussite dans {dataset_choice} : **{success_rate:.2%}**")

# Sélection des features
features_to_drop = ["G3", "passed"]
X = df.drop(columns=features_to_drop)
y = df["passed"]

# Encodage one-hot
X_encoded = pd.get_dummies(X, drop_first=True)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 3. Entraînement des modèles
# ---------------------------

def train_models():
    models = {}

    # Régression logistique
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    models["Régression logistique"] = (log_reg, True)

    # Arbre de décision
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    models["Arbre de décision"] = (tree_clf, False)

    # KNN
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_scaled, y_train)
    models["KNN (k=5)"] = (knn_clf, True)

    # Naïve Bayes
    nb_clf = GaussianNB()
    nb_clf.fit(X_train_scaled, y_train)
    models["Naïve Bayes"] = (nb_clf, True)

    return models

models = train_models()

# ---------------------------
# 4. Évaluation
# ---------------------------

st.subheader("Performance des modèles")

results = []
for name, (model, use_scaled) in models.items():
    X_te = X_test_scaled if use_scaled else X_test
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Modèle": name,
        "Accuracy": acc,
        "Précision": prec,
        "Rappel": rec,
        "F1-score": f1
    })

results_df = pd.DataFrame(results).set_index("Modèle")
st.table(results_df.style.format("{:.3f}"))

best_model_name = results_df["F1-score"].idxmax()
st.success(f"Meilleur modèle (selon F1-score) : **{best_model_name}**")

# ---------------------------
# 5. Démo de prédiction
# ---------------------------

st.subheader("Tester une prédiction individuelle")

model_to_use_name = st.selectbox("Choisir un modèle pour la prédiction :", list(models.keys()))
model_to_use, use_scaled = models[model_to_use_name]

st.write("Renseigner quelques informations (valeurs simples pour la démo) :")

age = st.slider("Age", 15, 22, 17)
absences = st.slider("Absences", 0, 30, 3)
G1 = st.slider("Note G1", 0, 20, 10)
G2 = st.slider("Note G2", 0, 20, 10)

# On part des moyennes pour les autres colonnes
x_demo = X.mean().to_dict()
x_demo["age"] = age
x_demo["absences"] = absences
x_demo["G1"] = G1
x_demo["G2"] = G2

x_demo_df = pd.DataFrame([x_demo])
x_demo_encoded = pd.get_dummies(x_demo_df, drop_first=True)

# Ré-aligner les colonnes avec X_encoded
x_demo_encoded = x_demo_encoded.reindex(columns=X_encoded.columns, fill_value=0)

if use_scaled:
    x_demo_final = scaler.transform(x_demo_encoded)
else:
    x_demo_final = x_demo_encoded

if st.button("Prédire la réussite"):
    pred = model_to_use.predict(x_demo_final)[0]
    proba = getattr(model_to_use, "predict_proba", None)
    if proba is not None:
        p = model_to_use.predict_proba(x_demo_final)[0][1]
        st.write(f"Probabilité de réussite : **{p:.2%}**")
    if pred == 1:
        st.success("Prédiction : l'étudiant **réussirait** (passed = 1).")
    else:
        st.error("Prédiction : l'étudiant **échouerait** (passed = 0).")
