import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

st.set_page_config(page_title="Détection d'attaques DDoS", page_icon="🛡️", layout="wide")

DOSSIER = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def charger_modele():
    model = joblib.load(os.path.join(DOSSIER, "model.pkl"))
    scaler = joblib.load(os.path.join(DOSSIER, "scaler.pkl"))
    features = joblib.load(os.path.join(DOSSIER, "features.pkl"))
    return model, scaler, features


@st.cache_data
def charger_exemples():
    chemin = os.path.join(DOSSIER, "exemples.json")
    if os.path.exists(chemin):
        with open(chemin) as f:
            return json.load(f)
    return {}


model, scaler, FEATURES = charger_modele()
exemples = charger_exemples()

AIDE = {
    "Dst Port": "Port de destination (80 = HTTP, 443 = HTTPS)",
    "Flow Duration": "Durée du flux (microsecondes)",
    "Tot Fwd Pkts": "Nombre de paquets envoyés vers le serveur",
    "Flow Pkts/s": "Paquets par seconde",
    "Flow Byts/s": "Octets par seconde",
    "Fwd Seg Size Min": "Taille minimale de segment (sens aller)",
    "Init Fwd Win Byts": "Fenêtre TCP initiale (sens aller)",
    "Init Bwd Win Byts": "Fenêtre TCP initiale (sens retour)",
    "Fwd Pkt Len Mean": "Taille moyenne des paquets aller",
    "Fwd Pkt Len Max": "Taille maximale des paquets aller",
    "Fwd Pkt Len Std": "Écart-type de la taille des paquets aller",
    "Fwd Header Len": "Total des en-têtes (sens aller)",
    "Bwd Header Len": "Total des en-têtes (sens retour)",
    "Fwd Act Data Pkts": "Paquets aller contenant des données",
    "Flow IAT Min": "Plus petit délai entre deux paquets",
}

for f in FEATURES:
    st.session_state.setdefault(f"in_{f}", 0.0)


def charger_exemple(nom):
    for f in FEATURES:
        st.session_state[f"in_{f}"] = float(exemples[nom][f])


def predire(df, seuil):
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    nb_corriges = int(X.isna().any(axis=1).sum())
    X = X.fillna(0)
    proba = model.predict_proba(scaler.transform(X))[:, 1]
    pred = (proba >= seuil).astype(int)
    return proba, pred, nb_corriges


with st.sidebar:
    st.title("🛡️ Détection DDoS")
    st.write("Modèle : **Random Forest** entraîné sur des flux réseau (Normal / DDoS).")
    seuil = st.slider("Seuil de décision", 0.10, 0.90, 0.50, 0.05,
                      help="Baisser le seuil détecte plus d'attaques (rappel ↑) "
                           "mais crée plus de fausses alertes.")
    st.caption(f"{len(FEATURES)} caractéristiques utilisées")

st.title("Détection des attaques DDoS par Machine Learning")

tab1, tab2, tab3 = st.tabs(["Saisie manuelle", "Fichier CSV", "Performances du modèle"])

with tab1:
    st.subheader("Saisir les caractéristiques du trafic réseau")
    if exemples:
        c1, c2, _ = st.columns([1, 1, 3])
        c1.button("Charger un exemple normal", on_click=charger_exemple, args=("Normal",))
        c2.button("Charger un exemple d'attaque", on_click=charger_exemple, args=("Attaque",))

    cols = st.columns(3)
    for i, f in enumerate(FEATURES):
        cols[i % 3].number_input(f, key=f"in_{f}", help=AIDE.get(f, ""), format="%.4f")

    if st.button("Analyser le trafic", type="primary"):
        ligne = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in FEATURES}])
        proba, pred, _ = predire(ligne, seuil)
        p = float(proba[0])
        if pred[0] == 1:
            st.error(f"🚨 Attaque DDoS détectée (probabilité d'attaque : {p:.1%})")
        else:
            st.success(f"✅ Trafic normal (probabilité d'attaque : {p:.1%})")
        st.progress(min(max(p, 0.0), 1.0))

with tab2:
    st.subheader("Analyser un fichier CSV")
    st.caption("Le fichier doit contenir les colonnes : " + ", ".join(FEATURES))
    fichier = st.file_uploader("Charger un fichier CSV", type="csv")

    if fichier is not None:
        data = pd.read_csv(fichier)
        data.columns = data.columns.str.strip()
        manquantes = [f for f in FEATURES if f not in data.columns]

        if manquantes:
            st.error("Colonnes manquantes : " + ", ".join(manquantes))
        else:
            proba, pred, corriges = predire(data, seuil)
            res = data.copy()
            res["Probabilité attaque"] = proba.round(4)
            res["Prédiction"] = np.where(pred == 1, "Attaque DDoS", "Normal")

            n = len(res)
            n_att = int(pred.sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Flux analysés", f"{n:,}")
            m2.metric("Attaques DDoS", f"{n_att:,}")
            m3.metric("Part d'attaques", f"{n_att / n:.1%}")

            if corriges:
                st.warning(f"{corriges} ligne(s) contenaient des valeurs manquantes ou infinies "
                           "(remplacées par 0).")

            if n_att / n > 0.5:
                st.error("🚨 Plus de la moitié du trafic est classé comme attaque DDoS.")
            elif n_att > 0:
                st.warning("⚠️ Des flux suspects ont été détectés.")
            else:
                st.success("✅ Aucun flux suspect détecté.")

            g1, g2 = st.columns([1, 2])
            with g1:
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                ax.pie([n - n_att, n_att], labels=["Normal", "Attaque"], autopct="%1.1f%%",
                       colors=["#2e7d32", "#c62828"])
                st.pyplot(fig)
            with g2:
                st.dataframe(res.head(1000), height=330)

            st.download_button("Télécharger les résultats (CSV)",
                               res.to_csv(index=False).encode("utf-8"),
                               "resultats_ddos.csv", "text/csv")

            if "Label" in data.columns:
                y_true = (data["Label"].astype(str).str.strip().str.lower()
                          .map({"0": 0, "1": 1, "0.0": 0, "1.0": 1, "benign": 0,
                                "normal": 0, "ddos": 1, "attack": 1, "attaque": 1}))
                if y_true.notna().all():
                    y_true = y_true.astype(int)
                    st.subheader("Évaluation (le fichier contient la colonne Label)")
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Accuracy", f"{accuracy_score(y_true, pred):.4f}")
                    e2.metric("Precision", f"{precision_score(y_true, pred, zero_division=0):.4f}")
                    e3.metric("Recall", f"{recall_score(y_true, pred, zero_division=0):.4f}")
                    e4.metric("F1-score", f"{f1_score(y_true, pred, zero_division=0):.4f}")

                    cm = confusion_matrix(y_true, pred, labels=[0, 1])
                    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
                    ax2.imshow(cm, cmap="Blues")
                    ax2.set_xticks([0, 1], ["Normal", "Attaque"])
                    ax2.set_yticks([0, 1], ["Normal", "Attaque"])
                    ax2.set_xlabel("Prédit")
                    ax2.set_ylabel("Réel")
                    ax2.set_title("Matrice de confusion")
                    for i in range(2):
                        for j in range(2):
                            ax2.text(j, i, cm[i, j], ha="center", va="center",
                                     color="white" if cm[i, j] > cm.max() / 2 else "black")
                    st.pyplot(fig2)

with tab3:
    st.subheader("Comparaison des modèles (jeu de test, 44 464 flux)")
    comparaison = pd.DataFrame({
        "Modèle": ["Random Forest", "Decision Tree", "KNN (k=5)",
                   "Logistic Regression", "Naive Bayes"],
        "Accuracy": [0.9993, 0.9991, 0.9980, 0.9865, 0.8194],
        "Precision": [0.9994, 0.9990, 0.9977, 0.9848, 0.7388],
        "Recall": [0.9992, 0.9991, 0.9982, 0.9868, 0.9582],
        "F1-score": [0.9993, 0.9991, 0.9979, 0.9858, 0.8343],
    })
    st.dataframe(comparaison, hide_index=True)
    st.bar_chart(comparaison.set_index("Modèle")[["Accuracy", "Precision", "Recall", "F1-score"]])

    st.subheader("Modèle déployé (Random Forest, 15 caractéristiques)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Accuracy", "0.9995")
    p2.metric("Precision", "0.9997")
    p3.metric("Recall", "0.9993")
    p4.metric("F1-score", "0.9995")
    st.write("Matrice de confusion : VN = 23357, FP = 7, FN = 14, VP = 21086.")

    st.info("Limite : le dataset fusionne plusieurs sources de trafic. Le modèle peut donc "
            "apprendre en partie la signature de la source en plus du comportement de "
            "l'attaque ; les performances sur un autre réseau pourraient être plus faibles.")
