from pathlib import Path
import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction", layout="centered")

@st.cache_resource
def load():
    art = joblib.load(Path(__file__).parent / "model.joblib")
    return art["model"], art["columns"]

model, columns = load()

st.title("Prédiction de churn : Telco")
st.write("Renseigne le profil d'un client pour estimer son risque de départ.")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Ancienneté (mois)", 1, 72, 6)
    monthly = st.slider("Charges mensuelles", 18, 120, 70)
    senior = st.checkbox("Senior")
with col2:
    contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Paiement", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

row = pd.DataFrame(0.0, index=[0], columns=columns)
row["tenure"] = tenure
row["MonthlyCharges"] = monthly
row["TotalCharges"] = tenure * monthly
row["SeniorCitizen"] = int(senior)
for c in [f"Contract_{contract}", f"InternetService_{internet}", f"PaymentMethod_{payment}"]:
    if c in columns:
        row[c] = 1.0

proba = float(model.predict_proba(row)[0, 1])

st.subheader(f"Probabilité de churn : {proba:.0%}")
if proba >= 0.6:
    st.error("Risque élevé : action de rétention recommandée.")
elif proba >= 0.3:
    st.warning("Risque moyen : à surveiller.")
else:
    st.success("Risque faible.")

st.subheader("Pourquoi ?")
explainer = shap.TreeExplainer(model)
sv = explainer(row)
shap.plots.waterfall(sv[0], max_display=8, show=False)
st.pyplot(plt.gcf())
plt.close()

st.caption("Les autres variables (services, etc.) sont fixées à leur valeur de référence.")
