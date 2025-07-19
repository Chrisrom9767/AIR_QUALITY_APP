import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go

# --- Charger le modèle ---
model = joblib.load('random_forest_model.pkl')

# --- Fonctions utiles ---
def categoriser_aqi(aqi):
    if aqi <= 50:
        return 'Bonne', '#009966'
    elif aqi <= 100:
        return 'Moyenne', '#FFDE33'
    elif aqi <= 150:
        return 'Mauvaise pour les sensibles', '#FF9933'
    elif aqi <= 200:
        return 'Mauvaise', '#CC0033'
    elif aqi <= 300:
        return 'Très mauvaise', '#660099'
    else:
        return 'Dangereuse', '#7E0023'

def afficher_gauge(aqi, categorie, couleur):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"AQI - {categorie}", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 500]},
            'bar': {'color': couleur},
            'steps': [
                {'range': [0, 50], 'color': '#009966'},
                {'range': [51, 100], 'color': '#FFDE33'},
                {'range': [101, 150], 'color': '#FF9933'},
                {'range': [151, 200], 'color': '#CC0033'},
                {'range': [201, 300], 'color': '#660099'},
                {'range': [301, 500], 'color': '#7E0023'},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# --- Configuration de la page ---
st.set_page_config(page_title="AQI Prédiction", layout="wide")

st.markdown("<h1 style='text-align: center; color:#4CAF50;'>🌍 AQI - Prédiction de la Qualité de l'Air</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Modèle de prédiction de l’indice AQI global basé sur les paramètres environnementaux</p>", unsafe_allow_html=True)

# --- Entrée utilisateur ---
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 25.0)
        no2 = st.slider("NO2 (ppb)", 0.0, 1000.0, 30.0)
        temp = st.slider("Température (°C)", -20.0, 50.0, 24.0)
        rain = st.slider("Pluie (mm)", 0.0, 100.0, 0.0)

    with col2:
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 35.0)
        co = st.slider("CO (ppm)", 0.0, 50.0, 1.0)
        rh = st.slider("Humidité relative (%)", 0.0, 100.0, 50.0)
        date = st.date_input("Date", value=datetime.date.today())

    with col3:
        so2 = st.slider("SO2 (ppb)", 0.0, 1000.0, 20.0)
        o3 = st.slider("O3 (ppb)", 0.0, 300.0, 30.0)
        wind = st.slider("Vent (m/s)", 0.0, 20.0, 3.0)
        hour = st.slider("Heure", 0, 23, 12)

    submitted = st.form_submit_button("🎯 Prédire")

if submitted:
    # Extraire les features temporelles
    year = date.year
    month = date.month
    day = date.day
    dayofweek = date.weekday()
    dayofyear = date.timetuple().tm_yday
    is_weekend = 1 if dayofweek >= 5 else 0

    # Créer le vecteur d'entrée
    features = np.array([[pm25, pm10, no2, co, so2, o3, temp, rh, wind, rain,
                          year, month, day, hour, dayofweek, dayofyear, is_weekend]])

    # Prédiction
    prediction = model.predict(features)[0]
    categorie, couleur = categoriser_aqi(prediction)

    # Affichage résultat
    st.success(f"✅ AQI global prédit : {round(prediction, 1)}")
    st.markdown(f"### 🟢 Qualité de l'air : `{categorie}`")
    afficher_gauge(prediction, categorie, couleur)

    # Sauvegarde locale dans session_state
    if "historique" not in st.session_state:
        st.session_state.historique = []

    st.session_state.historique.append({
        "Date": date,
        "Heure": hour,
        "AQI": round(prediction, 1),
        "Qualité": categorie
    })

# --- Affichage historique ---
if "historique" in st.session_state:
    st.markdown("### 📊 Historique des prédictions")
    df_hist = pd.DataFrame(st.session_state.historique)
    st.dataframe(df_hist)

    csv = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger l'historique", data=csv, file_name="historique_aqi.csv", mime="text/csv")
