# =============================
# Space Material Intelligence System (SMIS) - PREMIUM UI
# =============================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PAGE CONFIG + THEME
# -----------------------------
st.set_page_config(page_title="SMIS Premium", layout="wide")

st.markdown("""
<style>
    .main {background-color: #0e1117; color: #ffffff;}
    .stMetric {background-color: #1c1f26; padding: 15px; border-radius: 12px;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Space Material Intelligence System")
st.caption("AI-driven material selection, degradation prediction & intelligent substitution")

# -----------------------------
# DATA GENERATION
# -----------------------------
def generate_data(n=300):
    data = []
    materials = ["Aluminum", "Titanium", "CFRP", "Ceramic"]

    for _ in range(n):
        temp = np.random.uniform(-180, 150)
        rad = np.random.uniform(1, 100)
        pressure = np.random.uniform(1e-10, 1e-5)
        oxy = np.random.uniform(1e8, 1e10)
        duration = np.random.uniform(1, 10)

        if rad > 70:
            material = "Titanium"
        elif temp > 100:
            material = "Ceramic"
        elif oxy > 5e9:
            material = "CFRP"
        else:
            material = "Aluminum"

        strength_loss = min(100, (rad * 0.3 + abs(temp) * 0.1 + duration * 5) / 2)
        failure_risk = min(100, strength_loss * 0.8 + rad * 0.2)

        data.append([temp, rad, pressure, oxy, duration, material, strength_loss, failure_risk])

    return pd.DataFrame(data, columns=[
        "temperature", "radiation", "pressure", "atomic_oxygen", "duration",
        "material", "strength_loss", "failure_risk"
    ])

# -----------------------------
# MODEL TRAINING
# -----------------------------
df = generate_data()
le = LabelEncoder()
df['material_enc'] = le.fit_transform(df['material'])

X = df[["temperature", "radiation", "pressure", "atomic_oxygen", "duration"]]
y_mat = df['material_enc']
y_strength = df['strength_loss']
y_risk = df['failure_risk']

model_mat = RandomForestClassifier()
model_strength = RandomForestRegressor()
model_risk = RandomForestRegressor()

model_mat.fit(X, y_mat)
model_strength.fit(X, y_strength)
model_risk.fit(X, y_risk)

# -----------------------------
# FUNCTIONS
# -----------------------------
def predict_all(temp, rad, press, oxy, dur):
    inp = np.array([[temp, rad, press, oxy, dur]])
    mat = le.inverse_transform(model_mat.predict(inp))[0]
    strength = model_strength.predict(inp)[0]
    risk = model_risk.predict(inp)[0]
    return mat, strength, risk


def rank_materials(temp, rad, press, oxy, dur):
    materials = ["Aluminum", "Titanium", "CFRP", "Ceramic"]
    scores = {}

    for m in materials:
        score = 100
        if m == "Titanium": score += rad * 0.5
        if m == "CFRP": score += (100 - rad) * 0.3
        if m == "Ceramic": score += abs(temp) * 0.4
        if m == "Aluminum": score += 50 - rad * 0.2
        scores[m] = score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("⚙️ Mission Parameters")

temp = st.sidebar.slider("Temperature (°C)", -180, 150, 20)
rad = st.sidebar.slider("Radiation (krad)", 1, 100, 30)
press = st.sidebar.number_input("Pressure (Pa)", value=1e-7, format="%e")
oxy = st.sidebar.number_input("Atomic Oxygen", value=1e9, format="%e")
dur = st.sidebar.slider("Mission Duration (years)", 1, 10, 3)

analyze = st.sidebar.button("🚀 Analyze")

# -----------------------------
# MAIN OUTPUT
# -----------------------------
if analyze:
    mat, strength, risk = predict_all(temp, rad, press, oxy, dur)

    col1, col2, col3 = st.columns(3)
    col1.metric("Material", mat)
    col2.metric("Strength Loss (%)", round(strength, 2))
    col3.metric("Failure Risk (%)", round(risk, 2))

    st.markdown("---")

    # ALTERNATIVES
    st.subheader("🔁 Alternative Materials (Trade-off Analysis)")
    ranked = rank_materials(temp, rad, press, oxy, dur)
    rank_df = pd.DataFrame(ranked, columns=["Material", "Score"])
    st.dataframe(rank_df, use_container_width=True)

    # DEGRADATION GRAPH
    st.subheader("📉 Degradation Over Time")
    years = np.arange(1, int(dur)+1)
    degradation = [min(100, strength * (y / dur)) for y in years]

    fig, ax = plt.subplots()
    ax.plot(years, degradation, linewidth=3)
    ax.set_xlabel("Years")
    ax.set_ylabel("Strength Loss (%)")
    ax.set_title("Material Degradation Curve")
    st.pyplot(fig)

    # RISK BAR CHART
    st.subheader("📊 Risk Distribution")
    materials = [m[0] for m in ranked]
    scores = [m[1] for m in ranked]

    fig2, ax2 = plt.subplots()
    ax2.bar(materials, scores)
    ax2.set_title("Material Suitability Score")
    st.pyplot(fig2)

    # EXPLANATION
    st.subheader("🧠 Decision Explanation")
    if mat == "Titanium":
        st.success("Selected for high radiation resistance and durability.")
    elif mat == "CFRP":
        st.success("Selected for lightweight structure with good strength.")
    elif mat == "Ceramic":
        st.success("Selected due to excellent thermal resistance.")
    else:
        st.success("Selected as cost-effective solution under moderate conditions.")

st.markdown("---")
st.caption("🚀 Premium Space Material Intelligence System")
