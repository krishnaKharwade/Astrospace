# =============================
# ASTROSPACE FINAL STABLE VERSION
# =============================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(page_title="ASTROSPACE", layout="wide")

st.title("🚀 ASTROSPACE: Space Material Intelligence System")
st.markdown("### AI-Based Material Selection for Space Applications")

# -----------------------------
# 📂 CSV UPLOAD (PRESERVED)
# -----------------------------
st.sidebar.subheader("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

required_cols = [
    "Material","Density","Melting_Point",
    "Radiation_Resistance","Radiation",
    "Temp_Resistance","Oxygen_Resistance"
]

default_data = [
    ["Aluminum Alloy",2700,660,0.7,30,120,40],
    ["Titanium Alloy",4500,1660,0.8,90,150,60],
    ["Nickel Superalloy",8400,1350,0.9,85,140,70],
    ["Stainless Steel",8000,1450,0.85,70,130,60],
    ["CFRP",1600,300,0.6,50,100,90],
    ["Ceramic Matrix Composite",3000,1800,0.95,60,200,75],
    ["Silicon Carbide",3200,2700,0.98,65,220,80],
    ["Carbon-Carbon Composite",1800,3500,0.97,55,250,70]
]

columns = required_cols

if uploaded_file:
    df_materials = pd.read_csv(uploaded_file)

    if not all(col in df_materials.columns for col in required_cols):
        st.error("❌ Invalid CSV format")
        st.stop()

    dataset_name = "Custom Dataset"
else:
    df_materials = pd.DataFrame(default_data, columns=columns)
    dataset_name = "Default Dataset"

st.info(f"📂 Using: {dataset_name}")

# -----------------------------
# DATASET
# -----------------------------
st.subheader("📊 Dataset")
st.dataframe(df_materials)

materials = df_materials["Material"].tolist()

# -----------------------------
# 🌍 REAL INPUTS
# -----------------------------
st.sidebar.header("🌍 Space Conditions")

temp = st.sidebar.number_input("Temperature (°C)", -180, 150, 25)
rad = st.sidebar.number_input("Radiation (krad/year)", 0.0, 500.0, 50.0)
oxy = st.sidebar.number_input("Atomic Oxygen (atoms/cm²/s)", 1e8, 1e10, 1e9)
dur = st.sidebar.number_input("Mission Duration (years)", 1.0, 20.0, 5.0)

st.sidebar.header("⚙️ Constraints")

strength = st.sidebar.number_input("Strength Requirement (MPa)", 50, 3000, 500)
fatigue = st.sidebar.number_input("Fatigue Cycles", 1000, 10000000, 100000)
mass_limit = st.sidebar.number_input("Mass Limit (kg)", 1, 1000, 100)
cost = st.sidebar.number_input("Cost ($/kg)", 1, 2000, 100)

# -----------------------------
# TRAINING DATA
# -----------------------------
def generate_data(n=300):
    rows = []

    for _ in range(n):
        t = np.random.uniform(-180, 150)
        r = np.random.uniform(0, 500)
        o = np.random.uniform(1e8, 1e10)
        d = np.random.uniform(1, 20)

        strength = np.random.uniform(50, 3000)
        fatigue = np.random.uniform(1e3, 1e7)
        weight = np.random.uniform(10, 1000)
        cost = np.random.uniform(5, 2000)

        for _, row in df_materials.iterrows():

            score = (
                100 - abs(row["Radiation_Resistance"] - r/5) +
                100 - abs(row["Temp_Resistance"] - abs(t)) +
                100 - abs(row["Oxygen_Resistance"] - min(100, o/1e8)) +
                (1000 / row["Density"]) * (mass_limit/100) -
                cost * 0.05
            )

            rows.append([
                t,r,o,d,
                strength,fatigue,weight,cost,
                row["Material"],score
            ])

    return pd.DataFrame(rows, columns=[
        "temp","rad","oxy","dur",
        "strength","fatigue","weight","cost",
        "material","score"
    ])

df = generate_data()

# -----------------------------
# ML MODEL
# -----------------------------
le = LabelEncoder()
df["material_enc"] = le.fit_transform(df["material"])

X = df.drop(["material","score"], axis=1)
X["material"] = df["material_enc"]

y = df["score"]

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X, y)

# -----------------------------
# INPUT NORMALIZATION
# -----------------------------
def normalize_inputs():
    return [
        temp,
        rad,
        oxy,
        dur,
        strength,
        fatigue,
        mass_limit,
        cost
    ]

# -----------------------------
# PREDICTION (FIXED)
# -----------------------------
def predict():
    results = []

    inputs = normalize_inputs()

    for mat in materials:
        enc = le.transform([mat])[0]

        x_input = np.array(inputs + [enc]).reshape(1, -1)

        pred = model.predict(x_input)[0]

        results.append((mat, pred))

    return sorted(results, key=lambda x: x[1], reverse=True)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def feature_importance():
    features = [
        "Temperature","Radiation","Oxygen","Duration",
        "Strength","Fatigue","Weight","Cost","Material"
    ]

    return pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

# -----------------------------
# CLUSTERING
# -----------------------------
def clustering():
    features = df_materials.drop("Material", axis=1)

    scaled = StandardScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_materials["Cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)

    df_materials["PC1"] = reduced[:,0]
    df_materials["PC2"] = reduced[:,1]

    return df_materials

# -----------------------------
# SIMILAR MATERIALS
# -----------------------------
def similar(best):
    features = df_materials.drop("Material", axis=1)

    dist = euclidean_distances(
        features[df_materials["Material"] == best],
        features
    )

    df_materials["Sim"] = dist[0]

    return df_materials.sort_values("Sim")[1:4]["Material"]

# -----------------------------
# RUN
# -----------------------------
if st.sidebar.button("🚀 Analyze"):

    results = predict()
    df_res = pd.DataFrame(results, columns=["Material","Score"])

    best = df_res.iloc[0]["Material"]

    st.subheader("🏆 Best Material")
    st.success(best)

    st.plotly_chart(px.bar(df_res, x="Score", y="Material", orientation="h"), use_container_width=True)

    st.subheader("📊 Feature Importance")
    st.plotly_chart(px.bar(feature_importance(), x="Importance", y="Feature", orientation="h"), use_container_width=True)

    st.subheader("🧬 Clustering")
    cdf = clustering()
    st.plotly_chart(px.scatter(cdf, x="PC1", y="PC2", color="Cluster", text="Material"), use_container_width=True)

    st.subheader("🔁 Similar Materials")
    for m in similar(best):
        st.write("•", m)

    st.subheader("📋 Results")
    st.dataframe(df_res)