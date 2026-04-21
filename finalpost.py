# =============================
# SMIS-R FINAL ULTIMATE SYSTEM
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
# 📂 CSV UPLOAD
# -----------------------------
st.sidebar.subheader("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Default dataset
materials_data = [
    ["Aluminum Alloy",2700,660,0.7,30,120,40,90],
    ["Titanium Alloy",4500,1660,0.8,90,150,60,70],
    ["Nickel Superalloy",8400,1350,0.9,85,140,70,50],
    ["Stainless Steel",8000,1450,0.85,70,130,60,40],
    ["CFRP",1600,300,0.6,50,100,90,95],
    ["Ceramic Matrix Composite",3000,1800,0.95,60,200,75,60],
    ["Silicon Carbide",3200,2700,0.98,65,220,80,55],
    ["Carbon-Carbon Composite",1800,3500,0.97,55,250,70,65],
    ["Kapton Polymer",1400,400,0.5,45,110,95,98],
    ["PTFE Polymer",2200,327,0.4,35,100,90,92]
]

columns = [
    "Material","Density","Melting_Point",
    "Radiation_Resistance","Radiation",
    "Temp_Resistance","Oxygen_Resistance",
    "Weight_Score"
]

if uploaded_file is not None:
    df_materials = pd.read_csv(uploaded_file)
    dataset_name = "Custom Dataset"
else:
    df_materials = pd.DataFrame(materials_data, columns=columns)
    dataset_name = "Default Dataset"

st.info(f"📂 Using: {dataset_name}")

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.subheader("📂 Active Dataset")
st.write(f"Total Materials: {len(df_materials)}")
st.dataframe(df_materials)

materials = df_materials["Material"].tolist()

# -----------------------------
# INPUTS
# -----------------------------
st.sidebar.header("⚙️ Space Conditions")

temp = st.sidebar.slider("Temperature (°C)", -270, 200, 20)   # space range
rad = st.sidebar.slider("Radiation (Gy)", 0, 1000, 100)       # real scale
oxy = st.sidebar.number_input("Atomic Oxygen (atoms/cm³)", value=1e9, format="%.2e")
dur = st.sidebar.slider("Mission Duration (years)", 1, 30, 5)

st.sidebar.header("🧪 Advanced Factors")

thermal_cycle = st.sidebar.slider("Thermal Cycling (cycles)", 1, 10000, 500)
uv = st.sidebar.slider("UV Exposure (W/m²)", 0, 2000, 500)
outgas = st.sidebar.slider("Outgassing Rate", 0.0, 1.0, 0.1)

strength_req = st.sidebar.slider("Strength Required (MPa)", 50, 2000, 500)
fatigue = st.sidebar.slider("Fatigue Life (cycles)", 1e3, 1e9, 1e6, format="%.0f")

weight_priority = st.sidebar.slider("Weight Importance (%)", 0, 100, 70)
cost = st.sidebar.slider("Cost (kg)", 1, 1000, 100)
# -----------------------------
# GENERATE TRAINING DATA
# -----------------------------
def generate_data(n=300):
    rows = []

    for _ in range(n):
        t = np.random.uniform(-180, 150)
        r = np.random.uniform(1, 100)
        o = np.random.uniform(1e8, 1e10)
        d = np.random.uniform(1, 10)

        thermal = np.random.uniform(1, 100)
        uv_exp = np.random.uniform(1, 100)
        outgas_exp = np.random.uniform(1, 100)
        strength_req = np.random.uniform(1, 100)
        fatigue_req = np.random.uniform(1, 100)
        weight_imp = np.random.uniform(1, 100)
        cost_imp = np.random.uniform(1, 100)

        for _, row in df_materials.iterrows():

            score = (
                
    (row["Radiation_Resistance"] * (1 - r/1000)) +
    (row["Temp_Resistance"] * (1 - abs(t)/300)) +
    (1 / (row["Density"] + 1)) * weight_imp +
    (row["Oxygen_Resistance"] * (1 - min(o/1e10, 1))) -
    d * 0.2 -
    cost_imp * 0.3
           )

            rows.append([
                t, r, o, d,
                thermal, uv_exp, outgas_exp,
                strength_req, fatigue_req,
                weight_imp, cost_imp,
                row["Material"],
                score
            ])

    return pd.DataFrame(rows, columns=[
        "temp","rad","oxy","dur",
        "thermal","uv","outgas",
        "strength","fatigue",
        "weight_imp","cost_imp",
        "material","score"
    ])

df = generate_data()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numeric_cols = [
    "temp","rad","oxy","dur",
    "thermal","uv","outgas",
    "strength","fatigue",
    "weight_imp","cost_imp"
]

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# ML MODEL
# -----------------------------
le = LabelEncoder()
df["material_enc"] = le.fit_transform(df["material"])

X = df[[
    "temp","rad","oxy","dur",
    "thermal","uv","outgas",
    "strength","fatigue",
    "weight_imp","cost_imp",
    "material_enc"
]]

y = df["score"]

model = RandomForestRegressor(n_estimators=120)
model.fit(X, y)

# -----------------------------
# PREDICTION
# -----------------------------
def predict_scores():
    results = []

    for mat in materials:
        enc = le.transform([mat])[0]

        pred = model.predict([[
            temp, rad, oxy, dur,
            thermal_cycle, uv, outgas,
            strength_req, fatigue,
            weight_priority, cost,
            enc
        ]])[0]

        results.append((mat, pred))

    return sorted(results, key=lambda x: x[1], reverse=True)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def feature_importance():
    features = [
        "Temperature","Radiation","Atomic Oxygen","Duration",
        "Thermal","UV","Outgassing",
        "Strength","Fatigue",
        "Weight","Cost","Material"
    ]

    return pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

# -----------------------------
# CLUSTERING (FIXED)
# -----------------------------
def clustering():
    features = df_materials[[
        "Density","Melting_Point",
        "Radiation_Resistance",
        "Temp_Resistance",
        "Oxygen_Resistance"
    ]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

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
    features = df_materials[[
        "Density","Melting_Point",
        "Radiation_Resistance",
        "Temp_Resistance",
        "Oxygen_Resistance"
    ]]

    dist = euclidean_distances(
        df_materials[df_materials["Material"] == best][features.columns],
        features
    )

    df_materials["Sim"] = dist[0]

    return df_materials.sort_values("Sim")[1:4]["Material"]

# -----------------------------
# PARETO FRONT
# -----------------------------
def pareto(df_res):
    df_res["Weight"] = df_res["Material"].map(
        df_materials.set_index("Material")["Density"]
    )

    pareto_list = []

    for i, row in df_res.iterrows():
        dominated = False
        for j, other in df_res.iterrows():
            if j != i:
                if (other["Score"] >= row["Score"] and
                    other["Weight"] <= row["Weight"] and
                    (other["Score"] > row["Score"] or other["Weight"] < row["Weight"])):
                    dominated = True
                    break
        if not dominated:
            pareto_list.append(row)

    return pd.DataFrame(pareto_list)

# -----------------------------
# RUN ANALYSIS
# -----------------------------
if st.sidebar.button("🚀 Analyze"):

    results = predict_scores()

    df_res = pd.DataFrame(results, columns=["Material","Score"])
    df_res["Confidence"] = (df_res["Score"]/df_res["Score"].sum())*100

    best = df_res.iloc[0]["Material"]

    st.subheader("🏆 Best Material")
    st.success(best)

    # Graphs
    st.plotly_chart(px.bar(df_res, x="Score", y="Material", orientation="h"), use_container_width=True)
    st.plotly_chart(px.pie(df_res, names="Material", values="Confidence"), use_container_width=True)

    st.dataframe(df_res)

    # Feature Importance
    st.subheader("🔬 Feature Importance")
    st.plotly_chart(px.bar(feature_importance(), x="Importance", y="Feature", orientation="h"), use_container_width=True)

    # Clustering
    st.subheader("🧬 Clustering")
    cdf = clustering()
    st.plotly_chart(px.scatter(cdf, x="PC1", y="PC2", color="Cluster", text="Material"), use_container_width=True)

    # Similar
    st.subheader("🔁 Alternatives")
    for m in similar(best):
        st.write("•", m)

    # Pareto
    st.subheader("⚖️ Pareto Optimization")
    p_df = pareto(df_res)

    st.dataframe(p_df[["Material","Score","Weight"]])

    fig = px.scatter(df_res, x="Weight", y="Score", text="Material")
    fig.add_scatter(x=p_df["Weight"], y=p_df["Score"], mode="markers", name="Pareto")
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    st.subheader("🧠 Explanation")
    for _, row in feature_importance().iterrows():
        st.write(f"{row['Feature']} → {round(row['Importance']*100,2)}%")