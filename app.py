# app.py
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.decomposition import PCA

# IMPORTANT: lets joblib resolve the custom transformer inside the PKL
from pipeline import FeatureBuilder  # noqa: F401

st.set_page_config(page_title="Customer Segmentation", page_icon="🧩", layout="wide")
MODEL_PATH = "models/cluster_pipeline.pkl"
SAMPLE_PATH = "data/marketing_campaign.xlsx"  # optional demo file

st.title("🧩 Customer Segmentation (KMeans)")
st.caption("Loads a pre-trained pipeline (FeatureBuilder → StandardScaler → KMeans). Upload a raw CSV/XLSX or try the what-if sliders.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Train/save it first.")
        st.stop()
    try:
        return load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {type(e).__name__}: {e}")
        st.stop()

def read_df(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    return pd.read_csv(uploaded) if name.endswith(".csv") else pd.read_excel(uploaded)

def pca_scatter(X_scaled, labels, title):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=labels, palette="tab10", ax=ax, s=20, legend="full")
    ax.set_title(title)
    st.pyplot(fig)

model = load_model()

# show model k
k = None
if hasattr(model, "named_steps") and "kmeans" in model.named_steps:
    try:
        k = model.named_steps["kmeans"].n_clusters
    except Exception:
        pass
if k is not None:
    st.write(f"**Model k:** {k}")

tab_upload, tab_whatif = st.tabs(["📤 Upload & Predict", "🎛️ What-if (feature sliders)"])

# --------------------- Upload & Predict ---------------------
with tab_upload:
    with st.sidebar:
        st.header("Data Source")
        uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx","xls"], key="uploader_upload")
        use_sample = st.checkbox("Use sample at data/marketing_campaign.xlsx", value=os.path.exists(SAMPLE_PATH), key="use_sample_upload")

    if uploaded is not None:
        df = read_df(uploaded)
    elif use_sample and os.path.exists(SAMPLE_PATH):
        df = pd.read_excel(SAMPLE_PATH)
    else:
        df = None

    if df is None:
        st.info("Upload a raw file or tick the sample checkbox.")
    else:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        if st.button("Predict segments", type="primary", key="predict_btn"):
            labels = model.predict(df)          # raw → engineered inside pipeline
            out = df.copy()
            out["cluster"] = labels
            st.success("Predictions ready")

            # profiling & PCA with pipeline internals
            fb = model.named_steps["features"]
            sc = model.named_steps["scaler"]
            Xf = fb.transform(df)      # engineered features (DataFrame)
            Xsc = sc.transform(Xf)     # scaled (NumPy)

            st.markdown("#### Segment counts")
            st.write(out["cluster"].value_counts().rename_axis("cluster").to_frame("count"))

            st.markdown("#### Cluster numeric means (this dataset)")
            feat_cols = getattr(fb, "feature_names_", None) or Xf.columns
            prof_df = pd.DataFrame(Xf, columns=feat_cols, index=df.index)
            prof_df["cluster"] = labels
            st.dataframe(
                prof_df.groupby("cluster")[feat_cols].mean().round(2),
                use_container_width=True
            )

            pca_scatter(Xsc, labels, "KMeans clusters (PCA 2D)")

            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False),
                file_name="segmented_customers.csv"
            )

# --------------------- What-if sliders ---------------------
with tab_whatif:
    st.write("Adjust engineered features and see the predicted cluster instantly.")
    scaler = model.named_steps.get("scaler")
    expected = list(getattr(scaler, "feature_names_in_", []))
    if not expected:
        st.error("Scaler is missing feature names. Retrain with this pipeline setup.")
        st.stop()

    cols = st.columns(2)
    values = {}
    for i, feat in enumerate(expected):
        with cols[i % 2]:
            if feat == "Age":
                values[feat] = st.slider("Age", 18, 100, 55)
            elif feat == "Customer_Tenure":
                values[feat] = st.slider("Customer Tenure (days)", 0, 8000, 4400)
            elif feat == "Income":
                values[feat] = st.slider("Income", 0, 200_000, 75_000, step=1_000)
            elif feat == "Total_Children":
                values[feat] = st.slider("Total Children", 0, 5, 1)
            elif feat == "Recency":
                values[feat] = st.slider("Recency (days since last purchase)", 0, 100, 48)
            elif feat == "Total_Purchases":
                values[feat] = st.slider("Total Purchases", 0, 100, 40)
            elif feat == "Total_Spending":
                values[feat] = st.slider("Total Spending", 0, 10000, 1200, step=50)
            elif feat == "Total_Campaigns_Accepted":
                values[feat] = st.slider("Total Campaigns Accepted", 0, 10, 1)
            else:
                values[feat] = st.number_input(feat, value=0.0)

    X_one = pd.DataFrame([values], columns=expected)
    Xsc_one = scaler.transform(X_one)
    label_one = model.named_steps["kmeans"].predict(Xsc_one)[0]
    st.success(f"Predicted cluster: **{label_one}**")
