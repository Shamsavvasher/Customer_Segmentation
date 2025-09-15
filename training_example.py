# training_example.py
import os
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from pipeline import FeatureBuilder
from auto_select import UnsupervisedFeatureSelector, CorrFilterConfig

RAW_PATH = "data/marketing_campaign.xlsx"
df_raw = pd.read_excel(RAW_PATH)

pipe = Pipeline([
    ("features", FeatureBuilder()),
    ("selector", UnsupervisedFeatureSelector(
        k=4, max_features=12, min_gain=0.003, seeds=(0,1,2),
        corr_filter=CorrFilterConfig(corr_thresh=0.92, prefer=["Total_Spending","Total_Purchases","Recency"])
    )),
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=10)),
])

pipe.fit(df_raw)
sel = pipe.named_steps["selector"]
print("Selected features:", sel.selected_)
print("Search history:", sel.history_)

os.makedirs("models", exist_ok=True)
dump(pipe, "models/cluster_pipeline.pkl", compress=3)
print("Saved models/cluster_pipeline.pkl")
