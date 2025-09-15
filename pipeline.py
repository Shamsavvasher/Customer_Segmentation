import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Build numeric features from the raw marketing file."""
    def __init__(self, income_cap=200_000, age_cap=100):
        self.income_cap = income_cap
        self.age_cap = age_cap
        self.feature_names_ = [
            "Age",
            "Customer_Tenure",
            "Income",
            "Total_Children",
            "Recency",
            "Total_Purchases",
            "Total_Spending",
            "Total_Campaigns_Accepted",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        out = pd.DataFrame(index=df.index)

        # Age
        if "Year_Birth" in df.columns:
            year_now = datetime.now().year
            out["Age"] = (year_now - pd.to_numeric(df["Year_Birth"], errors="coerce")).clip(0, self.age_cap)
        else:
            out["Age"] = 0

        # Customer Tenure (days)
        dt = pd.to_datetime(df.get("Dt_Customer"), errors="coerce")
        out["Customer_Tenure"] = (pd.Timestamp.today().normalize() - dt).dt.days

        # Income (cap)
        out["Income"] = pd.to_numeric(df.get("Income", 0), errors="coerce").clip(0, self.income_cap)

        # Total Children
        kid  = pd.to_numeric(df.get("Kidhome", 0), errors="coerce").fillna(0)
        teen = pd.to_numeric(df.get("Teenhome", 0), errors="coerce").fillna(0)
        out["Total_Children"] = kid + teen

        # Recency
        out["Recency"] = pd.to_numeric(df.get("Recency", 0), errors="coerce")

        # Total Purchases
        pcols = [c for c in df.columns if "Purchases" in c]
        out["Total_Purchases"] = df[pcols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if pcols else 0

        # Total Spending
        scols = [c for c in df.columns if c.startswith("Mnt")]
        out["Total_Spending"] = df[scols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if scols else 0

        # Total Campaigns Accepted
        acols = [c for c in df.columns if "AcceptedCmp" in c]
        out["Total_Campaigns_Accepted"] = df[acols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if acols else 0

        # Clean NaNs / negatives with per-column medians
        for c in self.feature_names_:
            col = pd.to_numeric(out[c], errors="coerce")
            med = np.nanmedian(col) if np.isfinite(np.nanmedian(col)) else 0
            out[c] = col.fillna(med).mask(col < 0, med)

        # Return a DataFrame so downstream steps keep column names
        return out[self.feature_names_]

def make_pipeline(n_clusters=4, random_state=42):
    """FeatureBuilder → StandardScaler → KMeans"""
    return Pipeline(steps=[
        ("features", FeatureBuilder()),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
    ])
