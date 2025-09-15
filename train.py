# train.py
import os
import argparse
import pandas as pd
from joblib import dump
from pipeline import make_pipeline  # assumes your pipeline.py defines make_pipeline

def load_any(path: str) -> pd.DataFrame:
    return pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/marketing_campaign.xlsx", help="Path to raw CSV/XLSX")
    ap.add_argument("--clusters", type=int, default=4, help="K for KMeans (default 4)")
    args = ap.parse_args()

    df_raw = load_any(args.data)
    print(f"Loaded raw data: {df_raw.shape}")

    pipe = make_pipeline(n_clusters=args.clusters, random_state=42)
    pipe.fit(df_raw)

    os.makedirs("models", exist_ok=True)
    dump(pipe, "models/cluster_pipeline.pkl", compress=3)
    print(f"Saved models/cluster_pipeline.pkl (k={args.clusters})")

if __name__ == "__main__":
    main()
