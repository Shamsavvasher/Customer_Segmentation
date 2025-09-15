# auto_select.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def _z(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(df.values)

def _avg_silhouette(X: np.ndarray, k: int, seeds: Iterable[int]) -> float:
    scores = []
    for s in seeds:
        km = KMeans(n_clusters=k, random_state=s, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            scores.append(silhouette_score(X, labels))
    return float(np.mean(scores)) if scores else -1.0

@dataclass
class CorrFilterConfig:
    var_thresh: float = 1e-8
    corr_thresh: float = 0.92
    prefer: Optional[List[str]] = None

class UnsupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Greedy forward selection for clustering:
    - drop near-constant & highly correlated features
    - add features that improve average Silhouette across seeds
    """
    def __init__(
        self,
        k: int = 4,
        max_features: int = 12,
        min_gain: float = 0.005,
        seeds: Iterable[int] = (0, 1, 2),
        corr_filter: CorrFilterConfig = CorrFilterConfig(),
    ):
        self.k = k
        self.max_features = max_features
        self.min_gain = min_gain
        self.seeds = tuple(seeds)
        self.corr_filter = corr_filter
        self.selected_: List[str] = []
        self.history_: List[dict] = []
        self.candidates_: List[str] = []

    def _corr_filter(self, X: pd.DataFrame) -> List[str]:
        variances = X.var(numeric_only=True)
        keep = variances[variances > self.corr_filter.var_thresh].index.tolist()
        Xk = X[keep]
        corr = Xk.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = set()
        prefer = set(self.corr_filter.prefer or [])
        for col in upper.columns:
            highs = [row for row, v in upper[col].dropna().items() if v >= self.corr_filter.corr_thresh]
            for row in highs:
                if (col in prefer) and (row not in prefer):
                    drop.add(row)
                elif (row in prefer) and (col not in prefer):
                    drop.add(col)
                else:
                    if variances[col] >= variances[row]:
                        drop.add(row)
                    else:
                        drop.add(col)
        return [c for c in Xk.columns if c not in drop]

    def fit(self, X: pd.DataFrame, y=None):
        X = X.select_dtypes(include=[np.number]).copy()
        cand = self._corr_filter(X)
        self.candidates_ = cand

        selected: List[str] = []
        best_score = -1.0
        while len(selected) < self.max_features:
            best_feat = None
            best_try_score = best_score
            for f in cand:
                if f in selected:
                    continue
                cols = selected + [f]
                score = _avg_silhouette(_z(X[cols]), self.k, self.seeds)
                if score > best_try_score + 1e-12:
                    best_try_score = score
                    best_feat = f
            gain = best_try_score - best_score
            self.history_.append({"step": len(selected)+1, "score": best_try_score, "gain": gain, "added": best_feat})
            if (best_feat is None) or (gain < self.min_gain):
                break
            selected.append(best_feat)
            best_score = best_try_score

        self.selected_ = selected
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_:
            raise RuntimeError("Selector not fitted or no features selected.")
        return X[self.selected_].copy()
