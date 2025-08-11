# research/modeling/calibration.py
import numpy as np
from typing import Tuple, Dict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

def fit_with_best_calibration(base_pipeline, X, y, cv) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    """
    Try sigmoid (Platt) and isotonic; pick the one with lowest OOF Brier.
    Note: isotonic can overfit on small samples; prefer sigmoid if N<1000. :contentReference[oaicite:5]{index=5}
    """
    methods = ["sigmoid", "isotonic"]
    scores = {}
    models = {}
    for m in methods:
        clf = CalibratedClassifierCV(estimator=base_pipeline, method=m, cv=cv)
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        scores[m] = brier_score_loss(y, p)
        models[m] = clf
    best = min(scores, key=scores.get)
    return models[best], scores
