# research/cli/train_eval.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd

# ---- robust sys.path guard ----
import sys, pathlib
THIS = pathlib.Path(__file__).resolve()
PKG_ROOT = THIS.parents[1]   # .../research
SRC_ROOT = THIS.parents[2]   # .../src
for p in (str(SRC_ROOT), str(PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.modeling.splits import PurgedKFold
from research.modeling.pipeline import build_pipeline
from research.modeling.calibration import fit_with_best_calibration
from research.modeling.metrics import ece
from research.modeling.multiple_tests import deflated_sharpe_ratio
from research.reports.winprob_eval import write_reports
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.dummy import DummyClassifier

class CalibratedModelBundle:
    """
    Adapter so live code can keep using:
      - .model.exog_names (with a 'const')
      - .predict(X_dataframe) -> probabilities
    """
    def __init__(self, sk_model, feature_names):
        self.sk_model = sk_model
        class _ModelLike:
            def __init__(self, names): self.exog_names = ["const"] + names
        self.model = _ModelLike(feature_names)

    def predict(self, X_df: pd.DataFrame):
        return self.sk_model.predict_proba(X_df)[:, 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data/dataset.parquet")
    ap.add_argument("--embargo_bars", type=int, default=48)
    ap.add_argument("--model_out", default="win_probability_model.pkl")
    ap.add_argument("--report_dir", default="research/results")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["y","opened_at"]).reset_index(drop=True)

    pipe, cont, cat, cyc = build_pipeline(df)
    feature_names = cont + cat + cyc
    X = df[feature_names].copy()
    y = df["y"].astype(int).values
    order = df["order_idx"].values if "order_idx" in df.columns else np.arange(len(df))

    # Purged K-Fold with embargo
    cv = list(PurgedKFold(n_splits=5, embargo=args.embargo_bars).split(X, y, order=order))

    # OOF probabilities with guards for single-class folds
    oof = np.zeros(len(df), dtype=float)
    for tr, te in cv:
        y_tr = y[tr]
        if np.unique(y_tr).size < 2:
            # fallback: constant predictor = in-fold base rate
            oof[te] = float(np.mean(y_tr))
            continue
        try:
            model, _ = fit_with_best_calibration(pipe, X.iloc[tr], y_tr, cv=3)
            oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        except Exception as e:
            # last-resort guard: use base rate if calibration/pipeline fails
            print(f"[WARN] fold failed ({e}); using base-rate for OOF slice")
            oof[te] = float(np.mean(y_tr))

    auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else np.nan
    brier = brier_score_loss(y, oof)
    baseline = y.mean() * (1 - y.mean())
    print(f"OOF AUC={auc:.4f}  Brier={brier:.5f}  (baseline approx={baseline:.5f})  N={len(df)}")
    print(f"ECE={ece(y,oof):.5f}")

    edge = (2*y - 1).astype(float)
    dsr = deflated_sharpe_ratio(edge, sr_benchmark=0.0, trials=3)
    print(f"DSR (approx)={dsr:.3f}")

    # Final fit on full data (with calibration selection)
    if np.unique(y).size < 2:
        # dataset degenerate: constant model
        dummy = DummyClassifier(strategy="constant", constant=int(np.round(y.mean())))
        dummy.fit(X, y)
        final_model = dummy
        scores = {"sigmoid": brier, "isotonic": brier}
    else:
        final_model, scores = fit_with_best_calibration(pipe, X, y, cv=5)

    best = min(scores, key=scores.get)
    print(f"Calibration scores: {scores} → using {best}")

    bundle = CalibratedModelBundle(final_model, feature_names)
    joblib.dump(bundle, args.model_out)
    print(f"Saved calibrated model to {args.model_out}")

    os.makedirs(args.report_dir, exist_ok=True)
    write_reports(y, oof, args.report_dir)
    print(f"Reports written to {args.report_dir}")

if __name__ == "__main__":
    main()
