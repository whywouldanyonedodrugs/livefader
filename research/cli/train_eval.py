# research/cli/train_eval.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from ..modeling.splits import PurgedKFold
from ..modeling.pipeline import build_pipeline
from ..modeling.calibration import fit_with_best_calibration
from ..modeling.metrics import calibration_table, ece
from ..modeling.multiple_tests import deflated_sharpe_ratio
from ..reports.winprob_eval import write_reports

class CalibratedModelBundle:
    """
    Adapter so live code can continue using:
      - .model.exog_names (with a 'const')
      - .predict(X_dataframe) -> probabilities
    """
    def __init__(self, sk_model, feature_names):
        self.sk_model = sk_model
        class _ModelLike:
            def __init__(self, names): self.exog_names = ["const"] + names
        self.model = _ModelLike(feature_names)

    def predict(self, X_df: pd.DataFrame):
        # the live bot will add constant; we ignore it: our pipeline expects only feature cols
        return self.sk_model.predict_proba(X_df)[:, 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data/dataset.parquet")
    ap.add_argument("--embargo_bars", type=int, default=48, help="5m bars in 4h exit ≈ 48")
    ap.add_argument("--model_out", default="win_probability_model.pkl")
    ap.add_argument("--report_dir", default="research/results")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["y","opened_at"]).reset_index(drop=True)

    # Build pipeline on available columns
    pipe, cont, cat, cyc = build_pipeline(df)
    # Columns pipeline expects (in DF order)
    feature_names = []
    feature_names += cont
    feature_names += cat
    feature_names += cyc
    X = df[feature_names].copy()
    y = df["y"].astype(int).values
    order = df["order_idx"].values

    # Purged K-Fold with embargo
    cv = list(PurgedKFold(n_splits=5, embargo=args.embargo_bars).split(X, y, order=order))

    # OOF probabilities
    oof = np.zeros(len(df), dtype=float)
    for tr, te in cv:
        model, scores = fit_with_best_calibration(pipe, X.iloc[tr], y[tr], cv=3)
        oof[te] = model.predict_proba(X.iloc[te])[:,1]

    auc = roc_auc_score(y, oof) if len(np.unique(y))>1 else np.nan
    brier = brier_score_loss(y, oof)
    baseline = y.mean() * (1 - y.mean())
    print(f"OOF AUC={auc:.4f}  Brier={brier:.5f}  (baseline approx={baseline:.5f})  N={len(df)}")
    print(f"ECE={ece(y,oof):.5f}")

    # Very rough DSR on a dummy equity curve = (y - oof) proxy not meaningful;
    # Instead, approximate per-trade edge by (2*y-1) and cumulate; you’ll replace with true P&L stream.
    edge = (2*y - 1).astype(float)
    dsr = deflated_sharpe_ratio(edge, sr_benchmark=0.0, trials=3)
    print(f"DSR (approx)={dsr:.3f}")

    # Fit final model on full data using best calibration = sigmoid (safer on small N) vs isotonic if better OOS
    final_model, scores = fit_with_best_calibration(pipe, X, y, cv=5)
    print(f"Calibration scores: {scores} → using {min(scores, key=scores.get)}")

    # Export bundle that mimics statsmodels interface expected by live bot
    bundle = CalibratedModelBundle(final_model, feature_names)
    joblib.dump(bundle, args.model_out)
    print(f"Saved calibrated model to {args.model_out}")

    # Write research reports
    os.makedirs(args.report_dir, exist_ok=True)
    write_reports(y, oof, args.report_dir)
    print(f"Reports written to {args.report_dir}")

if __name__ == "__main__":
    main()
