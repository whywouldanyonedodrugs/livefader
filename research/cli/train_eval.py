# research/cli/train_eval.py
import argparse, pathlib, sys
import numpy as np
import pandas as pd
import joblib

# import guards
THIS = pathlib.Path(__file__).resolve()
PKG_ROOT = THIS.parents[1]
SRC_ROOT = THIS.parents[2]
for p in (str(SRC_ROOT), str(PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.modeling.pipeline import build_pipeline
from research.modeling.splits import PurgedKFold
from research.modeling.calibration import fit_with_best_calibration
from research.modeling.metrics import ece
from sklearn.metrics import roc_auc_score, brier_score_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data/dataset.parquet")
    ap.add_argument("--embargo_bars", type=int, default=48)
    ap.add_argument("--model_out", default="win_probability_model.pkl")
    ap.add_argument("--report_dir", default="research/results")
    args = ap.parse_args()

    df = pd.read_parquet(args.data).dropna(subset=["y","opened_at"]).reset_index(drop=True)

    # Build pipeline and get RAW feature names (not engineered names)
    pipe, cont, cat, cyc = build_pipeline(df)
    feature_names = [c for c in (cont + cat + cyc) if c in df.columns]

    if not feature_names:
        raise SystemExit("No usable features present in dataset.")

    X = df[feature_names].copy()
    y = df["y"].astype(int).values
    order = df["order_idx"].values if "order_idx" in df.columns else np.arange(len(df))

    # OOF predictions with PurgedKFold + embargo
    cv = list(PurgedKFold(n_splits=5, embargo=args.embargo_bars).split(X, y, order=order))
    oof = np.zeros(len(df), dtype=float)

    for tr, te in cv:
        y_tr = y[tr]
        # guard single-class folds
        if np.unique(y_tr).size < 2:
            oof[te] = float(np.mean(y_tr))
            continue
        model, scores = fit_with_best_calibration(pipe, X.iloc[tr], y_tr, cv=3)
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]

    # Metrics
    auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else np.nan
    brier = brier_score_loss(y, oof)
    prior = np.mean(y)
    brier_base = prior*(1-prior) + (1-prior)*prior  # ~p(1-p)*2 for naive constant predictor
    print(f"OOF AUC={auc:.4f}  Brier={brier:.5f}  (baseline approx={brier_base:.5f})  N={len(df)}")
    print(f"ECE={ece(y, oof):.5f}")

    # Save OOF for downstream analysis
    out_dir = pathlib.Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": y, "prob": oof}).to_csv(out_dir / "winprob_oof.csv", index=False)

    # Fit final calibrated model on FULL data (choose best calibration inside helper)
    final_model, cal_scores = fit_with_best_calibration(pipe, X, y, cv=3)
    print(f"Calibration scores: {cal_scores} → using {min(cal_scores, key=cal_scores.get)}")

    # Persist model bundle (simple dict keeps pickling robust)
    bundle = {"pipeline": final_model, "feature_names": feature_names}
    joblib.dump(bundle, args.model_out)
    print(f"Saved calibrated model to {args.model_out}")

    # Basic reports
    (out_dir / "winprob_calibration.csv").write_text(
        "method,brier\n" + "\n".join([f"{k},{v}" for k, v in cal_scores.items()])
    )
    print(f"Reports written to {args.report_dir}")

if __name__ == "__main__":
    main()
