# research/cli/feature_ablate.py
import argparse, sys, pathlib, numpy as np, pandas as pd
from typing import List, Tuple

# --- import guards ---
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
from sklearn.metrics import roc_auc_score, brier_score_loss

def _oof_preds(df: pd.DataFrame, feats: List[str], embargo: int) -> np.ndarray:
    pipe, cont, cat, cyc = build_pipeline(df)
    X = df[feats].copy()
    y = df["y"].astype(int).values
    order = df["order_idx"].values if "order_idx" in df.columns else np.arange(len(df))

    cv = list(PurgedKFold(n_splits=5, embargo=embargo).split(X, y, order=order))
    oof = np.zeros(len(df), dtype=float)
    for tr, te in cv:
        y_tr = y[tr]
        # single-class guard
        if np.unique(y_tr).size < 2:
            oof[te] = float(np.mean(y_tr))
            continue
        model, _ = fit_with_best_calibration(pipe, X.iloc[tr], y_tr, cv=3)
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof

def _scores(y: np.ndarray, p: np.ndarray):
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    br  = brier_score_loss(y, p)
    ec  = ece(y, p)
    return auc, br, ec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data/dataset.parquet")
    ap.add_argument("--embargo_bars", type=int, default=48)
    ap.add_argument("--out_csv", default="research/results/ablation.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.data).dropna(subset=["y","opened_at"]).reset_index(drop=True)
    # build once to know the feature list the pipeline will actually use
    _, cont, cat, cyc = build_pipeline(df)
    features = cont + cat + cyc
    if not features:
        raise SystemExit("No usable features found.")

    y = df["y"].astype(int).values

    # Baseline OOF
    base_oof = _oof_preds(df, features, args.embargo_bars)
    base_auc, base_brier, base_ece = _scores(y, base_oof)

    rows = []
    rows.append({
        "feature": "__BASELINE__",
        "AUC": base_auc, "Brier": base_brier, "ECE": base_ece,
        "dAUC": 0.0, "dBrier": 0.0, "dECE": 0.0
    })

    # Drop-one ablation
    for f in features:
        fset = [x for x in features if x != f]
        oof = _oof_preds(df, fset, args.embargo_bars)
        auc, brier, ec = _scores(y, oof)
        rows.append({
            "feature": f,
            "AUC": auc, "Brier": brier, "ECE": ec,
            "dAUC": auc - base_auc,
            "dBrier": brier - base_brier,
            "dECE": ec - base_ece
        })

    out = pd.DataFrame(rows)
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved ablation table → {args.out_csv}")

if __name__ == "__main__":
    main()
