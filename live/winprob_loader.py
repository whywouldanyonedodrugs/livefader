# live/winprob_loader.py
from __future__ import annotations
"""
Win-probability loader + scorer.

Supported artifacts:
  1) Scikit-learn pipeline/estimator with predict_proba (recommended; may be a CalibratedClassifierCV).
  2) Legacy dict bundle: {"pipeline": estimator, "features": [...]}
  3) Statsmodels results object (has .model.exog_names and .predict).

Notes:
- We align incoming feature dicts to the pipeline's expected feature names via
  `feature_names_in_` when available (sklearn estimators/pipelines commonly set it
  once fit with a DataFrame). If not available, we pass columns as-is.  # see sklearn docs
- For statsmodels we add an explicit constant using `sm.add_constant` before predict.  # statsmodels predict + add_constant
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("winprob")


@dataclass
class _Loaded:
    kind: str                   # "sklearn" | "sklearn_dict" | "statsmodels"
    model: Any
    features: Sequence[str]     # expected raw feature names (preprocessing inputs)
    path: Optional[Path]


class WinProbScorer:
    def __init__(self, paths: Optional[Sequence[str]] = None):
        """
        Search (in order) for a compatible artifact and load it.
        """
        self.paths = [Path(p) for p in (paths or [
            "win_probability_model.pkl",
            "win_probability_model.joblib",
        ])]
        self._loaded: Optional[_Loaded] = None

        # diagnostics
        self._last_input_df: Optional[pd.DataFrame] = None
        self._last_raw: Optional[float] = None

        self._try_load_any()

    # ---------------------------- Public API ----------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def kind(self) -> str:
        return self._loaded.kind if self._loaded else "none"

    @property
    def expected_features(self) -> Sequence[str]:
        return self._loaded.features if self._loaded else []

    @property
    def path(self) -> Optional[str]:
        return str(self._loaded.path) if self._loaded and self._loaded.path else None

    def score(self, feat_dict: dict) -> float:
        """
        Return calibrated P(win) in [0,1] for a single example.
        Safe on missing/extra keys: missing→0.0 (for numeric inputs).
        """
        if not self._loaded:
            return 0.0

        model = self._loaded.model
        kind = self._loaded.kind
        cols = list(self._loaded.features) if self._loaded.features else None

        X = pd.DataFrame([feat_dict], index=[0])
        self._last_input_df = X.copy()

        # Align columns for sklearn if we know the inputs; fill missing with 0.0
        if kind.startswith("sklearn") and cols:
            X = X.reindex(columns=cols, fill_value=0.0)

        try:
            if kind == "statsmodels":
                # add explicit constant for statsmodels design matrix
                import statsmodels.api as sm
                Xc = sm.add_constant(X.reindex(columns=cols, fill_value=0.0) if cols else X,
                                     has_constant="add", prepend=True)
                pred = model.predict(Xc)
                p = float(pred.iloc[0] if hasattr(pred, "iloc") else pred[0])
                self._last_raw = p
            else:
                # sklearn path (pipeline or estimator with predict_proba)
                proba = model.predict_proba(X)[:, 1]
                p = float(proba[0])
                self._last_raw = p
        except Exception as e:
            LOG.warning("WinProb scoring failed: %s", e)
            return 0.0

        # clamp to [0,1]
        return float(max(0.0, min(1.0, p)))

    def diag(self) -> str:
        """
        JSON diagnostics: artifact kind/path, expected vs input columns, last raw output.
        Handy to attach to a Telegram command (/wpdiag).
        """
        exp = list(self.expected_features) if self.is_loaded else None
        inp_cols = list(self._last_input_df.columns) if self._last_input_df is not None else None
        missing = [c for c in (exp or []) if inp_cols is not None and c not in inp_cols]
        return json.dumps({
            "is_loaded": self.is_loaded,
            "kind": self.kind,
            "path": self.path,
            "expected_features_count": len(exp or []),
            "expected_features_sample": (exp or [])[:12],
            "input_columns": inp_cols,
            "missing_expected": missing,
            "last_raw": (None if self._last_raw is None else float(self._last_raw)),
        }, indent=2)

    # --------------------------- Loading logic --------------------------

    def _try_load_any(self):
        for p in self.paths:
            if not p.exists():
                continue
            try:
                obj = joblib.load(p)
                loaded = self._coerce(obj, p)
                if loaded:
                    self._loaded = loaded
                    LOG.info("WinProb loaded: kind=%s file=%s features=%d",
                             loaded.kind, p.name, len(loaded.features))
                    return
            except Exception as e:
                LOG.warning("Failed to load %s: %s", p, e)
        LOG.warning("No compatible win-probability model found. wp defaults to 0.0")

    def _coerce(self, obj: Any, path: Path) -> Optional[_Loaded]:
        """
        Try to interpret the loaded object in known formats.
        Discovery order prefers sklearn, then dict bundle, then statsmodels.
        """
        # --- 1) Plain sklearn estimator/pipeline with predict_proba ---
        if hasattr(obj, "predict_proba"):
            feats = self._discover_features_from_sklearn(obj)
            return _Loaded(kind="sklearn", model=obj, features=feats, path=path)

        # --- 2) Legacy dict bundle {"pipeline": estimator, "features": [...] } ---
        if isinstance(obj, dict) and "pipeline" in obj:
            est = obj["pipeline"]
            if hasattr(est, "predict_proba"):
                feats = list(obj.get("features") or []) or self._discover_features_from_sklearn(est)
                return _Loaded(kind="sklearn_dict", model=est, features=feats, path=path)

        # --- 3) Statsmodels results wrapper (has .model and .predict) ---
        if hasattr(obj, "model") and hasattr(obj, "predict"):
            try:
                exog = getattr(obj.model, "exog_names", None)
                if isinstance(exog, (list, tuple)):
                    feats = [c for c in exog if c != "const"]
                    return _Loaded(kind="statsmodels", model=obj, features=feats, path=path)
            except Exception:
                pass

        return None

    def _discover_features_from_sklearn(self, est: Any) -> list[str]:
        """
        Best-effort to recover the input feature names a fitted sklearn pipeline/estimator expects.
        Tries, in order:
          - estimator.feature_names_in_
          - pipeline.named_steps['preprocess'].feature_names_in_
        Returns [] if not discoverable (we'll feed columns as-is).
        """
        feats = []
        # 1) direct attribute on the estimator/pipeline
        for attr in ("feature_names_in_",):
            if hasattr(est, attr):
                try:
                    vals = list(getattr(est, attr))
                    if vals:
                        feats = [str(v) for v in vals]
                        return feats
                except Exception:
                    pass

        # 2) look into a common 'preprocess' step (ColumnTransformer)
        try:
            steps = getattr(est, "named_steps", {}) or {}
            preprocess = steps.get("preprocess")
            if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
                vals = list(getattr(preprocess, "feature_names_in_"))
                if vals:
                    feats = [str(v) for v in vals]
                    return feats
        except Exception:
            pass

        return feats
