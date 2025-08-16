# live/winprob_loader.py
from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("winprob")

# -------------------------
# Unpickle shims (critical)
# -------------------------
class ModelBundle:  # shim for research-side dataclass
    pass

def _hour_cyc(X):
    import numpy as _np
    h = _np.asarray(X, dtype=float).reshape(-1, 1)
    return _np.hstack([_np.sin(2 * _np.pi * h / 24.0), _np.cos(2 * _np.pi * h / 24.0)])

def _dow_onehot(X):
    import numpy as _np
    d = _np.asarray(X, dtype=int).reshape(-1, 1)
    out = _np.zeros((d.shape[0], 7), dtype=float)
    m = (d >= 0) & (d < 7)
    out[_np.arange(d.shape[0])[m.ravel()], d[m].ravel()] = 1.0
    return out

def _install_unpickle_shims():
    """
    Ensure the names used when the model was saved exist at import-time now.
    Typical offenders: __main__._hour_cyc and research.modeling.pipeline._hour_cyc.
    """
    for modname in ["__main__", "research.modeling.pipeline"]:
        if modname not in sys.modules:
            sys.modules[modname] = types.ModuleType(modname)
        m = sys.modules[modname]
        if not hasattr(m, "_hour_cyc"):
            setattr(m, "_hour_cyc", _hour_cyc)
        if not hasattr(m, "_dow_onehot"):
            setattr(m, "_dow_onehot", _dow_onehot)
        if not hasattr(m, "ModelBundle"):
            setattr(m, "ModelBundle", ModelBundle)

_install_unpickle_shims()

# -------------------------
# Loader core
# -------------------------
@dataclass
class _Loaded:
    kind: str                   # "statsmodels" | "sklearn_bundle" | "sklearn_dict" | "sklearn_estimator"
    model: Any
    features: Sequence[str]
    path: Optional[Path] = None

class WinProbScorer:
    """
    Unified scorer for your research artifact:
      1) statsmodels Results (has .model.exog_names & .predict)  -> add constant, then predict
      2) research ModelBundle (has .feature_names & .calibrator) -> use calibrator.predict_proba
      3) dict {"pipeline": est, "features":[...]}                -> est.predict_proba
         or  {"calibrator": est, "features":[...]}
      4) plain sklearn estimator/pipeline with feature_names_in_
    """
    def __init__(self, paths: Optional[Sequence[str]] = None):
        self.paths = [Path(p) for p in (paths or [
            "win_probability_model.pkl",
            "win_probability_model.joblib",
        ])]
        self._loaded: Optional[_Loaded] = None
        self._try_load_any()

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def expected_features(self) -> Sequence[str]:
        return self._loaded.features if self._loaded else []

    @property
    def kind(self) -> str:
        return self._loaded.kind if self._loaded else "none"

    def _try_load_any(self):
        for p in self.paths:
            if not p.exists():
                continue
            try:
                obj = joblib.load(p)
            except Exception as e:
                LOG.warning("Failed to unpickle %s: %s", p, e)
                continue
            loaded = self._coerce(obj)
            if loaded:
                loaded.path = p
                self._loaded = loaded
                LOG.info("WinProb model loaded: kind=%s file=%s features=%d",
                         loaded.kind, p.name, len(loaded.features))
                return
            else:
                # Helpful trace: what did we actually unpickle?
                LOG.warning("Unrecognized model artifact in %s (type=%s).", p.name, type(obj).__name__)
        LOG.warning("No compatible win-probability model found. wp defaults to 0.0")

    def _coerce(self, obj: Any) -> Optional[_Loaded]:
        # 1) statsmodels Results
        try:
            import statsmodels.api as sm  # noqa: F401
            if hasattr(obj, "model") and hasattr(obj, "predict"):
                exog = getattr(obj.model, "exog_names", None)
                if isinstance(exog, (list, tuple)) and len(exog) > 0:
                    feats = [c for c in exog if c != "const"]
                    return _Loaded(kind="statsmodels", model=obj, features=feats)
        except Exception:
            pass

        # 2) research ModelBundle (attrs feature_names + calibrator.predict_proba)
        if hasattr(obj, "feature_names") and hasattr(obj, "calibrator") and hasattr(obj.calibrator, "predict_proba"):
            feats = list(getattr(obj, "feature_names"))
            return _Loaded(kind="sklearn_bundle", model=obj.calibrator, features=feats)

        # 3a) legacy sklearn dict {"pipeline": estimator, "features":[...]}
        if isinstance(obj, dict) and "features" in obj:
            feats = list(obj["features"])
            if "pipeline" in obj and hasattr(obj["pipeline"], "predict_proba"):
                return _Loaded(kind="sklearn_dict", model=obj["pipeline"], features=feats)
            if "calibrator" in obj and hasattr(obj["calibrator"], "predict_proba"):
                return _Loaded(kind="sklearn_dict", model=obj["calibrator"], features=feats)

        # 3b) object with ".pipeline" attribute (sometimes saved as a small wrapper)
        if hasattr(obj, "pipeline") and hasattr(obj.pipeline, "predict_proba"):
            feats = self._infer_features_from_estimator(obj.pipeline)
            if feats:
                return _Loaded(kind="sklearn_estimator", model=obj.pipeline, features=feats)

        # 4) plain sklearn estimator/pipeline
        if hasattr(obj, "predict_proba"):
            feats = self._infer_features_from_estimator(obj)
            if feats:
                return _Loaded(kind="sklearn_estimator", model=obj, features=feats)

        return None

    def _infer_features_from_estimator(self, est) -> Optional[Sequence[str]]:
        """
        Try hard to recover original input feature names for sklearn pipelines:
        - estimator.feature_names_in_
        - pipeline.named_steps['preprocess'].feature_names_in_
        - first ColumnTransformer-like step's feature_names_in_
        """
        # Direct
        feats = getattr(est, "feature_names_in_", None)
        if feats is not None:
            return list(map(str, feats))

        # Pipeline path
        named_steps = getattr(est, "named_steps", None)
        if isinstance(named_steps, dict):
            # common names: 'preprocess', 'pre', 'transform'
            for key in ["preprocess", "pre", "transform"]:
                if key in named_steps:
                    ct = named_steps[key]
                    feats = getattr(ct, "feature_names_in_", None)
                    if feats is not None:
                        return list(map(str, feats))
            # fallback: search any step with feature_names_in_
            for step in named_steps.values():
                feats = getattr(step, "feature_names_in_", None)
                if feats is not None:
                    return list(map(str, feats))

        # ColumnTransformer-like directly
        try:
            from sklearn.compose import ColumnTransformer  # noqa: F401
            if hasattr(est, "transformers_") and hasattr(est, "feature_names_in_"):
                return list(map(str, est.feature_names_in_))
        except Exception:
            pass

        LOG.warning("Could not infer input feature names from estimator; sklearn pipeline may not be usable.")
        return None

    def score(self, features: dict) -> float:
        """
        Return calibrated P(win) in [0,1] for a single example.
        Statsmodels path adds a constant; sklearn paths use predict_proba.
        """
        if not self._loaded:
            return 0.0

        kind = self._loaded.kind
        model = self._loaded.model
        cols = list(self._loaded.features)

        X = pd.DataFrame([features], index=[0]).reindex(columns=cols, fill_value=0.0)

        try:
            if kind == "statsmodels":
                import statsmodels.api as sm
                Xc = sm.add_constant(X.astype(float), prepend=True, has_constant="add")
                p = float(model.predict(Xc)[0])
            else:
                p = float(model.predict_proba(X.astype(float))[:, 1][0])
        except Exception as e:
            LOG.warning("WinProb scoring failed (%s); returning 0.0", e)
            return 0.0

        return float(np.clip(p, 0.0, 1.0))
