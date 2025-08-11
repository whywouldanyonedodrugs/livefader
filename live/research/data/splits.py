# research/modeling/splits.py
import numpy as np
from sklearn.model_selection import KFold

class PurgedKFold:
    """
    Time-aware split with an embargo to prevent leakage from overlapping labels.
    Provide 'order_idx' (monotone increasing by time).
    """
    def __init__(self, n_splits=5, embargo: int = 0):
        self.n_splits = n_splits
        self.embargo = int(embargo)

    def split(self, X, y=None, groups=None, order=None):
        order_idx = np.arange(len(X)) if order is None else np.asarray(order)
        assert order_idx.shape[0] == len(X), "order length must match X"
        kf = KFold(self.n_splits, shuffle=False)
        for train_mask, test_mask in kf.split(order_idx):
            test_idx = order_idx[test_mask]
            lo, hi = int(test_idx.min()), int(test_idx.max())
            # Purge + embargo region from train
            lo_purge = max(0, lo - self.embargo)
            hi_purge = min(order_idx.max(), hi + self.embargo)
            keep = (order_idx < lo_purge) | (order_idx > hi_purge)
            yield np.where(keep)[0], test_mask
