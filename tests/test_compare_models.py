from __future__ import annotations

import numpy as np

from compare.traditional.fldepth import estimate_depth as estimate_fldepth
from compare.traditional.fwdet_v2 import estimate_depth as estimate_fwdet_v2
from compare.traditional.tsa import estimate_depth as estimate_tsa


def test_named_compare_models_are_finite_and_support_bounded() -> None:
    rows, columns = np.indices((21, 21))
    terrain = 0.03 * (rows - 10) ** 2 + 0.02 * (columns - 10) ** 2
    support = (rows - 10) ** 2 + (columns - 10) ** 2 <= 49
    for estimator in (estimate_fwdet_v2, estimate_tsa, estimate_fldepth):
        depth = estimator(support, terrain)
        assert depth.shape == terrain.shape
        assert np.isfinite(depth).all()
        assert np.all(depth[~support] == 0.0)
        assert np.any(depth[support] > 0.0)
