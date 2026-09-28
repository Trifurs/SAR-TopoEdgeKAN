from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def production_config():
    from tools.evaluate_sar_topo_edge_kan import embed_source_fingerprints
    from utils.config import load_config

    return embed_source_fingerprints(load_config(ROOT / "configs" / "sar_topo_edge_kan.xml"))
