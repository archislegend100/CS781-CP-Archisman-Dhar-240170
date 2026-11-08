import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import check


def test_all():
    for c in check.CHECKS:
        c()
    bad = [(n, m) for n, ok, m in check.RESULTS if not ok]
    assert not bad, bad
