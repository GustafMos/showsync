"""Shared pytest fixtures. `fixtures_dir` builds the quick synthetic scenario folders once."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def fixtures_dir():
    from tests import synth
    out = ROOT / "tests" / "_out" / "fixtures"
    man = out / "manifest.json"
    if man.exists():
        try:
            m = json.loads(man.read_text())
            if m.get("synth_version") == synth.SYNTH_VERSION and m.get("quick") is True:
                return out
        except Exception:
            pass
    synth.make_fixtures(out, quick=True)
    return out
