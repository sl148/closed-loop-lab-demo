"""The agent's tool rules match the fixture the web page's live agent is checked against
(scripts/export_kb.py writes it; the page's selfTest() runs the same cases in JavaScript)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "agent_tool_cases.json"


@pytest.mark.skipif(not (ROOT / "data" / "kb" / "chunks.jsonl").exists(), reason="document archive not present")
def test_python_tools_still_match_the_fixture(monkeypatch):
    import sys

    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_kb import cases

    assert cases() == json.loads(FIXTURE.read_text(encoding="utf-8"))
