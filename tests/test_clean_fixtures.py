import json
from pathlib import Path

import make_clean_fixtures


def test_fixture_file_matches_python_cleaning():
    path = Path(__file__).parent / "fixtures" / "clean_cases.json"
    assert json.loads(path.read_text(encoding="utf-8")) == make_clean_fixtures.make()
