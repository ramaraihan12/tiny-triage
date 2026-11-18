from pathlib import Path

from tiny_triage.labels import CATEGORIES, load_map, map_labels

MAP = {
    "bug": ["Type: Bug"],
    "feature": ["enhancement", "feature"],
    "question": ["question"],
    "docs": ["docs"],
    "needs_info": ["needs info"],
}


def test_single_category():
    m = map_labels(MAP, ["enhancement", "p2"])
    assert (m.status, m.category, m.needs_info) == ("ok", "feature", "no")


def test_two_labels_same_class_is_fine():
    assert map_labels(MAP, ["enhancement", "feature"]).category == "feature"


def test_ambiguous_across_classes_is_skipped():
    m = map_labels(MAP, ["Type: Bug", "enhancement"])
    assert m.status == "ambiguous" and m.category is None


def test_no_category_label():
    assert map_labels(MAP, ["help wanted"]).status == "none"


def test_needs_info_is_independent_and_case_insensitive():
    m = map_labels(MAP, ["TYPE: BUG", "Needs Info"])
    assert (m.category, m.needs_info) == ("bug", "yes")


def test_needs_info_unknown_when_repo_has_no_such_label():
    m = map_labels({"bug": ["bug"]}, ["bug"])
    assert m.category == "bug" and m.needs_info is None


def test_committed_label_map_is_well_formed():
    repos = load_map(Path(__file__).parents[1] / "data" / "label_map.yml")
    assert len(repos) >= 25
    for repo, classes in repos.items():
        assert "/" in repo
        assert "bug" in classes, repo
        assert set(classes) <= set(CATEGORIES) | {"needs_info"}
        assert all(isinstance(v, list) and v for v in classes.values())
