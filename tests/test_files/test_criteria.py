"""tests/test_criteria.py — Unit tests for the criteria gate engine."""

import pytest

from strategy.criteria import (
    Criterion,
    CriteriaGroup,
    build_criteria_groups,
    sync_criteria_groups,
)


# ── Criterion ─────────────────────────────────────────────────────────

class TestCriterion:
    def test_default_state(self):
        c = Criterion(name="foo")
        assert c.passing is False
        assert c.value is None
        assert c.required is True
        assert c.updated_at is None

    def test_update_passing(self):
        c = Criterion(name="foo")
        c.update(True, value=12.5)
        assert c.passing is True
        assert c.value == 12.5
        assert c.updated_at is not None

    def test_update_no_value(self):
        c = Criterion(name="foo")
        c.update(True)
        assert c.passing is True
        assert c.value is None

    def test_reset(self):
        c = Criterion(name="foo")
        c.update(True, 1.0)
        c.reset()
        assert c.passing is False
        assert c.value is None
        assert c.updated_at is None


# ── CriteriaGroup ─────────────────────────────────────────────────────

class TestCriteriaGroup:
    def _make_group(self) -> CriteriaGroup:
        return CriteriaGroup(
            name="EXEC",
            criteria=[
                Criterion(name="a", required=True),
                Criterion(name="b", required=True),
                Criterion(name="c", required=False),
            ],
        )

    def test_empty_group_semantics(self):
        # An empty group has nothing to block (required_passing = True)
        # but also nothing to confirm (all_passing = False).
        g = CriteriaGroup(name="EMPTY", criteria=[])
        assert g.required_passing is True
        assert g.all_passing is False

    def test_required_passing_when_optional_fails(self):
        g = self._make_group()
        g.update("a", True)
        g.update("b", True)
        g.update("c", False)
        assert g.required_passing is True
        assert g.all_passing is False

    def test_required_passing_blocks_on_required_fail(self):
        g = self._make_group()
        g.update("a", True)
        g.update("b", False)
        g.update("c", True)
        assert g.required_passing is False

    def test_failing_list(self):
        g = self._make_group()
        g.update("a", True)
        g.update("b", False)
        g.update("c", False)
        names = {c.name for c in g.failing}
        assert names == {"b", "c"}

    def test_get_missing(self):
        g = self._make_group()
        assert g.get("missing") is None

    def test_update_missing_is_noop(self):
        g = self._make_group()
        g.update("missing", True)  # must not raise
        assert g.failing  # still failing because nothing changed

    def test_reset_all(self):
        g = self._make_group()
        g.update("a", True, 5.0)
        g.reset_all()
        assert all(not c.passing and c.value is None for c in g.criteria)


# ── Factory ───────────────────────────────────────────────────────────

class TestBuildCriteriaGroups:
    def test_build_from_dicts(self):
        defs = [
            {
                "name": "ATTENTION",
                "criteria": [
                    {"name": "change_pct_min", "description": "% > 10"},
                    {"name": "volume_min",     "description": "vol > 50k"},
                    {"name": "dist_hod",       "required": False},
                ],
            },
        ]
        groups = build_criteria_groups(defs)
        assert len(groups) == 1
        g = groups[0]
        assert g.name == "ATTENTION"
        assert len(g.criteria) == 3
        assert g.criteria[2].required is False
        assert g.criteria[0].description == "% > 10"
        assert all(not c.passing for c in g.criteria)

    def test_build_empty(self):
        assert build_criteria_groups([]) == []

    def test_sync_preserves_matching_state_and_removes_old_criteria(self):
        old = CriteriaGroup(
            name="ATTENTION",
            criteria=[
                Criterion(name="change_pct_min", passing=True, value=12.3),
                Criterion(name="roc_positive", passing=True, value=0.5),
            ],
        )
        defs = [
            {
                "name": "ATTENTION",
                "criteria": [
                    {"name": "change_pct_min", "description": "% > 10"},
                    {"name": "volume_min", "description": "vol > 50k"},
                ],
            },
        ]

        groups = sync_criteria_groups([old], defs)

        attention = groups[0]
        assert attention.get("change_pct_min").passing is True
        assert attention.get("change_pct_min").value == 12.3
        assert attention.get("volume_min") is not None
        assert attention.get("roc_positive") is None
