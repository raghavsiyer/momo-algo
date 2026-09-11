"""
strategy/criteria.py — Boolean gate engine (parallel to scoring.py).

A *Criterion* is a pass/fail check that does NOT contribute to the
confidence score.  Criteria are organised into *CriteriaGroup*s that
sit alongside the ``VariableGroup``s in ``CompositeEvaluation``.

Use cases
---------
* Stage-1 attention gates (price%, volume, distance from HOD, …)
* Trend-direction visibility flags (close>VWAP, MACD>0, …)
* Stage-4 execution gates that block entry until resolved

Criteria are intentionally minimal — no quality function, no weight,
just a name, a current ``passing`` boolean, and an optional last-known
``value`` for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

__all__ = [
    "Criterion",
    "CriteriaGroup",
    "build_criteria_groups",
    "sync_criteria_groups",
]


# ── Criterion ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class Criterion:
    """A single pass/fail gate with optional last-known value."""

    name: str
    description: str = ""
    required: bool = True               # if False, advisory only
    passing: bool = False
    value: Optional[float] = None       # last raw value (for display)
    updated_at: Optional[datetime] = None

    def update(self, passing: bool, value: Optional[float] = None) -> None:
        self.passing = bool(passing)
        if value is not None:
            self.value = float(value)
        self.updated_at = datetime.now(tz=timezone.utc)

    def reset(self) -> None:
        """Reset to not-yet-evaluated state (passing=False, value=None)."""
        self.passing = False
        self.value = None
        self.updated_at = None


# ── CriteriaGroup ─────────────────────────────────────────────────────

@dataclass(slots=True)
class CriteriaGroup:
    """A named set of criteria.  Does not contribute to scoring."""

    name: str
    criteria: list[Criterion] = field(default_factory=list)

    @property
    def all_passing(self) -> bool:
        """True when every criterion (required or not) is passing."""
        return all(c.passing for c in self.criteria) if self.criteria else False

    @property
    def required_passing(self) -> bool:
        """True when every *required* criterion is passing.

        Empty group → True (no required gates means nothing to block).
        """
        return all(c.passing for c in self.criteria if c.required)

    @property
    def failing(self) -> list[Criterion]:
        """List of criteria currently failing."""
        return [c for c in self.criteria if not c.passing]

    def get(self, name: str) -> Optional[Criterion]:
        for c in self.criteria:
            if c.name == name:
                return c
        return None

    def update(self, name: str, passing: bool, value: Optional[float] = None) -> None:
        """Update a single criterion by name.  No-op if name not found."""
        c = self.get(name)
        if c is not None:
            c.update(passing, value)

    def reset_all(self) -> None:
        for c in self.criteria:
            c.reset()


# ── Factory ───────────────────────────────────────────────────────────

def build_criteria_groups(group_defs: list[dict]) -> list[CriteriaGroup]:
    """Construct ``CriteriaGroup`` instances from config-style dicts.

    Each dict::

        {
            "name": "ATTENTION",
            "criteria": [
                {"name": "change_pct_min", "description": "% change > 10%"},
                {"name": "volume_min",     "description": "vol > 50k"},
                {"name": "dist_from_hod_atr_max",
                 "description": "within N ATRs of HOD",
                 "required": False},
            ],
        }
    """
    groups: list[CriteriaGroup] = []
    for gd in group_defs:
        criteria = [
            Criterion(
                name=cd["name"],
                description=cd.get("description", ""),
                required=cd.get("required", True),
            )
            for cd in gd.get("criteria", [])
        ]
        groups.append(CriteriaGroup(name=gd["name"], criteria=criteria))
    return groups


def sync_criteria_groups(existing: list[CriteriaGroup] | None,
                         group_defs: list[dict]) -> list[CriteriaGroup]:
    """Rebuild criteria definitions while preserving matching live state."""
    current = build_criteria_groups(group_defs)
    old_by_group = {g.name: g for g in existing or []}
    for group in current:
        old_group = old_by_group.get(group.name)
        if old_group is None:
            continue
        old_by_name = {c.name: c for c in old_group.criteria}
        for criterion in group.criteria:
            old = old_by_name.get(criterion.name)
            if old is None:
                continue
            criterion.passing = old.passing
            criterion.value = old.value
            criterion.updated_at = old.updated_at
    return current
