"""Fitted entry posterior — Upgrade 1 of the entry redesign (2026-08-23).

Replaces the tape/OFI AND-of-floors with a single fitted probability:
P(level approach resolves up-first: +2% before -3% within 120s | evidence).
The gate stack's signals become graded features weighted by a logistic
fit on the level-approach corpus (logs/research/level_approach_labels.csv,
18,447 events over 30 symbol-days); the entry threshold is chosen offline
by expected value vs round-trip cost, not by feel.

Single-source contract (the F10 lesson, applied to features):
  - ``FEATURES`` + ``encode_features`` define the feature vector ONCE.
    research/fit_entry_posterior.py imports them for the offline fit;
    ``score_entry`` uses them live. The two can never drift.
  - The artifact (entry_posterior.json) stores the feature names it was
    fitted with; ``EntryPosterior.load`` refuses a name mismatch rather
    than silently mis-mapping coefficients.

Runtime is pure Python (math only — no numpy on the live path).

Known train/serve gaps (accepted, documented):
  - The corpus labels band RE-ENTRY instants; live, the score is read on
    every in-band tick and the first threshold crossing fires. OOF EV is
    therefore an approach-level estimate; PnL truth comes from replay A/B.
  - The armed_* corpus proxy is any-true over the trailing 10s of samples
    regardless of price location; the live latches stamp in-band only and
    clear on floor breaks (strictly stricter).
  - Scoring is restricted to rel_to_level <= SIM_LEVEL_CROSS_TOLERANCE_PCT
    (the corpus band tolerance): the model never extrapolates into the
    extended zone whose chased fills the 08-05/08-06 diagnosis documented.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from exec.order_flow import OrderFlowTracker

# ---------------------------------------------------------------------------
# Feature specification — the ONLY definition, shared by fit and live paths
# ---------------------------------------------------------------------------
# Raw inputs (keys of the dict fed to encode_features):
#   speed          directional prints/sec        (tracker.speed_gate)
#   ratio          buy share of fast kernel      (tracker.ratio_gate)
#   volimb         30s dollar-flow imbalance     (tracker.trade_imbalance)
#   accelz         accel significance z          (tracker.accel_gate_z)
#   accel_ready    0/1                           (tracker.accel_gate_ready)
#   activity       prints in activity window     (tracker.activity_count)
#   ofi_ema10      depth-normalized EMA OFI      (tracker.ofi_norm_gate)
#   depth_active   0/1                           (tracker.depth_active)
#   rel_to_level   last_price/setup_level - 1
#   progress15     trailing 15s price progress   (tracker.price_progress)
#   progress_ready 0/1                           (price_progress is not None)
#   armed_speed/ratio/volimb/ofi  0/1 latches    (gate_last_true_age <= TTL)
#
# Not-ready/no-depth semantics: the value term is zeroed and the paired
# indicator carries the "evidence absent" information — the fit learns the
# cost of missing evidence instead of a hand-picked vacuous pass.

FEATURES: tuple[str, ...] = (
    "log1p_speed",
    "ratio",
    "volimb",
    "accelz",
    "accel_ready",
    "log1p_activity",
    "ofi_ema10",
    "depth_active",
    "rel_level_pct",
    "progress_pct",
    "progress_ready",
    "armed_speed",
    "armed_ratio",
    "armed_volimb",
    "armed_ofi",
)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def encode_features(raw: dict) -> list[float]:
    """Raw input dict -> feature vector in ``FEATURES`` order.

    Clips bound the heavy-tailed reads to the corpus' well-populated
    support so a single wild sample cannot saturate the logit.
    """
    accel_ready = 1.0 if raw["accel_ready"] else 0.0
    depth = 1.0 if raw["depth_active"] else 0.0
    prog_ready = 1.0 if raw["progress_ready"] else 0.0
    return [
        math.log1p(_clip(float(raw["speed"]), 0.0, 200.0)),
        float(raw["ratio"]),
        float(raw["volimb"]),
        _clip(float(raw["accelz"]), -6.0, 6.0) * accel_ready,
        accel_ready,
        math.log1p(_clip(float(raw["activity"]), 0.0, 2000.0)),
        _clip(float(raw["ofi_ema10"]), -3.0, 3.0) * depth,
        depth,
        _clip(float(raw["rel_to_level"]), -0.08, 0.02) * 100.0,
        _clip(float(raw["progress15"]), -0.05, 0.05) * 100.0 * prog_ready,
        prog_ready,
        1.0 if raw["armed_speed"] else 0.0,
        1.0 if raw["armed_ratio"] else 0.0,
        1.0 if raw["armed_volimb"] else 0.0,
        1.0 if raw["armed_ofi"] else 0.0,
    ]


def tracker_features(
    ofi_tracker: "OrderFlowTracker",
    last_price: float,
    setup_level: float,
) -> dict:
    """Assemble the raw feature dict from live tracker state.

    Every read is a gate-source property (the same single-source reads the
    AND-stack used), so offline corpus values and live values agree by
    construction — ofi_tune_build.py sampled these exact properties.
    """
    import config as _config

    ttl = _config.SIM_ARM_EVIDENCE_TTL_SECS

    def _armed(gate: str) -> bool:
        age = ofi_tracker.gate_last_true_age(gate)
        return age is not None and age <= ttl

    depth = ofi_tracker.depth_active
    progress = ofi_tracker.price_progress
    return {
        "speed": ofi_tracker.speed_gate,
        "ratio": ofi_tracker.ratio_gate,
        "volimb": ofi_tracker.trade_imbalance,
        "accelz": ofi_tracker.accel_gate_z,
        "accel_ready": ofi_tracker.accel_gate_ready,
        "activity": ofi_tracker.activity_count,
        "ofi_ema10": ofi_tracker.ofi_norm_gate if depth else 0.0,
        "depth_active": depth,
        "rel_to_level": last_price / setup_level - 1.0,
        "progress15": progress if progress is not None else 0.0,
        "progress_ready": progress is not None,
        "armed_speed": _armed("speed"),
        "armed_ratio": _armed("ratio"),
        "armed_volimb": _armed("vol_flow"),
        # No-depth = vacuous armed_ofi, mirroring the corpus proxy and the
        # armed-band consumer (depth-inactive == not applicable).
        "armed_ofi": _armed("ofi") if depth else True,
    }


# ---------------------------------------------------------------------------
# Model artifact
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "entry_posterior.json")

logger = logging.getLogger(__name__)

# These names change character week to week; a fit from a month-old corpus
# is a different market. Warn (never fail — replays of old days are fine).
STALE_WARN_DAYS = 14


class EntryPosterior:
    """Standardized-logistic scorer loaded from a JSON artifact."""

    __slots__ = ("features", "mean", "scale", "coef", "intercept",
                 "threshold", "meta")

    def __init__(self, blob: dict) -> None:
        self.features = tuple(blob["features"])
        if self.features != FEATURES:
            raise ValueError(
                "entry_posterior artifact feature mismatch: "
                f"artifact={self.features} code={FEATURES} — refit with "
                "research/fit_entry_posterior.py before using this regime"
            )
        self.mean = [float(x) for x in blob["mean"]]
        self.scale = [float(x) for x in blob["scale"]]
        self.coef = [float(x) for x in blob["coef"]]
        self.intercept = float(blob["intercept"])
        self.threshold = float(blob["threshold"])
        self.meta = blob.get("meta", {})
        k = len(self.features)
        if not (len(self.mean) == len(self.scale) == len(self.coef) == k):
            raise ValueError("entry_posterior artifact length mismatch")

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "EntryPosterior":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def score(self, raw: dict) -> float:
        """P(up-first | evidence) for one raw feature dict."""
        x = encode_features(raw)
        z = self.intercept
        for i, coef in enumerate(self.coef):
            s = self.scale[i]
            if s > 0:
                z += coef * (x[i] - self.mean[i]) / s
        # Overflow-safe sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)


_model_lock = threading.Lock()
_model: Optional[EntryPosterior] = None
_model_path: Optional[str] = None


def get_model() -> EntryPosterior:
    """Load-once accessor for the live/replay path.

    Honors SIM_POSTERIOR_MODEL_PATH (env) for A/B experiments; raises with
    instructions when the artifact is absent — the posterior regime must
    fail loudly, never fall back to another regime silently.
    """
    global _model, _model_path
    path = os.environ.get("SIM_POSTERIOR_MODEL_PATH", "") or DEFAULT_MODEL_PATH
    with _model_lock:
        if _model is None or _model_path != path:
            try:
                _model = EntryPosterior.load(path)
            except FileNotFoundError:
                raise RuntimeError(
                    f"SIM_ENTRY_REGIME=posterior but no model artifact at "
                    f"{path} — run research/fit_entry_posterior.py first"
                ) from None
            _model_path = path
            created = _model.meta.get("created", "")
            try:
                age = (datetime.now(tz=timezone.utc)
                       - datetime.fromisoformat(created)).days
            except ValueError:
                age = None
            if age is None or age > STALE_WARN_DAYS:
                logger.warning(
                    "entry_posterior artifact is %s days old (created %s) — "
                    "these names drift; refit with research/"
                    "fit_entry_posterior.py --write",
                    "?" if age is None else age, created or "unknown",
                )
        return _model


def entry_threshold() -> float:
    """Fitted EV threshold, overridable via config for experiments."""
    import config as _config
    override = getattr(_config, "SIM_POSTERIOR_THRESHOLD", 0.0)
    if override and override > 0:
        return float(override)
    return get_model().threshold


def score_entry(
    ofi_tracker: "OrderFlowTracker",
    last_price: float,
    setup_level: float,
) -> float:
    """One-call live read: P(up-first) from current tracker state."""
    return get_model().score(
        tracker_features(ofi_tracker, last_price, setup_level)
    )
