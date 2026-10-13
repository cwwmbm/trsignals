"""Portfolio allocation models.

An allocation model is a pure function from the *effective signal state* to a target
portfolio exposure in [0, 1]. It knows nothing about prices, cash, positions, costs, or
dates; turning a target into trades is the job of ``api.rebalancer``. Keeping the two
apart means a new sizing idea (volatility targeting, regime-adjusted sizing,
correlation-aware weighting) is one new class here and nothing else changes.

Models that need more than signal counts (volatility, regime, correlation) read the
optional ``context`` mapping on ``SignalState``. The engine fills it; today it is empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol

AllocationModelId = str

ALL_IN = "all_in"
PROPORTIONAL = "proportional"

PROPORTIONAL_LINEAR = "linear"
PROPORTIONAL_EARLY_CONVICTION = "early_conviction"
PROPORTIONAL_CONFIRMATION_REQUIRED = "confirmation_required"

DEFAULT_ALLOCATION_MODEL = ALL_IN
DEFAULT_PROPORTIONAL_MODE = PROPORTIONAL_LINEAR

# exposure = (active / total) ** power
PROPORTIONAL_POWERS: dict[str, float] = {
    PROPORTIONAL_LINEAR: 1.0,
    PROPORTIONAL_EARLY_CONVICTION: 0.5,
    PROPORTIONAL_CONFIRMATION_REQUIRED: 2.0,
}

PROPORTIONAL_LABELS: dict[str, str] = {
    PROPORTIONAL_LINEAR: "Linear",
    PROPORTIONAL_EARLY_CONVICTION: "Early conviction",
    PROPORTIONAL_CONFIRMATION_REQUIRED: "Confirmation required",
}


@dataclass(frozen=True)
class SignalState:
    """What the allocator is allowed to see on one bar.

    ``eligible_count`` is the denominator: strategies whose signals can be evaluated on
    this bar, whether or not they have ever entered. ``active_count`` is how many of
    those hold (or just entered) a position. Always ``0 <= active_count <= eligible_count``.
    """

    active_count: int
    eligible_count: int
    context: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.eligible_count < 0 or self.active_count < 0:
            raise ValueError("Signal counts cannot be negative")
        if self.active_count > self.eligible_count:
            raise ValueError("active_count cannot exceed eligible_count")

    @property
    def active_fraction(self) -> float:
        if self.eligible_count == 0:
            return 0.0
        return self.active_count / self.eligible_count


def clamp_exposure(value: float) -> float:
    """Bound to [0, 1]; non-finite input is treated as no exposure."""
    if value is None or not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


class AllocationModel(Protocol):
    """Pure mapping from signal state to target exposure."""

    id: str
    label: str

    def target_exposure(self, state: SignalState) -> float: ...


@dataclass(frozen=True)
class AllInModel:
    """100% when any strategy is active, otherwise 0%."""

    id: str = ALL_IN
    label: str = "All-in"

    def target_exposure(self, state: SignalState) -> float:
        return 1.0 if state.active_count > 0 else 0.0


@dataclass(frozen=True)
class ProportionalModel:
    """exposure = (active / total) ** power, 0% when nothing is active, 100% when all are."""

    mode: str
    power: float

    @property
    def id(self) -> str:
        return f"{PROPORTIONAL}_{self.mode}"

    @property
    def label(self) -> str:
        return f"Proportional ({PROPORTIONAL_LABELS.get(self.mode, self.mode)})"

    def target_exposure(self, state: SignalState) -> float:
        if state.active_count <= 0 or state.eligible_count <= 0:
            return 0.0
        return clamp_exposure(state.active_fraction**self.power)


def proportional_model(mode: str) -> ProportionalModel:
    if mode not in PROPORTIONAL_POWERS:
        raise ValueError(f"Unsupported proportional mode: {mode}")
    return ProportionalModel(mode=mode, power=PROPORTIONAL_POWERS[mode])


def resolve_allocation_model(
    allocation_model: str | None = None,
    proportional_mode: str | None = None,
) -> AllocationModel:
    """Map the request/storage fields onto a model. Unset means the legacy default."""
    name = allocation_model or DEFAULT_ALLOCATION_MODEL
    if name == ALL_IN:
        return AllInModel()
    if name == PROPORTIONAL:
        return proportional_model(proportional_mode or DEFAULT_PROPORTIONAL_MODE)
    raise ValueError(f"Unsupported allocation model: {name}")


def comparison_models() -> list[AllocationModel]:
    """Every built-in model, in the order the comparison view lists them."""
    return [AllInModel(), *(proportional_model(mode) for mode in PROPORTIONAL_POWERS)]
