"""Child-portfolio variants of a flat portfolio, for the allocation comparison.

A portfolio of four strategies can be run flat, or as two child portfolios of two strategies each,
every portfolio with its own allocation model. This lists those variants. It builds no signals and
runs nothing; ``api.portfolio_service`` does that.

What is generated, and why it is bounded
- **Structures:** the balanced splits into two groups (sizes ``ceil(n/2)`` and ``floor(n/2)``): for 4
  strategies the three ways to pair them, for 3 a pair and a single. A group of two or more becomes
  a child portfolio; a group of one stays a direct member of the parent. The strategies are named
  A, B, C, ... in the portfolio's own order, so a variant reads ``A+B | C+D``.
- **Models:** each variant pairs a parent model with a child model (both children share one). Every
  non-regime model is offered, 5 x 5 = 25 pairs per structure. Regime-adjusted sizing is left out:
  it needs its own regime history for every child.
- **Budget:** at most ``MAX_RUNS`` variants, since each is a full engine run (about 0.4 s). Over
  budget, the model list shrinks first (to four, then to three) and only then are structures dropped,
  spread evenly across the list rather than taking the first few.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from itertools import combinations, product

MAX_RUNS = 80

FULL_MODELS = ("all_in", "proportional_linear", "proportional_early_conviction", "tiered", "proportional_confirmation_required")
# What is offered, from most to fewest models, as the budget requires.
MODEL_TIERS = (
    FULL_MODELS,
    ("all_in", "proportional_linear", "proportional_early_conviction", "tiered"),
    ("all_in", "proportional_linear", "tiered"),
)

SHORT_LABELS = {
    "all_in": "All-in",
    "proportional_linear": "Linear",
    "proportional_early_conviction": "Early",
    "proportional_confirmation_required": "Confirm",
}


def short_model_label(model) -> str:
    """A model as it reads inside a variant's label."""
    if model.id in SHORT_LABELS:
        return SHORT_LABELS[model.id]
    single = getattr(model, "single", None)
    if model.id == "tiered" and single is not None:
        return f"Tiered {single * 100:g}"
    return model.label


def name_of(index: int) -> str:
    """Strategy ``index`` as a letter: A, B, C, ..."""
    return string.ascii_uppercase[index]


@dataclass(frozen=True)
class Structure:
    """One way to group the strategies: ``blocks`` hold indices into the portfolio's strategy list."""

    blocks: tuple[tuple[int, ...], ...]

    @property
    def key(self) -> str:
        return "-".join("".join(name_of(i) for i in block) for block in self.blocks)

    def label(self) -> str:
        return " | ".join("+".join(name_of(i) for i in block) for block in self.blocks)


@dataclass(frozen=True)
class NestingPlan:
    structures: tuple[Structure, ...]
    model_ids: tuple[str, ...]
    total_structures: int

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """(parent model id, child model id) for every variant of one structure."""
        return list(product(self.model_ids, self.model_ids))

    @property
    def runs(self) -> int:
        return len(self.structures) * len(self.model_ids) ** 2

    @property
    def truncated(self) -> bool:
        return len(self.structures) < self.total_structures or len(self.model_ids) < len(MODEL_TIERS[0])


def balanced_splits(n: int) -> list[Structure]:
    """Every distinct split of ``n`` strategies into two groups, as even as possible."""
    if n < 3:
        return []
    big = (n + 1) // 2
    out = []
    for first in combinations(range(n), big):
        rest = tuple(i for i in range(n) if i not in first)
        if len(first) == len(rest) and 0 not in first:
            continue  # the mirror image of a split already listed
        out.append(Structure((first, rest)))
    return out


def plan_nesting(n: int, max_runs: int = MAX_RUNS) -> NestingPlan | None:
    """The variants to run for a flat portfolio of ``n`` strategies, or None when there are none."""
    structures = balanced_splits(n)
    if not structures:
        return None
    for models in MODEL_TIERS:
        if len(structures) * len(models) ** 2 <= max_runs:
            return NestingPlan(tuple(structures), tuple(models), len(structures))
    models = MODEL_TIERS[-1]
    keep = max(1, max_runs // len(models) ** 2)
    if keep >= len(structures):
        picks = list(range(len(structures)))
    elif keep == 1:
        picks = [0]
    else:
        # spread across the list, so the strategies are not always grouped the same way
        picks = sorted({round(i * (len(structures) - 1) / (keep - 1)) for i in range(keep)})
    return NestingPlan(tuple(structures[i] for i in picks), tuple(models), len(structures))
