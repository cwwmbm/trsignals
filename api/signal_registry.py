from collections.abc import Callable

import indicators as ind


SIGNAL_REGISTRY: dict[str, Callable] = {
    name: getattr(ind, name)
    for name in dir(ind)
    if name.startswith("buy_signal") and callable(getattr(ind, name))
}

SIGNAL_REGISTRY.update(
    {
        "og_buy_signal": ind.og_buy_signal,
        "og_new_buy_signal": ind.og_new_buy_signal,
    }
)


def get_signal(name: str) -> Callable:
    try:
        return SIGNAL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown signal: {name}") from exc


def resolve_signal(expression: dict) -> Callable:
    kind = expression.get("kind", "single")
    if kind == "single":
        return get_signal(expression["name"])

    if kind == "combined":
        primary = get_signal(expression["primary"])
        secondary = get_signal(expression["secondary"])
        mode = expression.get("mode", "and")
        return ind.combined_signal(primary, secondary, mode)

    raise ValueError(f"Unknown signal expression kind: {kind}")


def list_signals() -> list[dict]:
    return [
        {"name": name, "label": name.replace("_", " ")}
        for name in sorted(SIGNAL_REGISTRY)
    ]
