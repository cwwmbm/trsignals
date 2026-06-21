import tempfile
import unittest
from pathlib import Path

from api.legacy_strategy_migration import (
    build_migrated_strategies_payload,
    is_custom_strategy,
    merge_migrated_with_store,
    sync_migrated_strategies,
)


class LegacyStrategyMergeTests(unittest.TestCase):
    def test_is_custom_strategy(self):
        self.assertTrue(is_custom_strategy({"name": "My strat"}))
        self.assertTrue(is_custom_strategy({"name": "My strat", "legacy_signal": None}))
        self.assertFalse(is_custom_strategy({"name": "buy_signal7", "legacy_signal": "buy_signal7"}))

    def test_merge_preserves_custom_strategies(self):
        custom = [
            {
                "id": "custom-1",
                "name": "My RSI dip",
                "symbol": "SPY",
                "direction": "long",
                "hold_days": 2,
                "profit": 1,
                "description": "",
                "conditions": [{"left": "RSI2", "operator": "<=", "right": "20", "logic": "AND"}],
                "sell_conditions": [],
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ]
        merged = merge_migrated_with_store(custom, include_non_scan=False)
        self.assertEqual(merged[:1], custom)
        self.assertTrue(all(item.get("legacy_signal") for item in merged[1:]))

    def test_sync_replaces_migrated_but_keeps_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.json"
            custom = {
                "id": "custom-1",
                "name": "Custom",
                "symbol": "SPY",
                "direction": "long",
                "hold_days": 2,
                "profit": 1,
                "description": "",
                "conditions": [{"left": "RSI2", "operator": "<=", "right": "20", "logic": "AND"}],
                "sell_conditions": [],
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
            stale_migrated = build_migrated_strategies_payload(include_non_scan=False)[0]
            stale_migrated["description"] = "stale"
            path.write_text(__import__("json").dumps([custom, stale_migrated], indent=2), encoding="utf-8")

            custom_count, migrated_count = sync_migrated_strategies(path, include_non_scan=False)
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))

            self.assertEqual(custom_count, 1)
            self.assertGreater(migrated_count, 0)
            self.assertEqual(payload[0]["name"], "Custom")
            self.assertNotEqual(payload[1]["description"], "stale")


if __name__ == "__main__":
    unittest.main()
