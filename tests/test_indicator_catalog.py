import unittest

from api.indicator_catalog import builder_eligible_ids, list_indicators
from backtest_runners import load_ticker_data


class IndicatorCatalogTests(unittest.TestCase):
    def test_builder_eligible_ids_exist_on_sample_dataframe(self):
        data = load_ticker_data("SOXX", years=2)
        columns = set(data.columns)
        missing = [indicator_id for indicator_id in builder_eligible_ids() if indicator_id not in columns]
        self.assertEqual(missing, [], f"Catalog ids missing from dataframe: {missing}")

    def test_list_indicators_returns_builder_entries_by_default(self):
        indicators = list_indicators(builder_only=True)
        self.assertGreaterEqual(len(indicators), 75)
        self.assertTrue(all(item.get("builderEligible", True) for item in indicators))
        self.assertTrue(all("id" in item and "label" in item and "category" in item for item in indicators))

    def test_builder_indicators_have_descriptions(self):
        missing = [
            item["id"]
            for item in list_indicators(builder_only=True)
            if not item.get("description", "").strip()
        ]
        self.assertEqual(missing, [])

    def test_builder_indicators_have_compare_mode(self):
        for item in list_indicators(builder_only=True):
            self.assertIn("compareMode", item)


if __name__ == "__main__":
    unittest.main()
