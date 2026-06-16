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
        self.assertGreaterEqual(len(indicators), 90)
        self.assertTrue(all(item.get("builderEligible", True) for item in indicators))
        self.assertTrue(all("id" in item and "label" in item and "category" in item for item in indicators))


if __name__ == "__main__":
    unittest.main()
