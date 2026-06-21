import unittest

from api.indicator_catalog import (
    enrich_indicator,
    get_indicator_def,
    is_compare_indicator_allowed,
    list_indicators,
    resolve_compare_defaults,
    resolve_compare_indicator_kinds,
    resolve_compare_mode,
    validate_compare_target,
)


class CompareTargetTests(unittest.TestCase):
    def test_flag_indicators_use_none_mode(self):
        item = get_indicator_def("EMA8CrossUp")
        assert item is not None
        self.assertEqual(item["compareMode"], "none")

    def test_oscillators_use_number_mode(self):
        for indicator_id in ("RSI2", "CCI", "ValueCharts", "Stoch"):
            item = get_indicator_def(indicator_id)
            assert item is not None
            self.assertEqual(item["compareMode"], "number")
            self.assertIn("defaultCompareNumber", item)

    def test_price_and_trend_use_both_mode(self):
        for indicator_id in ("Close", "SMA200", "EMA8", "VWAP"):
            item = get_indicator_def(indicator_id)
            assert item is not None
            self.assertEqual(item["compareMode"], "both")
            self.assertIn("compareIndicatorKinds", item)

    def test_lag_prices_use_both_mode(self):
        item = get_indicator_def("CloseLag1")
        assert item is not None
        self.assertEqual(item["compareMode"], "both")

    def test_percent_indicators_use_number_mode(self):
        for indicator_id in ("Close_EMA8", "BBPercentB"):
            item = get_indicator_def(indicator_id)
            assert item is not None
            self.assertEqual(item["compareMode"], "number")

    def test_close_defaults_to_sma200(self):
        item = get_indicator_def("Close")
        assert item is not None
        self.assertEqual(item.get("defaultCompareIndicator"), "SMA200")

    def test_rsi2_defaults_to_20(self):
        item = get_indicator_def("RSI2")
        assert item is not None
        self.assertEqual(item.get("defaultCompareNumber"), "20")

    def test_trend_defaults_to_close_on_right(self):
        default_indicator, _ = resolve_compare_defaults(get_indicator_def("SMA200") or {})
        self.assertEqual(default_indicator, "Close")

    def test_compare_indicator_kind_filter(self):
        kinds = resolve_compare_indicator_kinds(get_indicator_def("Close") or {})
        self.assertEqual(kinds, ["price", "trend"])
        self.assertTrue(is_compare_indicator_allowed("Close", "SMA200"))
        self.assertFalse(is_compare_indicator_allowed("Close", "RSI2"))
        self.assertFalse(is_compare_indicator_allowed("RSI2", "Close"))

    def test_vwap_compare_kinds_include_volume_and_volatility(self):
        kinds = resolve_compare_indicator_kinds(get_indicator_def("VWAP") or {})
        self.assertIn("volume", kinds)
        self.assertIn("volatility", kinds)

    def test_validate_rejects_oscillator_vs_indicator(self):
        with self.assertRaisesRegex(ValueError, "numeric threshold"):
            validate_compare_target("RSI2", "Close")

    def test_validate_accepts_close_vs_sma(self):
        validate_compare_target("Close", "SMA200")

    def test_validate_accepts_rsi_vs_number(self):
        validate_compare_target("RSI2", "20")

    def test_validate_accepts_lag_price_comparisons(self):
        validate_compare_target("CloseLag1", "CloseLag3")
        validate_compare_target("LowLag1", "LowMin2Lag1")
        validate_compare_target("High", "CloseLag1")

    def test_is_compare_allowed_for_lag_prices(self):
        self.assertTrue(is_compare_indicator_allowed("CloseLag1", "CloseLag3"))
        self.assertTrue(is_compare_indicator_allowed("LowLag1", "LowMin2Lag1"))
        self.assertTrue(is_compare_indicator_allowed("High", "CloseLag1"))
        self.assertFalse(is_compare_indicator_allowed("RSI2", "CloseLag1"))

    def test_all_builder_indicators_resolve_compare_mode(self):
        for item in list_indicators(builder_only=True):
            enriched = enrich_indicator(item)
            compare_mode = resolve_compare_mode(enriched)
            self.assertIn(compare_mode, {"none", "number", "indicator", "both"})
            if compare_mode in {"indicator", "both"}:
                self.assertTrue(enriched.get("compareIndicatorKinds"))


if __name__ == "__main__":
    unittest.main()
