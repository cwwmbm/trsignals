import unittest

import pandas as pd

from api.strategy_compiler import compile_buy_mask, compile_sell_mask, format_condition_preview


class StrategyCompilerTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "Close": [100.0, 99.0, 101.0, 102.0],
                "SMA200": [101.0, 100.5, 100.0, 99.5],
                "RSI2": [18.0, 22.0, 15.0, 30.0],
            }
        )

    def test_and_conditions(self):
        mask = compile_buy_mask(
            self.data,
            [
                {"left": "Close", "operator": "<", "right": "SMA200", "logic": "AND"},
                {"left": "RSI2", "operator": "<=", "right": "20", "logic": "AND"},
            ],
        )
        self.assertEqual(mask.tolist(), [True, False, False, False])

    def test_or_condition(self):
        mask = compile_buy_mask(
            self.data,
            [
                {"left": "Close", "operator": "<", "right": "SMA200", "logic": "AND"},
                {"left": "RSI2", "operator": "<=", "right": "20", "logic": "OR"},
            ],
        )
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_flag_is_true(self):
        data = pd.DataFrame({"LowerCloses3": [1, -1, 1, -1]})
        mask = compile_buy_mask(
            data,
            [{"left": "LowerCloses3", "operator": "is true", "right": "", "logic": "AND"}],
        )
        self.assertEqual(mask.tolist(), [True, False, True, False])

    def test_flag_is_false(self):
        data = pd.DataFrame({"LowerCloses3": [1, -1, 1, -1]})
        mask = compile_buy_mask(
            data,
            [{"left": "LowerCloses3", "operator": "is false", "right": "", "logic": "AND"}],
        )
        self.assertEqual(mask.tolist(), [False, True, False, True])

    def test_format_flag_condition_preview(self):
        preview = format_condition_preview(
            [{"left": "LowerCloses3", "operator": "is true", "right": "", "logic": "AND"}],
            {"LowerCloses3": "Lower closes (3)"},
        )
        self.assertEqual(preview, "Lower closes (3) is true")

        preview = format_condition_preview(
            [
                {"left": "Close", "operator": "<", "right": "SMA200", "logic": "AND"},
                {"left": "RSI2", "operator": "<=", "right": "20", "logic": "AND"},
            ],
            {"Close": "Close", "SMA200": "SMA(200)", "RSI2": "RSI(2)"},
        )
        self.assertEqual(preview, "Close < SMA(200) AND RSI(2) <= 20")


    def test_compile_sell_mask(self):
        data = pd.DataFrame({"Stoch": [10.0, 25.0, 15.0, 30.0]})
        mask = compile_sell_mask(
            data,
            [
                {"left": "Stoch", "operator": ">", "right": "20", "logic": "AND"},
            ],
        )
        self.assertEqual(mask.tolist(), [False, True, False, True])


if __name__ == "__main__":
    unittest.main()
