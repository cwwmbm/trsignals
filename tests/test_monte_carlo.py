import unittest

import numpy as np

from api.monte_carlo import run_monte_carlo


class MonteCarloTests(unittest.TestCase):
    def test_requires_at_least_two_trades(self):
        with self.assertRaises(ValueError):
            run_monte_carlo([0.1])

    def test_actual_path_matches_hand_compounded(self):
        returns = [0.10, -0.05, 0.20]
        start = 15_000.0
        expected_final = start * 1.10 * 0.95 * 1.20
        equity = [start]
        peak = start
        max_dd = 0.0
        for r in returns:
            equity.append(equity[-1] * (1.0 + r))
            peak = max(peak, equity[-1])
            max_dd = max(max_dd, (peak - equity[-1]) / peak)

        result = run_monte_carlo(returns, method="shuffle", n_sims=100, start_capital=start)
        self.assertAlmostEqual(result["actual"]["final_equity"], expected_final, places=6)
        self.assertAlmostEqual(result["actual"]["max_drawdown"], max_dd, places=6)
        self.assertEqual(result["n_trades"], 3)
        self.assertGreater(len(result["drawdown_distribution"]), 0)
        self.assertEqual(result["confidence_marker"]["percentile"], 95)

    def test_drawdown_distribution_counts_sum_to_n_sims(self):
        result = run_monte_carlo(
            [0.05, -0.02, 0.03, 0.01, -0.04],
            method="bootstrap",
            n_sims=300,
            seed=3,
        )
        total = sum(row["count"] for row in result["drawdown_distribution"])
        self.assertEqual(total, result["n_sims"])
        last = result["drawdown_distribution"][-1]
        self.assertAlmostEqual(last["cumulative_pct"], 100.0, places=6)
        for prev, curr in zip(
            result["drawdown_distribution"],
            result["drawdown_distribution"][1:],
        ):
            self.assertLessEqual(prev["cumulative_pct"], curr["cumulative_pct"])
            self.assertLessEqual(prev["drawdown_pct"], curr["drawdown_pct"])

    def test_confidence_marker_near_p95(self):
        result = run_monte_carlo(
            [0.08, -0.06, 0.04, 0.02, -0.03, 0.05],
            method="shuffle",
            n_sims=500,
            seed=11,
        )
        marker = result["confidence_marker"]
        p95 = result["summary"]["max_drawdown"]["p95"] * 100.0
        self.assertAlmostEqual(marker["drawdown_pct"], p95, places=6)
        self.assertEqual(marker["cumulative_pct"], 95.0)

    def test_shuffle_preserves_mean_return_per_sim(self):
        returns = np.array([0.02, -0.01, 0.03, 0.04, -0.02], dtype=float)
        result = run_monte_carlo(returns.tolist(), method="shuffle", n_sims=200, seed=7)
        fe = result["summary"]["final_equity"]
        self.assertLessEqual(fe["p5"], fe["p25"])
        self.assertLessEqual(fe["p25"], fe["p50"])
        self.assertLessEqual(fe["p50"], fe["p75"])
        self.assertLessEqual(fe["p75"], fe["p95"])
        self.assertIn("pct_sims_final_equity_ge_actual", result["summary"])
        self.assertIn("pct_sims_max_drawdown_le_actual", result["summary"])

        again = run_monte_carlo(returns.tolist(), method="shuffle", n_sims=200, seed=7)
        self.assertEqual(result["summary"]["final_equity"]["p50"], again["summary"]["final_equity"]["p50"])

    def test_bootstrap_can_differ_from_shuffle(self):
        returns = [0.10, -0.08, 0.05, 0.12, -0.03, 0.07, -0.04, 0.02]
        shuffle = run_monte_carlo(returns, method="shuffle", n_sims=500, seed=1)
        bootstrap = run_monte_carlo(returns, method="bootstrap", n_sims=500, seed=1)
        self.assertEqual(shuffle["method"], "shuffle")
        self.assertEqual(bootstrap["method"], "bootstrap")
        self.assertNotEqual(
            shuffle["summary"]["final_equity"]["p5"],
            bootstrap["summary"]["final_equity"]["p5"],
        )

    def test_clamps_n_sims(self):
        result = run_monte_carlo([0.1, -0.05, 0.02], method="shuffle", n_sims=5)
        self.assertEqual(result["n_sims"], 100)
        result_hi = run_monte_carlo([0.1, -0.05, 0.02], method="shuffle", n_sims=50_000)
        self.assertEqual(result_hi["n_sims"], 10_000)


if __name__ == "__main__":
    unittest.main()
