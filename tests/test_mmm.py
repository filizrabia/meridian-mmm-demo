import unittest

import numpy as np

from mmm.data import make_dataset
from mmm.model import MarketingMixModel
from mmm.transforms import geometric_adstock, hill


class TransformTests(unittest.TestCase):
    def test_adstock_preserves_total_and_delays_effect(self):
        spend = np.zeros(20)
        spend[0] = 100.0
        out = geometric_adstock(spend, decay=0.5, max_lag=8)
        self.assertAlmostEqual(out.sum(), 100.0)
        self.assertTrue(np.all(np.diff(out[:9]) < 0))

    def test_zero_decay_is_identity(self):
        spend = np.random.default_rng(0).random(30)
        np.testing.assert_allclose(geometric_adstock(spend, 0.0), spend)

    def test_hill_half_saturation(self):
        self.assertAlmostEqual(hill(np.array([50.0]), 50.0, 2.0)[0], 0.5)
        self.assertTrue(np.all(np.diff(hill(np.linspace(0, 500, 50), 50.0, 1.5)) >= 0))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            geometric_adstock(np.ones(5), 1.0)
        with self.assertRaises(ValueError):
            hill(np.ones(5), 0.0, 1.0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = make_dataset()
        cls.model = MarketingMixModel(cls.ds.channels).fit(cls.ds.df)

    def test_fit_quality(self):
        self.assertGreater(self.model.r2(self.ds.df), 0.95)

    def test_contributions_sum_to_prediction(self):
        contrib = self.model.contributions(self.ds.df)
        np.testing.assert_allclose(contrib.sum(axis=1), self.model.predict(self.ds.df))

    def test_roi_close_to_truth_for_large_channels(self):
        true, est = self.ds.true_roi(), self.model.roi(self.ds.df)
        for ch in ["tv", "search", "social"]:
            self.assertLess(abs(est[ch] - true[ch]) / true[ch], 0.3, ch)

    def test_lift_test_calibration_moves_roi_towards_experiment(self):
        true = self.ds.true_roi()["display"]
        cal = MarketingMixModel(self.ds.channels).fit(self.ds.df, lift_tests={"display": (true, 0.1)})
        before = abs(self.model.roi(self.ds.df)["display"] - true)
        after = abs(cal.roi(self.ds.df)["display"] - true)
        self.assertLess(after, before)
        self.assertLess(after, 0.1)

    def test_holdout_error_is_close_to_training_error(self):
        df = self.ds.df
        mask = np.arange(len(df)) < len(df) - 26
        m = MarketingMixModel(self.ds.channels).fit(df, train_mask=mask)
        y, pred = df["revenue"].to_numpy(), m.predict(df)
        mape = lambda sel: np.mean(np.abs(y[sel] - pred[sel]) / y[sel])  # noqa: E731
        self.assertLess(mape(~mask), 2 * mape(mask))

    def test_budget_optimizer_respects_constraints(self):
        cur = {c: float(self.ds.df[c].mean()) for c in self.ds.channels}
        plan = self.model.optimize_budget(sum(cur.values()), cur, max_change=0.5)
        self.assertAlmostEqual(plan["optimized_spend"].sum(), sum(cur.values()), delta=len(cur))
        self.assertGreaterEqual(plan["optimized_revenue"].sum(), plan["current_revenue"].sum() - 1)


if __name__ == "__main__":
    unittest.main()
