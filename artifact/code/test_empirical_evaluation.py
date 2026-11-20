"""Guard against leakage and incorrect uncertainty in the market evaluation."""
import unittest
import numpy as np

from empirical_evaluation import extract_prefix, split_date, paired_date_bootstrap, fit_logistic, logistic_predict, audit_label_availability


def candle(t, p, v=10):
    return {"end_period_ts": t, "price": {"close_dollars": str(p)}, "volume_fp": str(v)}


class InformationSetTests(unittest.TestCase):
    def test_future_candles_and_settlement_fields_cannot_change_features(self):
        market = {"open_time": "2026-08-03T00:45:00Z", "close_time": "2026-08-03T01:00:00Z"}
        close = 1785718800
        rows = [candle(close - 900 + 60 * i, .2 + .02 * i) for i in range(1, 16)]
        a = extract_prefix(market, rows, 5, .001)
        changed = [r if r["end_period_ts"] <= close - 360 else candle(r["end_period_ts"], .999, 1e12) for r in rows]
        market.update(result="yes", last_price_dollars="1", volume_fp="100000000")
        b = extract_prefix(market, changed, 5, .001)
        np.testing.assert_array_equal(a["prices"], b["prices"])
        np.testing.assert_array_equal(a["volume"], b["volume"])
        self.assertEqual(len(a["prices"]), 9)
        self.assertAlmostEqual(a["prices"][-1], .38)

    def test_missing_duplicate_or_untraded_prefix_is_rejected(self):
        m = {"open_time": "2026-08-03T00:45:00Z", "close_time": "2026-08-03T01:00:00Z"}
        rows = [candle(1785717900 + 60 * i, .4) for i in range(1, 16)]
        for broken in [rows[1:], rows + [rows[0]], [candle(rows[0]["end_period_ts"], .4, 0)] + rows[1:]]:
            with self.assertRaises(ValueError):
                extract_prefix(m, broken, 5, .001)

    def test_chronological_boundaries_keep_entire_dates_together(self):
        self.assertEqual(split_date("2026-08-30"), "train")
        self.assertEqual(split_date("2026-08-31"), "validation")
        self.assertEqual(split_date("2026-09-13"), "validation")
        self.assertEqual(split_date("2026-09-14"), "test")
        self.assertEqual(split_date("2026-09-27"), "test")
        with self.assertRaises(ValueError):
            split_date("2026-09-28")

    def test_bootstrap_preserves_paired_date_clusters(self):
        # Two observations per date with opposite huge within-day values;
        # every whole-date mean is 1. Resampling individual rows would vary.
        diff = np.array([101., -99., 501., -499., 31., -29.])
        mean, lo, hi = paired_date_bootstrap(diff, np.array(["a", "a", "b", "b", "c", "c"]), 1000)
        self.assertEqual((mean, lo, hi), (1., 1., 1.))

    def test_logistic_scaler_is_frozen_at_training_values(self):
        x = np.array([[-2.], [-1.], [1.], [2.]])
        fit = fit_logistic(x, np.array([0, 0, 1, 1]), 1)
        before = logistic_predict(fit, np.array([[0.]]))[0]
        batch = logistic_predict(fit, np.array([[0.], [1e9]]))[0]
        self.assertAlmostEqual(before, .5, places=7)
        self.assertEqual(before, batch)

    def test_late_settlement_cannot_enter_the_next_fitting_period(self):
        records = [dict(ticker="a", split="train", cutoff=0), dict(ticker="b", split="validation", cutoff=140), dict(ticker="c", split="test", cutoff=340)]
        markets = {"a": {"settlement_ts": "1970-01-01T00:03:19Z"}, "b": {"settlement_ts": "1970-01-01T00:06:39Z"}}
        result = audit_label_availability(records, markets)
        self.assertTrue(result[0]["all_labels_available"])
        markets["a"]["settlement_ts"] = "1970-01-01T00:03:21Z"
        with self.assertRaises(ValueError):
            audit_label_availability(records, markets)


if __name__ == "__main__":
    unittest.main()
