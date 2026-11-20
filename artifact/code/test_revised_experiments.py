"""Scientific checks for the finite-prior experiment suite."""
import importlib.util
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit


class RevisedExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.find_spec('revised_experiments')
        if spec is not None:
            cls.m = __import__('revised_experiments')

    def test_implementation_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('revised_experiments'))

    def test_density_integrates(self):
        for label in (0, 1):
            for v in (.1, 1., 4.):
                integral = quad(lambda x: np.exp(self.m.log_density(x, v, label, .4, .5)), -np.inf, np.inf)[0]
                self.assertAlmostEqual(integral, 1., places=7)

    def test_exact_marginalization_enumeration(self):
        x = np.array([[.2, -.1, .7]])
        v = np.array([[.5, 1., 2.]])
        actual = self.m.infer(x, v, [3])['marginal'][0, 0]
        evidence = []
        for label in (0, 1):
            evidence.append(np.mean([np.prod(np.exp(self.m.log_density(x, v, label, w, mu))) for w, mu in self.m.STATES]))
        self.assertAlmostEqual(actual, np.log(evidence[1] / evidence[0]), places=12)

    def test_nonuniform_prior_enumeration(self):
        x = np.array([[.2, -.1, .7]]); v = np.array([[.5, 1., 2.]])
        prior = np.arange(1, 10, dtype=float); prior /= prior.sum()
        actual = self.m.infer(x, v, [3], prior_weights=prior)['marginal'][0, 0]
        e = [sum(prior[j]*np.prod(np.exp(self.m.log_density(x, v, y, w, m))) for j, (w, m) in enumerate(self.m.STATES)) for y in (0, 1)]
        self.assertAlmostEqual(actual, np.log(e[1]/e[0]), places=12)

    def test_symmetries_and_no_signal(self):
        x = np.array([[.2, -.1, .7], [-.2, .1, -.7]])
        v = np.ones_like(x)
        odds = self.m.infer(x, v, [1, 3])['marginal']
        np.testing.assert_allclose(odds[0], -odds[1], atol=1e-12)
        np.testing.assert_allclose(self.m.orientation_log_odds(x, v, [1, 3]), 0., atol=1e-12)
        np.testing.assert_allclose(self.m.infer(x, v, [1, 3], signal_scale=0.)['marginal'], 0., atol=1e-12)

    def test_stable_extreme_scores(self):
        brier, loss, info = self.m.scores(np.array([1, 0, 1, 0]), np.array([1000., -1000., -1000., 1000.]))
        np.testing.assert_allclose(brier, [0., 0., 1., 1.])
        np.testing.assert_allclose(loss, [0., 0., 1000., 1000.])
        np.testing.assert_allclose(info, np.log(2.))
        b, l, i = self.m.scores(np.array([0, 1]), np.zeros(2))
        np.testing.assert_allclose(b, .25)
        np.testing.assert_allclose(l, np.log(2.))
        np.testing.assert_allclose(i, 0.)

    def test_stability_bound(self):
        rng = np.random.default_rng(77)
        x = rng.normal(size=(20, 13)); v = rng.gamma(2, .5, size=x.shape)
        d = rng.normal(0, .04, size=x.shape)
        bound = self.m.stability_bound(x, x + d, v)
        a = self.m.infer(x, v, [13])['marginal'][:, 0]
        b = self.m.infer(x + d, v, [13])['marginal'][:, 0]
        self.assertTrue(np.all(np.abs(a - b) <= bound + 1e-12))
        self.assertTrue(np.all(bound > 0))

    def test_deterministic_simulation(self):
        a = self.m.simulate(24, 15, seed=8)
        b = self.m.simulate(24, 15, seed=8)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertEqual(set(a['labels']), {0, 1})

    def test_logistic_endpoint_calibration(self):
        rng=np.random.default_rng(19)
        x=rng.normal(size=(10000, 1)); y=rng.binomial(1, expit(-.4+1.3*x[:,0]))
        fitted=self.m.fit_endpoint_calibration(x,y)
        self.assertLess(abs(fitted[0]['intercept']+.4), .08)
        self.assertLess(abs(fitted[0]['slope']-1.3), .08)
        self.assertTrue(fitted[0]['success'])

    def test_paired_calibrated_comparison(self):
        stored={'demo_labels':np.array([0,1]),'demo_calibrated_endpoint_log_odds':np.zeros((2,6)), 'demo_marginal_log_odds':np.tile([-2.,2.],(6,1)).T}
        rows=[dict(scenario='demo',method='marginal',horizon=10)]
        self.m.add_calibrated_comparisons(rows,stored)
        expected=expit(-2.)**2-.25
        self.assertAlmostEqual(rows[0]['delta_brier_vs_calibrated'],expected)
        self.assertAlmostEqual(rows[0]['delta_brier_vs_calibrated_lo'],expected)

    def test_latent_ar_marginals(self):
        # Matched one-dimensional transforms: AR dependence must not rescale marginals.
        a = self.m.simulate(4000, 3, seed=9, rho=.6)
        self.assertLess(abs(a['latent_z'].mean()), .03)
        self.assertLess(abs(a['latent_z'].var() - 1.), .04)
        self.assertLess(abs(np.corrcoef(a['latent_z'][:, 0], a['latent_z'][:, 1])[0, 1] - .6), .04)

if __name__ == '__main__':
    unittest.main()
