"""Data-free scientific checks for the empirical finite catalogue."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit, logsumexp
import empirical_mixture as em


class EmpiricalMixtureTests(unittest.TestCase):
    def setUp(self):
        self.z=np.array([[.1,-.2,.3],[-.1,.2,-.3],[.4,.1,-.2],[-.4,-.1,.2]])
        self.v=np.array([[1.,2.,3.],[2.,3.,1.],[1.,1.,2.],[2.,1.,1.]])
        self.y=np.array([1,0,1,0])
        self.model=em.EmpiricalMixture().fit(self.z,self.v,self.y)

    def test_step_densities_integrate(self):
        for label in [0,1]:
            for volume in [0.,2.,100.]:
                for state in [0,7,11]:
                    integral=quad(lambda x:np.exp(self.model.step_log_densities(np.array([[x]]),np.array([[volume]]),label)[0,0,state]),-np.inf,np.inf,epsabs=1e-8)[0]
                    self.assertAlmostEqual(integral,1.,places=6)

    def test_shared_state_full_path_enumeration(self):
        # Use two state masses to compare whole-path marginalization with a hand sum.
        model=em.EmpiricalMixture.from_dict(self.model.to_dict())
        model.catalogue_=model.catalogue_[[3,7]]
        model.weights_=np.array([.3,.7])
        x=self.z[:1];v=self.v[:1]
        likelihoods=[]
        for label in [0,1]:
            steps=model.step_log_densities(x,v,label)[0]
            likelihoods.append(sum(model.weights_[j]*np.prod(np.exp(steps[:,j])) for j in range(2)))
        actual=model.log_bayes_factor(x,v)[0]
        self.assertAlmostEqual(actual,np.log(likelihoods[1]/likelihoods[0]),places=12)
        wrong=[]
        for label in [0,1]:
            steps=model.step_log_densities(x,v,label)[0]
            wrong.append(np.sum(logsumexp(steps+np.log(model.weights_),axis=1)))
        self.assertGreater(abs(actual-(wrong[1]-wrong[0])),1e-5)

    def test_outcome_reflection_symmetry(self):
        np.testing.assert_allclose(self.model.log_bayes_factor(-self.z,self.v),-self.model.log_bayes_factor(self.z,self.v),atol=1e-12)
        flipped=em.EmpiricalMixture().fit(-self.z,self.v,1-self.y)
        np.testing.assert_allclose(flipped.weights_,self.model.weights_,atol=1e-12)
        np.testing.assert_allclose(flipped.predict_proba(-self.z,self.v),1-self.model.predict_proba(self.z,self.v),atol=1e-12)

    def test_no_volume_invariance(self):
        model=em.EmpiricalMixture().fit(self.z,self.v,self.y,use_volume=False)
        np.testing.assert_array_equal(model.log_bayes_factor(self.z,self.v),model.log_bayes_factor(self.z,self.v*100000))
        alternative=em.EmpiricalMixture().fit(self.z,self.v*100000,self.y,use_volume=False)
        np.testing.assert_allclose(model.weights_,alternative.weights_,atol=1e-12)

    def test_training_scalers_unchanged_at_prediction(self):
        expected=max(1.4826*np.median(np.abs(self.z-np.median(self.z))),.05)
        self.assertAlmostEqual(self.model.scale_,expected)
        self.assertEqual(self.model.volume_median_,np.median(self.v[self.v>0]))
        before=self.model.to_dict()
        self.model.predict_proba(self.z*1000,self.v*1000)
        self.assertEqual(before,self.model.to_dict())

    def test_finite_extreme_inputs(self):
        x=np.array([[1e300,-1e300,0.],[1e6,-1e6,1.]])
        v=np.array([[1e300,0.,1e-300],[1e100,1e-100,1.]])
        self.assertTrue(np.isfinite(self.model.log_bayes_factor(x,v)).all())
        self.assertTrue(np.isfinite(self.model.predict_proba(x,v)).all())

    def test_em_smoothing_and_objective(self):
        self.assertTrue(np.all(self.model.weights_>0))
        self.assertAlmostEqual(self.model.weights_.sum(),1.)
        objectives=self.model.diagnostics_['penalized_objective']
        self.assertTrue(np.all(np.diff(objectives)>=-1e-10))
        self.assertTrue(self.model.diagnostics_['converged'])
        self.assertEqual(len(self.model.weights_),12)

    def test_serialization_snapshot_is_independent(self):
        payload=self.model.to_dict()
        payload['config']['weight_grid'][0]=.99
        self.assertEqual(self.model.to_dict()['config']['weight_grid'],[.2,.5,.8])

    def test_no_signal_catalogue_and_roundtrip(self):
        restored=em.EmpiricalMixture.from_dict(self.model.to_dict())
        np.testing.assert_array_equal(restored.log_bayes_factor(self.z,self.v),self.model.log_bayes_factor(self.z,self.v))
        restored.catalogue_=restored.catalogue_[restored.catalogue_[:,1]==0]
        restored.weights_=np.ones(len(restored.catalogue_))/len(restored.catalogue_)
        np.testing.assert_array_equal(restored.log_bayes_factor(self.z,self.v),np.zeros(4))
        np.testing.assert_allclose(restored.predict_proba(self.z,self.v),restored.label_prior_)

    def test_scale_floor_constant_outcome_and_input_rejection(self):
        fitted=em.EmpiricalMixture().fit(np.zeros((3,2)),np.ones((3,2)),np.ones(3))
        self.assertEqual(fitted.scale_,.05)
        self.assertAlmostEqual(fitted.label_prior_,3.5/4)
        for z,v,y in [(self.z,self.v,-self.y),(self.z,np.full_like(self.v,-1),self.y),(self.z,np.zeros_like(self.v),self.y),(self.z[:,:0],self.v[:,:0],self.y)]:
            with self.assertRaises(ValueError):em.EmpiricalMixture().fit(z,v,y)


class GaussianHistoryTests(unittest.TestCase):
    def setUp(self):
        self.z=np.array([[0.,0.],[1.,1.],[2.,1.],[3.,2.]])
        self.asset=np.array(['BTC']*4);self.y=np.array([0,0,1,1])

    def test_manual_multivariate_gaussian_likelihood_ratio(self):
        from scipy.stats import multivariate_normal
        model=em.GaussianHistory().fit(self.z,self.asset,self.y)
        mean0=self.z[:2].mean(axis=0);mean1=self.z[2:].mean(axis=0)
        residual=self.z-np.array([mean0,mean0,mean1,mean1])
        raw=residual.T@residual/4
        cov=.9*raw+.1*np.diag(np.diag(raw))
        x=np.array([[1.,0.],[.3,2.]])
        expected=multivariate_normal.logpdf(x,mean=mean1,cov=cov)-multivariate_normal.logpdf(x,mean=mean0,cov=cov)
        np.testing.assert_allclose(model.log_bayes_factor(x,['BTC']*2),expected,atol=1e-12)
        diagonal=(x-.5*(mean1+mean0))@np.linalg.solve(np.diag(np.diag(cov)),mean1-mean0)
        self.assertGreater(np.max(np.abs(expected-diagonal)),1.)

    def test_label_flip_and_asset_specific_fits(self):
        x=np.concatenate([self.z,2*self.z])
        asset=np.array(['BTC']*4+['ETH']*4);y=np.concatenate([self.y,1-self.y])
        model=em.GaussianHistory().fit(x,asset,y)
        flipped=em.GaussianHistory().fit(x,asset,1-y)
        np.testing.assert_allclose(flipped.log_bayes_factor(x,asset),-model.log_bayes_factor(x,asset),atol=1e-12)
        np.testing.assert_allclose(flipped.predict_proba(x,asset),1-model.predict_proba(x,asset),atol=1e-12)
        self.assertEqual(set(model.to_dict()['assets']),{'BTC','ETH'})

    def test_covariance_floor_and_serialized_roundtrip(self):
        x=np.zeros((4,2));model=em.GaussianHistory().fit(x,self.asset,self.y)
        payload=model.to_dict();state=payload['assets']['BTC']
        np.testing.assert_allclose(state['covariance'],np.eye(2)*1e-4)
        self.assertEqual(state['diagnostics']['diagonal_floor_count'],2)
        self.assertFalse(state['diagnostics']['eigenvalue_floor_applied'])
        restored=em.GaussianHistory.from_dict(payload)
        np.testing.assert_array_equal(model.predict_proba(x,self.asset),restored.predict_proba(x,self.asset))
        with self.assertRaises(ValueError):model.predict_proba(x,['unknown']*4)
        with self.assertRaises(ValueError):em.GaussianHistory().fit(x,self.asset,np.ones(4))


if __name__=='__main__':unittest.main()
