#!/usr/bin/env python3
"""Data-free verification of manuscript/mixture_endpoint.tex and its appendix.

Run: /opt/anaconda3/bin/python3 -B code/check_mixture_endpoint.py
This script does not read empirical datasets, fitted artifacts or scores. It
never estimates a parameter: scale=1, volume median=1 and uniform state masses
specify one member of the frozen empirical model family.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import expit,logsumexp
from scipy.stats import norm,t
from empirical_mixture import CONFIG,EmpiricalMixture

STATES=np.array([(w,a) for w in [.2,.5,.8] for a in [0.,.1,.3,.6]])
WEIGHTS=np.full(len(STATES),1/len(STATES))


def specified_model():
    """Manually specify a probability law; no training or artifact restoration."""
    model=EmpiricalMixture()
    model.scale_=1.;model.volume_median_=1.;model.use_volume_=True
    model.catalogue_=STATES.copy();model.weights_=WEIGHTS.copy()
    model.label_prior_=.5
    # This flag means that the numerical likelihood parameters are available;
    # no empirical fitting occurs, and no fitting artifact is serialized.
    model.fitted_=True
    return model


def analytic_log_bf_gradient(z,volume,states=STATES,weights=WEIGHTS):
    """Independent SciPy densities and two-level responsibility derivatives."""
    z=np.asarray(z,float);volume=np.asarray(volume,float)
    if z.ndim!=2 or volume.shape!=z.shape:raise ValueError('Expected matching 2D arrays')
    w=states[:,0];a=states[:,1]
    gates=np.stack(np.broadcast_arrays(w*(1+volume[...,None])**.5,
        (1-w)/2*(1+volume[...,None])**.3,
        (1-w)/2+np.zeros_like(volume[...,None])),axis=-1)
    gates/=gates.sum(axis=-1,keepdims=True)
    sd_plus=1/np.sqrt(1+.25*volume[...,None]);student_scale=np.sqrt(3/5)
    means=a*(2*volume/(1+volume))[...,None]
    evidence=[];gradients=[]
    for y in [0,1]:
        mean=(2*y-1)*means
        components=np.stack(np.broadcast_arrays(
            norm.logpdf(z[...,None],loc=mean,scale=sd_plus),
            norm.logpdf(z[...,None],loc=-.5*mean,scale=1.),
            t.logpdf(z[...,None],df=5,loc=0,scale=student_scale)),axis=-1)
        weighted=components+np.log(gates)
        log_steps=logsumexp(weighted,axis=-1)
        component_responsibilities=np.exp(weighted-log_steps[...,None])
        component_scores=np.stack(np.broadcast_arrays(
            -(z[...,None]-mean)/sd_plus**2,
            -(z[...,None]+.5*mean),
            -6*z[...,None]/(3+z[...,None]**2)),axis=-1)
        state_step_scores=(component_responsibilities*component_scores).sum(axis=-1)
        joint=np.log(weights)+log_steps.sum(axis=1)
        log_evidence=logsumexp(joint,axis=-1)
        shared_state_responsibilities=np.exp(joint-log_evidence[:,None])
        score=(state_step_scores*shared_state_responsibilities[:,None,:]).sum(axis=-1)
        evidence.append(log_evidence);gradients.append(score)
    return evidence[1]-evidence[0],gradients[1]-gradients[0]


def finite_difference_gradient(z,volume,h=1e-5):
    result=np.empty_like(z,dtype=float)
    model=specified_model()
    for j in range(z.shape[1]):
        plus=z.copy();minus=z.copy();plus[:,j]+=h;minus[:,j]-=h
        result[:,j]=(model.log_bayes_factor(plus,volume)-model.log_bayes_factor(minus,volume))/(2*h)
    return result


def output_numbers():
    witness=np.array([[.5,.5],[3.,-2.]])
    bf,gradient=analytic_log_bf_gradient(witness,np.ones_like(witness))
    tail=[]
    for horizon in [2,3,5]:
        for radius in [3.,10.,30.,1000.]:
            path=np.full(horizon,radius);path[-1]=1-(horizon-1)*radius
            tail.append({'horizon':horizon,'radius':radius,'sum':float(path.sum()),'log_bf':float(analytic_log_bf_gradient(path[None,:],np.ones((1,horizon)))[0][0])})
    fd=finite_difference_gradient(witness,np.ones_like(witness))
    return {'parameters':{'scale':1.,'volume_median':1.,'state_weights':'uniform over frozen 12-state catalogue','volumes':[1.,1.],'initial_price':.5,'outcome_prior':.5},
        'same_endpoint_paths':witness.tolist(),'endpoint_price':float(expit(1.)),
        'log_bayes_factors':bf.tolist(),'posterior_probabilities':expit(bf).tolist(),
        'gradient':gradient.tolist(),'within_path_coordinate_gradient_difference':(gradient[:,0]-gradient[:,1]).tolist(),
        'maximum_analytic_finite_difference_error':float(np.max(np.abs(gradient-fd))),
        'balanced_tail_paths':tail,
        'empirical_model_source_sha256':hashlib.sha256(Path(__file__).with_name('empirical_mixture.py').read_bytes()).hexdigest(),
        'checker_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class EndpointChecks(unittest.TestCase):
    def test_same_endpoint_different_bayes_factors(self):
        x=np.array([[.5,.5],[3.,-2.]])
        bf,gradient=analytic_log_bf_gradient(x,np.ones_like(x))
        np.testing.assert_array_equal(x.sum(axis=1),np.ones(2))
        self.assertGreater(bf[0],.2);self.assertLess(bf[1],-.05)
        self.assertGreater(abs(gradient[1,0]-gradient[1,1]),.01)

    def test_independent_calculation_matches_frozen_model(self):
        self.assertEqual(CONFIG['weight_grid'],[.2,.5,.8])
        self.assertEqual(CONFIG['amplitude_grid'],[0.,.1,.3,.6])
        rng=np.random.default_rng(37);x=rng.normal(size=(21,4));v=rng.gamma(2,1,size=x.shape)
        with patch.object(EmpiricalMixture,'fit',side_effect=AssertionError('Fitting forbidden')):
            independent,_=analytic_log_bf_gradient(x,v)
            np.testing.assert_allclose(independent,specified_model().log_bayes_factor(x,v),atol=2e-14,rtol=1e-12)

    def test_responsibility_gradient_matches_central_differences(self):
        x=np.array([[.5,.5],[3.,-2.],[.2,-1.]])
        v=np.array([[1.,1.],[1.,1.],[.5,3.]])
        _,gradient=analytic_log_bf_gradient(x,v)
        np.testing.assert_allclose(gradient,finite_difference_gradient(x,v),atol=2e-8,rtol=2e-7)

    def test_balanced_tail_convergence_with_shared_state(self):
        for horizon in [2,3,5]:
            for total in [-2.,0.,1.,4.]:
                path=np.full(horizon,1000.);path[-1]=total-(horizon-1)*1000.
                actual,_=analytic_log_bf_gradient(path[None,:],np.ones((1,horizon)))
                self.assertLess(abs(actual[0]),1e-12)

    def test_no_signal_and_outcome_reflection(self):
        x=np.array([[.5,.5],[3.,-2.]])
        bf,_=analytic_log_bf_gradient(x,np.ones_like(x))
        reverse,_=analytic_log_bf_gradient(-x,np.ones_like(x))
        np.testing.assert_allclose(reverse,-bf,atol=1e-14)
        states=STATES[STATES[:,1]==0]
        no_signal,gradient=analytic_log_bf_gradient(x,np.ones_like(x),states,np.full(len(states),1/len(states)))
        np.testing.assert_array_equal(no_signal,np.zeros(len(x)))
        np.testing.assert_array_equal(gradient,np.zeros_like(x))

    def test_independent_product_special_case(self):
        x=np.array([[.5,.5],[3.,-2.]])
        states=STATES[-1:];weights=np.ones(1)
        joint,_=analytic_log_bf_gradient(x,np.ones_like(x),states,weights)
        separate=sum(analytic_log_bf_gradient(x[:,j:j+1],np.ones_like(x[:,j:j+1]),states,weights)[0] for j in range(2))
        np.testing.assert_allclose(joint,separate,atol=1e-14)
        self.assertGreater(abs(joint[0]-joint[1]),.05)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(EndpointChecks)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    print(json.dumps(output_numbers(),indent=2,allow_nan=False))
