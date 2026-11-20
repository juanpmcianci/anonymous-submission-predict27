#!/usr/bin/env python3
"""Reproducible synthetic finite-prior Bayesian inverse experiments.

Run: python code/revised_experiments.py --histories 2000
No SMC, empirical forecast claim, or continuous-nuisance KL projection is made.
Volumes are observed exogenous Gamma(2,.5). Paths are unbounded logit increments;
the synthetic last-price baseline is expit(sum increments), without clipping.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp, ndtr, xlogy
from scipy.stats import norm, t

STATES = np.array([(w, m) for w in (.15, .4, .65) for m in (.25, .5, .75)])
HORIZONS = np.array([10, 25, 50, 100, 200, 500])
METHODS = ['marginal', 'oracle', 'last_price', 'prior', 'no_volume', 'gaussian_tail']
DISPLAY = dict(marginal='Finite-prior Bayes', oracle='Known-parameter likelihood', last_price='Synthetic last price', prior='Prior', no_volume='Volume omitted', gaussian_tail='Gaussian tail')
COLORS = dict(marginal='#0072B2', oracle='#009E73', last_price='#D55E00', prior='#777777', no_volume='#CC79A7', gaussian_tail='#E69F00')
DISPLAY['calibrated_endpoint']='Calibrated endpoint'
COLORS['calibrated_endpoint']='#332288'
ROOT = Path(__file__).resolve().parents[1]


def gate_weights(v, omega):
    v, omega = np.asarray(v), np.asarray(omega)
    raw = np.stack(np.broadcast_arrays(omega * (1 + v)**.5, (1-omega)*2/3 + np.zeros_like(v), (1-omega)/3 * (1+v)**.3), axis=-1)
    return raw / raw.sum(axis=-1, keepdims=True)


def log_density(x, v, label, omega, mu, gaussian_tail=False, signal_scale=1.):
    """Normalized mixture density, broadcast over observations and nuisance states."""
    x, v = np.asarray(x), np.asarray(v)
    sign = 2*np.asarray(label)-1
    mean_i = signal_scale*sign*mu*(-np.expm1(-.1*v))
    sd_i = .3/np.sqrt(1+.05*v)
    mean_o = -signal_scale*sign*.3*expit((v-1.5)/.3)
    li = -.5*((x-mean_i)/sd_i)**2 - np.log(sd_i) - .5*np.log(2*np.pi)
    ln = -.5*(x/.5)**2 - np.log(.5) - .5*np.log(2*np.pi)
    if gaussian_tail:
        sd_o = .4*np.sqrt(5/3)
        lo = -.5*((x-mean_o)/sd_o)**2 - np.log(sd_o) - .5*np.log(2*np.pi)
    else:
        z = (x-mean_o)/.4
        lo = gammaln(3)-gammaln(2.5)-.5*np.log(5*np.pi)-np.log(.4)-3*np.log1p(z*z/5)
    weights = gate_weights(v, omega)
    parts = np.stack(np.broadcast_arrays(li, ln, lo), axis=-1)
    with np.errstate(divide='ignore'):
        return logsumexp(parts + np.log(weights), axis=-1)


def simulate(n, horizon, seed, prior_predictive=False, omega=.4, mu=.5, rho=0., signal_scale=1.):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2
    rng.shuffle(labels)
    states = rng.integers(len(STATES), size=n) if prior_predictive else np.full(n, 4)
    params = STATES[states] if prior_predictive else np.tile([omega, mu], (n, 1))
    v = rng.gamma(2., .5, size=(n, horizon))
    weights = gate_weights(v, params[:, 0, None])
    component = (rng.random((n, horizon, 1)) > np.cumsum(weights, axis=-1)).sum(axis=-1)
    z = rng.standard_normal((n, horizon))
    if rho:
        for j in range(1, horizon):
            z[:, j] = rho*z[:, j-1] + np.sqrt(1-rho*rho)*z[:, j]
    sign = (2*labels-1)[:, None]
    x = .5*z
    mask = component == 0
    informed = signal_scale*sign*params[:, 1, None]*(-np.expm1(-.1*v)) + .3/np.sqrt(1+.05*v)*z
    x[mask] = informed[mask]
    mask = component == 2
    if rho:
        opposing_noise = t.ppf(np.clip(ndtr(z[mask]), np.finfo(float).eps, 1-np.finfo(float).eps), 5)
    else:
        opposing_noise = rng.standard_t(5, size=mask.sum())
    opposing_mean = -signal_scale*sign*.3*expit((v-1.5)/.3)
    x[mask] = opposing_mean[mask] + .4*opposing_noise
    return dict(x=x, v=v, labels=labels, params=params, state_indices=states, latent_z=z)


def evidence(x, v, horizons, gaussian_tail=False, signal_scale=1., prior_weights=None):
    """log p(history prefix | label), summed over ONE shared path-level state."""
    prior_weights = np.full(len(STATES), 1/len(STATES)) if prior_weights is None else np.asarray(prior_weights)
    if prior_weights.shape != (len(STATES),) or np.any(prior_weights <= 0) or not np.isclose(prior_weights.sum(), 1):
        raise ValueError("Prior must have nine positive weights summing to one")
    result = np.empty((len(x), len(horizons), 2))
    for label in (0, 1):
        logs = log_density(x[..., None], v[..., None], label, STATES[:, 0], STATES[:, 1], gaussian_tail, signal_scale)
        cumulative = np.cumsum(logs, axis=1)[:, np.asarray(horizons)-1, :]
        result[..., label] = logsumexp(cumulative+np.log(prior_weights), axis=-1)
    return result


def infer(x, v, horizons, params=None, signal_scale=1., all_methods=False, batch=100, prior_weights=None):
    methods = METHODS if all_methods else ['marginal']
    out = {key: np.empty((len(x), len(horizons))) for key in methods}
    for start in range(0, len(x), batch):
        sl = slice(start, start+batch)
        ev = evidence(x[sl], v[sl], horizons, signal_scale=signal_scale, prior_weights=prior_weights)
        out['marginal'][sl] = ev[..., 1]-ev[..., 0]
        if all_methods:
            p = params[sl] if params is not None else np.tile([.4, .5], (len(x[sl]), 1))
            lr = log_density(x[sl], v[sl], 1, p[:, 0, None], p[:, 1, None], signal_scale=signal_scale)-log_density(x[sl], v[sl], 0, p[:, 0, None], p[:, 1, None], signal_scale=signal_scale)
            out['oracle'][sl] = np.cumsum(lr, axis=1)[:, np.asarray(horizons)-1]
            out['last_price'][sl] = np.cumsum(x[sl], axis=1)[:, np.asarray(horizons)-1]
            out['prior'][sl] = 0.
            for name in ('no_volume', 'gaussian_tail'):
                vv = np.ones_like(v[sl]) if name == 'no_volume' else v[sl]
                ev = evidence(x[sl], vv, horizons, gaussian_tail=name == 'gaussian_tail', signal_scale=signal_scale, prior_weights=prior_weights)
                out[name][sl] = ev[..., 1]-ev[..., 0]
    return out


def orientation_log_odds(x, v, horizons):
    """Explicitly marginalize equal orientation +/-1; second reverses label laws."""
    ev = evidence(x, v, horizons)
    oriented = np.stack([ev, ev[..., ::-1]], axis=-1)
    mixed = logsumexp(oriented, axis=-1)-np.log(2)
    return mixed[..., 1]-mixed[..., 0]


def scores(labels, odds):
    """Brier, numerically stable log loss, and realized posterior/prior KL (nats)."""
    labels, odds = np.asarray(labels), np.asarray(odds)
    p = expit(odds)
    loss = np.logaddexp(0., -(2*labels-1)*odds)
    info = np.log(2)+xlogy(p, p)+xlogy(1-p, 1-p)
    return (p-labels)**2, loss, info


def fit_endpoint_calibration(endpoints, labels):
    """Fit intercept+slope by stable empirical logistic loss on TRAINING data only."""
    fitted=[]
    for j in range(endpoints.shape[1]):
        x=endpoints[:,j]; center=float(x.mean()); scale=float(x.std()) or 1.
        design=np.column_stack([np.ones(len(x)),(x-center)/scale])
        def objective(beta):
            eta=design@beta
            loss=np.mean(np.logaddexp(0.,eta)-labels*eta)
            gradient=design.T@(expit(eta)-labels)/len(labels)
            return loss,gradient
        result=minimize(objective,np.zeros(2),jac=True,method='L-BFGS-B',options={'ftol':1e-13,'gtol':1e-8,'maxiter':1000})
        if not result.success:raise RuntimeError('Endpoint calibration failed: '+str(result.message))
        slope=float(result.x[1]/scale); intercept=float(result.x[0]-slope*center)
        fitted.append(dict(intercept=intercept,slope=slope,training_log_loss=float(result.fun),success=bool(result.success),n_training=len(labels)))
    return fitted


def mean_interval(values):
    values = np.asarray(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    return mean, mean-1.96*se, mean+1.96*se


def stability_bound(x, perturbed, v):
    """Rigorous 2*L*||delta||_1 bound, fixed volumes, all nine nuisance states.

    On each line segment |x| <= R=max(|x|,|x'|). Gaussian score <=
    (R+|mean|)/sd^2. Student t5 location score <= 6/(2*.4*sqrt(5)).
    Mixture and nuisance-marginal scores are convex combinations. L is the
    maximum component bound over times/states on a given path.
    """
    r = np.maximum(np.abs(x), np.abs(perturbed))
    bi = (r+.75*(-np.expm1(-.1*v)))*(1+.05*v)/.09
    bn = r/.25
    bt = np.full_like(r, 6/(2*.4*np.sqrt(5)))
    per_step = np.maximum.reduce([bi, bn, bt])
    return 2*per_step.max(axis=1)*np.abs(perturbed-x).sum(axis=1)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(histories=2000, seed=20260928):
    results = ROOT/'code/results'; generated = ROOT/'manuscript/generated'
    results.mkdir(parents=True, exist_ok=True); generated.mkdir(parents=True, exist_ok=True)
    rows, stored, experiments = [], {}, {}
    calibration_coefficients={}
    for index,(scenario,pp,signal) in enumerate([('prior_predictive',True,1.),('fixed_default',False,1.),('no_signal',False,0.)]):
        training=simulate(histories,500,seed+100+index,prior_predictive=pp,signal_scale=signal)
        endpoint=np.cumsum(training['x'],axis=1)[:,HORIZONS-1]
        calibration_coefficients[scenario]=fit_endpoint_calibration(endpoint,training['labels'])
        for j,coefficient in enumerate(calibration_coefficients[scenario]):coefficient.update(horizon=int(HORIZONS[j]),seed=seed+100+index,training_scenario=scenario)
    calibration_coefficients['dependent']=calibration_coefficients['prior_predictive']
    (results/'revised_endpoint_coefficients.json').write_text(json.dumps(calibration_coefficients,indent=2)+'\n')
    scenarios = [('fixed_default', False, 0., 1.), ('prior_predictive', True, 0., 1.), ('dependent', True, .6, 1.), ('no_signal', False, 0., 0.)]
    for idx, (scenario, pp, rho, signal) in enumerate(scenarios):
        print(f'Running {scenario}: {histories} histories', flush=True)
        data = simulate(histories, 500, seed+idx, prior_predictive=pp, rho=rho, signal_scale=signal)
        predictions = infer(data['x'], data['v'], HORIZONS, data['params'], signal_scale=signal, all_methods=True)
        coefficients=calibration_coefficients[scenario]
        predictions['calibrated_endpoint']=np.array([c['intercept'] for c in coefficients])+predictions['last_price']*np.array([c['slope'] for c in coefficients])
        stored[scenario+'_labels'] = data['labels']; stored[scenario+'_params'] = data['params']
        # Complete replication-level input histories allow independent recomputation.
        stored[scenario+'_increments'] = data['x']; stored[scenario+'_volumes'] = data['v']
        for method, odds in predictions.items():
            stored[scenario+'_'+method+'_log_odds'] = odds
        baseline = scores(data['labels'][:, None], predictions['last_price'])
        for method, odds in predictions.items():
            brier, loss, info = scores(data['labels'][:, None], odds)
            correctness = (odds > 0) == data['labels'][:, None]
            accuracy = np.where(odds == 0, .5, correctness)
            truth_p = expit((2*data['labels'][:, None]-1)*odds)
            for j, h in enumerate(HORIZONS):
                row = dict(scenario=scenario, method=method, horizon=int(h), n=histories)
                for key, value in [('brier', brier), ('log_loss', loss), ('information_gain', info), ('accuracy', accuracy), ('true_label_probability', truth_p), ('delta_brier_vs_last', brier-baseline[0]), ('delta_log_loss_vs_last', loss-baseline[1])]:
                    avg, lo, hi = mean_interval(value[:, j])
                    row.update({key: avg, key+'_lo':lo, key+'_hi':hi})
                rows.append(row)
        experiments[scenario] = (data, predictions)
    add_calibrated_comparisons(rows,stored)
    write_csv(results/'revised_summary.csv', rows)
    # Symmetric nuisance-orientation experiment computes both label evidences.
    data = experiments['prior_predictive'][0]
    symmetric = np.concatenate([orientation_log_odds(data['x'][i:i+100], data['v'][i:i+100], HORIZONS) for i in range(0, histories, 100)])
    stored['orientation_symmetric_log_odds'] = symmetric
    # Informed-weight sweep is finite-horizon recovery, not structural identification.
    sweep = []
    for j, weight in enumerate([0., .1, .15, .25, .4, .55, .65, .8, 1.]):
        print(f'Weight sweep {weight:.2f}', flush=True)
        d = simulate(histories, 200, seed+20+j, omega=weight)
        pred = infer(d['x'], d['v'], [200], d['params'], all_methods=True)
        stored[f'sweep_{j}_labels'] = d['labels']
        for method in ['marginal', 'oracle', 'last_price']:
            odds = pred[method][:, 0]
            stored[f'sweep_{j}_{method}_log_odds'] = odds
            b, loss, info = scores(d['labels'], odds)
            row = dict(omega=float(weight), method=method, horizon=200, n=histories)
            for key, val in [('accuracy', ((odds>0)==d['labels']).astype(float)), ('brier', b), ('log_loss', loss)]:
                avg, lo, hi = mean_interval(val); row.update({key:avg, key+'_lo':lo, key+'_hi':hi})
            sweep.append(row)
    write_csv(results/'revised_sweep.csv', sweep)
    # Direct KL estimates from iid one-observation pairs (X,V), true label=1.
    # Reflection makes the label=0 KL identical, without needing a second simulation.
    kl_rows = []; n_kl = max(100000, histories*50)
    for j, weight in enumerate([0., .15, .4, .65, 1.]):
        d = simulate(n_kl, 1, seed+50+j, omega=weight)
        x = d['x'][:, 0]*(2*d['labels']-1); v = d['v'][:, 0]
        lp = log_density(x, v, 1, weight, .5)
        for name, w, mu in [('fixed_parameter', weight, .5)]+[('grid_candidate', w, mu) for w, mu in STATES]:
            lr = lp-log_density(x, v, 0, w, mu)
            avg, lo, hi = mean_interval(lr)
            kl_rows.append(dict(true_omega=weight, kind=name, candidate_omega=float(w), candidate_mu=float(mu), n=n_kl, kl=avg, kl_lo=lo, kl_hi=hi, mc_se=float(lr.std(ddof=1)/np.sqrt(n_kl))))
    write_csv(results/'revised_kl.csv', kl_rows)
    projection = []
    for weight in [0., .15, .4, .65, 1.]:
        candidates = [r for r in kl_rows if r['true_omega']==weight and r['kind']=='grid_candidate']
        selected = min(candidates, key=lambda r:r['kl'])
        z = norm.ppf(1-.05/(2*len(STATES)))
        projection.append(dict(true_omega=weight, selected_omega=selected['candidate_omega'], selected_mu=selected['candidate_mu'], estimated_grid_min=selected['kl'], simultaneous_mc_lo=min(r['kl']-z*r['mc_se'] for r in candidates), simultaneous_mc_hi=min(r['kl']+z*r['mc_se'] for r in candidates)))
    write_csv(results/'revised_grid_projection.csv', projection)
    # Paired prior sensitivity, including truth outside the finite support.
    sensitivity = []
    priors = dict(uniform=np.ones(9)/9, low_informed=np.repeat([.6,.3,.1],3)/3, high_informed=np.repeat([.1,.3,.6],3)/3)
    for j, (name, w, m) in enumerate([('on_grid',.4,.5),('off_grid',.3,.6)]):
        print(f'Prior sensitivity {name}', flush=True)
        d=simulate(histories,500,seed+90+j,omega=w,mu=m)
        stored[name+'_sensitivity_labels']=d['labels']
        endpoint=np.cumsum(d['x'],axis=1)[:,np.array([100,500])-1]
        coef=[calibration_coefficients['prior_predictive'][j] for j in [3,5]]
        endpoint_odds=np.array([c['intercept'] for c in coef])+endpoint*np.array([c['slope'] for c in coef])
        stored[name+'_calibrated_endpoint_sensitivity_log_odds']=endpoint_odds
        reference=None
        for prior_name, prior in priors.items():
            odds=infer(d['x'],d['v'],[100,500],prior_weights=prior)['marginal']
            stored[name+'_'+prior_name+'_sensitivity_log_odds']=odds
            b,loss,info=scores(d['labels'][:,None],odds)
            if reference is None:reference=(b,loss)
            for jj,h in enumerate([100,500]):
                row=dict(truth=name,true_omega=w,true_mu=m,prior=prior_name,horizon=h,n=histories)
                for key,val in [('brier',b),('log_loss',loss),('information_gain',info),('delta_brier_vs_uniform',b-reference[0]),('delta_log_loss_vs_uniform',loss-reference[1])]:
                    avg,lo,hi=mean_interval(val[:,jj]);row.update({key:avg,key+'_lo':lo,key+'_hi':hi})
                sensitivity.append(row)
    write_csv(results/'revised_prior_sensitivity.csv',sensitivity)
    # Hellinger affinity gives an independent deterministic-design fractional bound.
    fractional=[]
    for w,m in STATES:
        affinity,error=quad(lambda xx: np.exp(.5*(log_density(xx,1.,1,.4,.5)+log_density(xx,1.,0,w,m))),-np.inf,np.inf,epsabs=1e-10,epsrel=1e-10)
        fractional.append(dict(candidate_omega=float(w),candidate_mu=float(m),volume=1.,s=.5,affinity=float(affinity),quadrature_error=float(error),exponent=float(-np.log(affinity))))
    write_csv(results/'revised_fractional_affinity.csv',fractional)
    fractional_c=min(r['exponent'] for r in fractional)
    write_csv(results/'revised_fractional_bound.csv',[dict(horizon=int(h),exponent=fractional_c,coefficient=9.,error_probability_upper_bound=float(min(1.,9*np.exp(-h*fractional_c)))) for h in [10,25,50,100,200,500,1000,2000,5000]])
    # Additive fixed-volume perturbations of real simulated histories.
    stability = []; base = experiments['fixed_default'][0]
    x, v = base['x'][:, :100], base['v'][:, :100]
    original = infer(x, v, [100])['marginal'][:, 0]
    rng = np.random.default_rng(seed+80)
    direction = rng.standard_normal(x.shape)
    for epsilon in [.0001, .001, .01, .05, .1]:
        perturbed = x+epsilon*direction
        changed = infer(perturbed, v, [100])['marginal'][:, 0]
        bound = stability_bound(x, perturbed, v)
        difference = np.abs(changed-original)
        for i in range(histories):
            stability.append(dict(epsilon=epsilon, replication=i, absolute_log_bf_change=float(difference[i]), bound=float(bound[i]), ratio=float(difference[i]/bound[i]), violation=bool(difference[i]>bound[i]+1e-10)))
    write_csv(results/'revised_stability.csv', stability)
    stored['horizons'] = HORIZONS
    np.savez_compressed(results/'revised_predictions.npz', **stored)
    diagnostics = dict(deterministic_volume_fractional_exponent=fractional_c, symmetric_orientation_max_abs_log_bf=float(np.abs(symmetric).max()), symmetric_orientation_max_abs_bf_minus_one=float(np.abs(np.exp(symmetric)-1).max()), no_signal_max_abs_marginal_log_bf=float(np.abs(experiments['no_signal'][1]['marginal']).max()), stability_violations=sum(r['violation'] for r in stability), stability_max_ratio=max(r['ratio'] for r in stability), grid_projection_note='Minimum of Monte Carlo estimates over nine wrong-label candidate states; not continuous projection. Bonferroni-normal intervals account for nine candidate means asymptotically, not finite-sample guaranteed.')
    (results/'revised_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    make_figures(rows, sweep, stability, experiments, generated)
    make_latex(rows, diagnostics, projection, generated, histories)
    source_files = [Path(__file__), ROOT/'code/test_revised_experiments.py']
    manifest = dict(seed=seed, histories_per_scenario=histories, horizons=HORIZONS.tolist(), nuisance_states=STATES.tolist(), nuisance_prior_weights=[1/9]*9, label_prior=.5, labels='balanced and randomly permuted in each simulation', volumes=dict(distribution='Gamma', shape=2., scale=.5, exogenous=True), opposing_activation='expit((v-1.5)/.3)', opposing_distribution=dict(name='Student t', df=5, scale=.4), informed_sd='.3/sqrt(1+.05*v)', informed_mean='(2*label-1)*mu*(1-exp(-.1*v))', opposing_mean='-(2*label-1)*.3*expit((v-1.5)/.3)', noise_sd=.5, gates='normalize([omega*(1+v)^.5, (1-omega)*2/3, (1-omega)/3*(1+v)^.3])', dependence='Stationary latent Gaussian AR(1), rho=.6. Opposing innovations t5.ppf(Phi(z)); all component one-step marginals preserved. Product inference ignores dependence.', scenarios=[dict(name=s[0], seed=seed+i, prior_predictive=s[1], latent_rho=s[2], signal_scale=s[3]) for i,s in enumerate(scenarios)], confidence_intervals='1.96 standard error across independent paths; paired path differences vs last-price; pointwise, asymptotic Monte Carlo intervals. Balanced label strata have symmetric score laws. No empirical-market inference.', sweep_seed='seed+20+weight_index', sweep_weights=[0.,.1,.15,.25,.4,.55,.65,.8,1.], kl_seed='seed+50+weight_index', kl_n=n_kl, prior_sensitivity_seeds=[seed+90,seed+91], prior_sensitivity_weights={k:v.tolist() for k,v in priors.items()}, stability_seed=seed+80, endpoint_calibration='Unpenalized logistic intercept+slope per horizon, independent training histories; dependent and off-grid reuse prior-predictive fit; fixed default and no-signal use independently matched training.', endpoint_calibration_seeds=[seed+100,seed+101,seed+102], software=dict(matplotlib=__import__('matplotlib').__version__, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()), command=sys.argv, source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}, diagnostics=diagnostics)
    manifest['artifact_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(results.glob('revised_*'))+list(generated.glob('revised_*')) if p.name!='revised_manifest.json'}
    (results/'revised_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(diagnostics, indent=2), flush=True)


def add_calibrated_comparisons(rows,stored):
    for row in rows:
        scenario,method=row['scenario'],row['method']
        j=int(np.where(HORIZONS==int(row['horizon']))[0][0])
        labels=stored[scenario+'_labels']
        reference=scores(labels,stored[scenario+'_calibrated_endpoint_log_odds'][:,j])
        actual=scores(labels,stored[scenario+'_'+method+'_log_odds'][:,j])
        for i,key in enumerate(['delta_brier_vs_calibrated','delta_log_loss_vs_calibrated']):
            avg,lo,hi=mean_interval(actual[i]-reference[i]);row.update({key:avg,key+'_lo':lo,key+'_hi':hi})


def refresh_derived():
    """Regenerate summaries and figures from saved replications; never resimulate."""
    results=ROOT/'code/results';generated=ROOT/'manuscript/generated'
    def read_rows(name):
        with (results/name).open() as f:rows=list(csv.DictReader(f))
        for row in rows:
            for key,value in row.items():
                try:row[key]=float(value)
                except ValueError:pass
        return rows
    rows=read_rows('revised_summary.csv');sweep=read_rows('revised_sweep.csv');stability=read_rows('revised_stability.csv')
    with np.load(results/'revised_predictions.npz') as stored:
        add_calibrated_comparisons(rows,stored)
        experiments={}
        for scenario in ['fixed_default','prior_predictive','dependent','no_signal']:
            experiments[scenario]=({'labels':stored[scenario+'_labels']},{m:stored[scenario+'_'+m+'_log_odds'] for m in METHODS+['calibrated_endpoint']})
        write_csv(results/'revised_summary.csv',rows)
        make_figures(rows,sweep,stability,experiments,generated)
    diagnostics=json.loads((results/'revised_diagnostics.json').read_text())
    make_latex(rows,diagnostics,read_rows('revised_grid_projection.csv'),generated,int(rows[0]['n']))
    manifest=json.loads((results/'revised_manifest.json').read_text())
    manifest['derived_refresh_command']=sys.argv
    manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/test_revised_experiments.py']}
    manifest['artifact_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(results.glob('revised_*'))+list(generated.glob('revised_*')) if p.name!='revised_manifest.json'}
    (results/'revised_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def make_figures(rows, sweep, stability, experiments, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/bip-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9.5, 'axes.titlesize':10, 'axes.labelsize':9.5, 'xtick.labelsize':9, 'ytick.labelsize':9, 'legend.fontsize':9, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42, 'axes.grid':True, 'grid.alpha':.18})
    def save(fig, name):
        fig.savefig(output/(name+'.pdf'), bbox_inches='tight', metadata={'Creator':'revised_experiments.py', 'CreationDate':None})
        fig.savefig(output/(name+'.png'), dpi=180, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), constrained_layout=True)
    for ax, scenario, title in zip(axes, ['fixed_default','prior_predictive'], ['Fixed default nuisance','Nuisance drawn from prior']):
        for method in ['marginal','oracle','last_price','prior']:
            rr = [r for r in rows if r['scenario']==scenario and r['method']==method]
            ax.plot(HORIZONS, [r['true_label_probability'] for r in rr], 'o-', color=COLORS[method], label=DISPLAY[method], ms=3)
            ax.fill_between(HORIZONS, [r['true_label_probability_lo'] for r in rr], [r['true_label_probability_hi'] for r in rr], color=COLORS[method], alpha=.12)
        ax.set(xscale='log', xlabel='Observed increments', ylabel='Mean probability on true label', title=title, ylim=(.4,1.02))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
    save(fig,'revised_concentration')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), constrained_layout=True)
    fig.set_size_inches(7.0, 4.3)
    for ax, metric, title in zip(axes,['brier','log_loss'],['Brier score','Log loss (log scale)']):
        for method in METHODS+['calibrated_endpoint']:
            rr=[r for r in rows if r['scenario']=='prior_predictive' and r['method']==method]
            ax.plot(HORIZONS,[r[metric] for r in rr], 'o-', color=COLORS[method],label=DISPLAY[method],ms=3)
        ax.set(xscale='log',xlabel='Observed increments',ylabel=title)
    axes[1].set_yscale('log')
    axes[1].set_yticks([.2, .5, 1., 2., 5.], labels=['0.2', '0.5', '1', '2', '5'])
    axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
    save(fig,'revised_scores')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), constrained_layout=True)
    for method in ['marginal','oracle','last_price']:
        rr=[r for r in sweep if r['method']==method]
        axes[0].errorbar([r['omega'] for r in rr],[r['accuracy'] for r in rr], yerr=[1.96*(r['accuracy_hi']-r['accuracy'])/1.96 for r in rr], color=COLORS[method],marker='o',ms=3,label=DISPLAY[method],capsize=2)
    axes[0].set(xlabel='Base informed weight',ylabel='Accuracy at 200 increments',ylim=(0,1.02),title='Finite-horizon recovery')
    for scenario, style in [('prior_predictive','-'),('dependent','--')]:
        for method in ['marginal','oracle']:
            rr=[r for r in rows if r['scenario']==scenario and r['method']==method]
            axes[1].plot([r['information_gain'] for r in rr],[r['log_loss'] for r in rr],style+'o',ms=3,color=COLORS[method],label=f"{DISPLAY[method]}: {'iid' if scenario=='prior_predictive' else 'dependent'}")
    axes[1].set(xlabel='Mean posterior/prior KL (nats)',ylabel='Log loss',title='Confidence versus log loss\nSolid: iid; dashed: dependent')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
    save(fig,'revised_robustness')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), constrained_layout=True)
    rr=[r for r in stability if r['epsilon']==.01]
    axes[0].scatter([r['bound'] for r in rr],[r['absolute_log_bf_change'] for r in rr],s=4,alpha=.25,color=COLORS['marginal'])
    axes[0].set(xscale='log',yscale='log',xlabel='Pathwise upper bound',ylabel='Absolute change in log BF',title='Fixed volumes (noise SD 0.01)')
    axes[0].set_xticks([20, 40, 100], labels=['20', '40', '100'])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    eps=np.unique([r['epsilon'] for r in stability]); values=[[r['ratio'] for r in stability if r['epsilon']==e] for e in eps]
    axes[1].boxplot(values,tick_labels=['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '0.05', '0.1'],showfliers=False)
    axes[1].set(xlabel='Perturbation SD',ylabel='Absolute change / bound',title='Zero bound violations')
    save(fig,'revised_stability')
    # Reliability bins have pointwise binomial intervals, not calibration proof.
    fig, axes = plt.subplots(1,2,figsize=(7.0,3.8),constrained_layout=True)
    calibration=[]
    for ax,scenario,title in zip(axes,['prior_predictive','dependent'],['Matched iid prior predictive','Dependent innovations']):
        data,pred=experiments[scenario]
        for method in ['marginal','calibrated_endpoint','last_price']:
            p=expit(pred[method][:,2]); y=data['labels']; bins=np.minimum((p*10).astype(int),9)
            xx=[]; yy=[]; ee=[]
            for k in range(10):
                mask=bins==k; n=int(mask.sum())
                if not n: continue
                predavg=float(p[mask].mean()); obs=float(y[mask].mean())
                # Wilson interval for label frequency inside this probability bin.
                z=1.96; center=(obs+z*z/(2*n))/(1+z*z/n); half=z*np.sqrt(obs*(1-obs)/n+z*z/(4*n*n))/(1+z*z/n)
                xx.append(predavg);yy.append(obs);ee.append([obs-(center-half),center+half-obs])
                calibration.append(dict(scenario=scenario,method=method,horizon=50,bin=k,n=n,mean_probability=predavg,event_frequency=obs,frequency_lo=center-half,frequency_hi=center+half))
            ax.errorbar(xx,yy,yerr=np.maximum(0,np.array(ee).T),color=COLORS[method],marker='o',ms=3,capsize=2,label=DISPLAY[method])
        ax.plot([0,1],[0,1],':',color='#777777');ax.set(xlabel='Mean forecast probability',ylabel='Observed label frequency',title=title,xlim=(0,1),ylim=(0,1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, frameon=False)
    save(fig,'revised_calibration')
    write_csv(ROOT/'code/results/revised_calibration.csv',calibration)


def make_latex(rows, diagnostics, projection, output, histories):
    lines = ['% Generated by code/revised_experiments.py; do not edit manually.',r'\begin{tabular}{llrrrr}',r'\toprule',r'Scenario & Method & Brier & Log loss & $\Delta$ Brier vs calibrated [95\% MC CI] & IG \\',r'\midrule']
    for scenario, label in [('fixed_default','Fixed'),('prior_predictive','Prior predictive'),('dependent','Dependent')]:
        for method in METHODS+['calibrated_endpoint']:
            r=next(r for r in rows if r['scenario']==scenario and r['method']==method and r['horizon']==200)
            delta=f"{r['delta_brier_vs_calibrated']:.3f} [{r['delta_brier_vs_calibrated_lo']:.3f}, {r['delta_brier_vs_calibrated_hi']:.3f}]"
            lines.append(f"{label} & {DISPLAY[method]} & {r['brier']:.3f} & {r['log_loss']:.3f} & {delta} & {r['information_gain']:.3f} \\\\")
        if scenario!='dependent':lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}']
    (output/'revised_tables.tex').write_text('\n'.join(lines)+'\n')
    macros = {'SimulationHistories':str(histories), 'SymmetryMaxLogBF':f"{diagnostics['symmetric_orientation_max_abs_log_bf']:.1f}", 'StabilityViolations':str(diagnostics['stability_violations']), 'StabilityMaxRatio':f"{diagnostics['stability_max_ratio']:.4f}"}
    for scenario,prefix in [('fixed_default','Fixed'),('prior_predictive','Prior'),('dependent','Dependent')]:
        for method,suffix in [('marginal','Marginal'),('oracle','Oracle'),('last_price','Last')]:
            r=next(r for r in rows if r['scenario']==scenario and r['method']==method and r['horizon']==200)
            for field,label in [('brier','Brier'),('log_loss','LogLoss'),('information_gain','IG'),('accuracy','Accuracy')]:macros[prefix+suffix+label]=f"{r[field]:.3f}"
    (output/'revised_macros.tex').write_text('% Automatically generated numerical values.\n'+'\n'.join('\\newcommand{\\'+key+'}{'+val+'}' for key,val in macros.items())+'\n')
    kl = ['% Nine-state projection; simultaneous asymptotic MC intervals.',r'\begin{tabular}{rrrrr}',r'\toprule',r'True $\omega_I$ & Selected $\omega_I$ & Selected $\mu$ & Grid KL minimum & 95\% MC interval \\',r'\midrule']
    for r in projection:kl.append(f"{r['true_omega']:.2f} & {r['selected_omega']:.2f} & {r['selected_mu']:.2f} & {r['estimated_grid_min']:.5f} & [{r['simultaneous_mc_lo']:.5f}, {r['simultaneous_mc_hi']:.5f}] \\\\")
    kl.extend([r'\bottomrule',r'\end{tabular}']);(output/'revised_kl_table.tex').write_text('\n'.join(kl)+'\n')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20260928)
    parser.add_argument('--refresh-derived',action='store_true',help='Regenerate summaries/figures from saved replication arrays')
    args=parser.parse_args()
    if args.histories<2:parser.error('--histories must be at least 2')
    if args.refresh_derived:refresh_derived()
    else:run(args.histories,args.seed)
