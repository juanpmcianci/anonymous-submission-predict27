"""Training-only conditional finite-mixture classifier for market histories.

The empirical protocol and revision/empirical_model_design.md fix the catalogue.
Each row is a path of logit increments with aligned actual interval volumes.
One nuisance catalogue index is shared across the entire path. This classifier
assumes conditionally independent increments; it makes no exogeneity claim.
"""
from __future__ import annotations
import hashlib
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.special import expit, gammaln, logsumexp

MODEL_VERSION='empirical-finite-mixture-v1'
CONFIG={
    'weight_grid':[.2,.5,.8],
    'amplitude_grid':[0.,.1,.3,.6],
    'robust_scale_mad_multiplier':1.4826,
    'scale_floor_logit_units':.05,
    'volume_gate':'2*r/(1+r)',
    'volume_normalizer':'training median strictly positive interval volume',
    'directional_gaussian_sd_volume_coefficient':.25,
    'opposite_mean_multiplier':.5,
    'component_gate_exponents':[.5,.3,0.],
    'opposite_and_centered_base_split':[.5,.5],
    'student_df':5.,
    'student_scale_multiplier':float(np.sqrt(3/5)),
    'catalogue_total_pseudocount':1.,
    'label_prior_beta_pseudocount_each':.5,
}


def _normal_logpdf(x,loc,log_sd):
    # log-domain squared residual avoids intermediate division overflow.
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        log_abs_residual=np.log(np.abs(x-loc))
        return -.5*np.exp(2*(log_abs_residual-log_sd))-log_sd-.5*np.log(2*np.pi)


def _check_arrays(z,volume):
    z=np.asarray(z,dtype=float);volume=np.asarray(volume,dtype=float)
    if z.ndim!=2 or volume.shape!=z.shape or not z.size or z.shape[1]<1:
        raise ValueError('z and volume must have matching nonempty (paths, increments) shapes')
    if not np.isfinite(z).all() or not np.isfinite(volume).all() or np.any(volume<0):
        raise ValueError('Increments must be finite and volumes finite and nonnegative')
    return z,volume


class EmpiricalMixture:
    """Exact finite catalogue, fitted solely from the arrays passed to fit."""
    def __init__(self,max_iter=2000,tol=1e-9):
        if int(max_iter)!=max_iter or max_iter<1 or not np.isfinite(tol) or tol<=0:
            raise ValueError('max_iter and tol must be positive')
        self.max_iter=int(max_iter);self.tol=float(tol)
        self.fitted_=False

    def fit(self,z,volume,y,use_volume=True):
        z,volume=_check_arrays(z,volume)
        y=np.asarray(y)
        if y.shape!=(len(z),) or not np.isin(y,[0,1]).all():
            raise ValueError('y must contain exactly one binary outcome per path')
        y=y.astype(int)
        positive=volume[volume>0]
        if not len(positive):
            raise ValueError('Training requires at least one strictly positive volume')
        with np.errstate(over='ignore',invalid='ignore'):
            median=float(np.median(z))
            robust_scale=float(CONFIG['robust_scale_mad_multiplier']*np.median(np.abs(z-median)))
        if not np.isfinite(robust_scale):
            raise ValueError('Training robust scale is outside finite floating-point range')
        self.scale_=max(robust_scale,CONFIG['scale_floor_logit_units'])
        self.volume_median_=float(np.median(positive))
        self.use_volume_=bool(use_volume)
        self.catalogue_=np.array([(w,a) for w in CONFIG['weight_grid'] for a in CONFIG['amplitude_grid']],dtype=float)
        j=len(self.catalogue_)
        self.weights_=np.full(j,1/j)
        self.label_prior_=float((y.sum()+.5)/(len(y)+1))
        self.training_shape_=list(z.shape)
        self.training_positive_labels_=int(y.sum())
        # Scalers are now frozen. EM receives only this training likelihood matrix.
        self.fitted_=True
        train_logs=self.step_log_densities(z,volume,y).sum(axis=1)
        tau=float(CONFIG['catalogue_total_pseudocount'])
        def objective(weights):
            likelihood=float(logsumexp(train_logs+np.log(weights),axis=1).sum())
            penalty=float((tau/j)*np.log(weights).sum())
            return likelihood,likelihood+penalty
        likelihood,penalized=objective(self.weights_)
        likelihood_trace=[likelihood];objective_trace=[penalized]
        converged=False
        for iteration in range(1,self.max_iter+1):
            log_resp=train_logs+np.log(self.weights_)
            responsibilities=np.exp(log_resp-logsumexp(log_resp,axis=1,keepdims=True))
            new_weights=(responsibilities.sum(axis=0)+tau/j)/(len(y)+tau)
            likelihood,new_penalized=objective(new_weights)
            improvement=new_penalized-penalized
            if improvement < -1e-8*(1+abs(penalized)):
                raise RuntimeError('Penalized EM objective decreased beyond numerical tolerance')
            self.weights_=new_weights
            likelihood_trace.append(likelihood);objective_trace.append(new_penalized)
            converged=abs(improvement)<=self.tol*(1+abs(penalized))
            penalized=new_penalized
            if converged:break
        self.diagnostics_={
            'converged':bool(converged),'iterations':iteration,
            'max_iter':self.max_iter,'relative_objective_tolerance':self.tol,
            'log_likelihood':likelihood_trace,'penalized_objective':objective_trace,
            'final_max_responsibility_sum_error':float(np.max(np.abs(responsibilities.sum(axis=1)-1))),
            'minimum_catalogue_weight':float(self.weights_.min()),
            'scale_floor_active':bool(robust_scale<CONFIG['scale_floor_logit_units']),
            'unfloored_training_mad_scale':robust_scale,
        }
        return self

    def _require_fitted(self):
        if not self.fitted_:
            raise ValueError('Fit the model on training histories before prediction')

    def step_log_densities(self,z,volume,label):
        """Return shape (paths, increments, states), normalized over real increments.

        label may be scalar 0/1 or a vector of training labels. Volume enters in
        the log domain, so finite extreme volumes cannot overflow normalization.
        """
        self._require_fitted()
        z,volume=_check_arrays(z,volume)
        labels=np.asarray(label)
        if labels.ndim==0:labels=np.full(len(z),labels)
        if labels.shape!=(len(z),) or not np.isin(labels,[0,1]).all():
            raise ValueError('label must be binary scalar or one binary label per path')
        if self.use_volume_:
            with np.errstate(divide='ignore'):
                log_r=np.log(volume)-np.log(self.volume_median_)
        else:
            log_r=np.zeros_like(volume)
        log_one_plus_r=np.logaddexp(0.,log_r)
        gate=2*expit(log_r)
        w=self.catalogue_[:,0];a=self.catalogue_[:,1]
        sign=(2*labels-1)[:,None,None]
        mean=sign*(a*self.scale_)*gate[...,None]
        log_s=np.log(self.scale_)
        log_sd_plus=log_s-.5*np.logaddexp(0.,np.log(.25)+log_r)
        positive=_normal_logpdf(z[...,None],mean,log_sd_plus[...,None])
        opposing=_normal_logpdf(z[...,None],-.5*mean,log_s)
        log_student_scale=log_s+np.log(CONFIG['student_scale_multiplier'])
        with np.errstate(divide='ignore'):
            log_abs_standardized=np.log(np.abs(z))-log_student_scale
        # log1p(x*x/nu) evaluated without ever squaring x.
        centered=(gammaln(3)-gammaln(2.5)-.5*np.log(5*np.pi)-log_student_scale
                  -3*np.logaddexp(0.,2*log_abs_standardized-np.log(5)))
        log_base_other=np.log1p(-w)-np.log(2)
        gate_logs=np.stack(np.broadcast_arrays(
            np.log(w)+.5*log_one_plus_r[...,None],
            log_base_other+.3*log_one_plus_r[...,None],
            log_base_other+np.zeros_like(log_one_plus_r[...,None])),axis=-1)
        gate_logs-=logsumexp(gate_logs,axis=-1,keepdims=True)
        components=np.stack(np.broadcast_arrays(positive,opposing,centered[...,None]),axis=-1)
        return logsumexp(components+gate_logs,axis=-1)

    def log_evidence(self,z,volume,label):
        """Logsumexp over states AFTER summing increments within each state."""
        path_log_likelihood=self.step_log_densities(z,volume,label).sum(axis=1)
        return logsumexp(path_log_likelihood+np.log(self.weights_),axis=1)

    def log_bayes_factor(self,z,volume):
        return self.log_evidence(z,volume,1)-self.log_evidence(z,volume,0)

    def predict_proba(self,z,volume):
        self._require_fitted()
        prior_log_odds=np.log(self.label_prior_)-np.log1p(-self.label_prior_)
        return expit(prior_log_odds+self.log_bayes_factor(z,volume))

    def to_dict(self):
        self._require_fitted()
        return {
            'model_version':MODEL_VERSION,'config':deepcopy(CONFIG),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scale':self.scale_,'volume_median':self.volume_median_,
            'use_volume':self.use_volume_,'catalogue':self.catalogue_.tolist(),
            'catalogue_weights':self.weights_.tolist(),'label_prior':self.label_prior_,
            'training_shape':self.training_shape_.copy(),
            'training_positive_labels':self.training_positive_labels_,
            'max_iter':self.max_iter,'tol':self.tol,
            'diagnostics':{k:v.copy() if isinstance(v,list) else v for k,v in self.diagnostics_.items()},
        }

    @classmethod
    def from_dict(cls,payload):
        if payload.get('model_version')!=MODEL_VERSION or payload.get('config')!=CONFIG:
            raise ValueError('Serialized model version/config does not match this implementation')
        model=cls(payload['max_iter'],payload['tol'])
        model.scale_=float(payload['scale']);model.volume_median_=float(payload['volume_median'])
        model.use_volume_=bool(payload['use_volume'])
        model.catalogue_=np.array(payload['catalogue'],dtype=float)
        model.weights_=np.array(payload['catalogue_weights'],dtype=float)
        model.label_prior_=float(payload['label_prior'])
        if (not np.isfinite([model.scale_,model.volume_median_,model.label_prior_]).all()
            or model.scale_<=0 or model.volume_median_<=0 or not 0<model.label_prior_<1
            or model.catalogue_.ndim!=2 or model.catalogue_.shape[1]!=2
            or not np.isfinite(model.catalogue_).all()
            or np.any(model.catalogue_[:,0]<=0) or np.any(model.catalogue_[:,0]>=1)
            or np.any(model.catalogue_[:,1]<0)
            or model.weights_.shape!=(len(model.catalogue_),)
            or not np.isfinite(model.weights_).all() or np.any(model.weights_<=0)
            or not np.isclose(model.weights_.sum(),1.)):
            raise ValueError('Invalid serialized model parameters')
        model.training_shape_=list(payload['training_shape'])
        model.training_positive_labels_=int(payload['training_positive_labels'])
        model.diagnostics_={k:v.copy() if isinstance(v,list) else v for k,v in payload['diagnostics'].items()}
        model.fitted_=True
        return model


GAUSSIAN_CONFIG={
    'model_version':'asset-gaussian-history-v1',
    'covariance_shrinkage_to_diagonal':.1,
    'diagonal_variance_floor':1e-4,
    'emergency_eigenvalue_floor':1e-8,
    'label_prior_beta_pseudocount_each':.5,
}


def _check_gaussian_inputs(z,asset):
    z=np.asarray(z,dtype=float);asset=np.asarray(asset)
    if z.ndim!=2 or not z.size or not np.isfinite(z).all() or asset.shape!=(len(z),):
        raise ValueError('Require finite nonempty paths and one asset identifier per row')
    if any(a is None or str(a) in ('','nan','None') for a in asset):
        raise ValueError('Asset identifiers must be nonmissing')
    return z,asset.astype(str)


class GaussianHistory:
    """Asset-specific Gaussian discriminant with a shared class covariance.

    Means, covariance and label priors are fitted on training paths only. The
    multivariate covariance preserves within-path dependence. Prediction uses
    one fixed path dimension and has no volume input or market-price prior.
    """
    def __init__(self):
        self.fitted_=False

    def fit(self,z,asset,y):
        z,asset=_check_gaussian_inputs(z,asset)
        y=np.asarray(y)
        if y.shape!=(len(z),) or not np.isin(y,[0,1]).all():
            raise ValueError('y must contain one binary label per path')
        self.n_features_=z.shape[1]
        self.asset_states_={}
        for name in np.unique(asset):
            mask=asset==name;paths=z[mask];labels=y[mask].astype(int)
            if len(np.unique(labels))!=2:
                raise ValueError('Each training asset requires both outcome classes')
            means=np.stack([paths[labels==label].mean(axis=0) for label in (0,1)])
            residual=paths-means[labels]
            empirical=residual.T@residual/len(paths)
            covariance=.9*empirical+.1*np.diag(np.diag(empirical))
            diagonal=np.diag(covariance).copy()
            floor_mask=diagonal<GAUSSIAN_CONFIG['diagonal_variance_floor']
            covariance[np.diag_indices_from(covariance)]=np.maximum(diagonal,GAUSSIAN_CONFIG['diagonal_variance_floor'])
            covariance=.5*(covariance+covariance.T)
            eigenvalues,eigenvectors=np.linalg.eigh(covariance)
            minimum_before=float(eigenvalues.min())
            apply_eigen_floor=minimum_before<GAUSSIAN_CONFIG['emergency_eigenvalue_floor']
            if apply_eigen_floor:
                covariance=(eigenvectors*np.maximum(eigenvalues,GAUSSIAN_CONFIG['emergency_eigenvalue_floor']))@eigenvectors.T
                covariance=.5*(covariance+covariance.T)
            np.linalg.cholesky(covariance)
            coefficient=np.linalg.solve(covariance,means[1]-means[0])
            self.asset_states_[name]={
                'means':means,'covariance':covariance,'coefficient':coefficient,
                'label_prior':float((labels.sum()+.5)/(len(labels)+1)),
                'n_training':len(labels),'n_positive':int(labels.sum()),
                'diagnostics':{
                    'diagonal_floor_count':int(floor_mask.sum()),
                    'eigenvalue_floor_applied':bool(apply_eigen_floor),
                    'minimum_eigenvalue_before_emergency_floor':minimum_before,
                    'minimum_final_eigenvalue':float(np.linalg.eigvalsh(covariance).min()),
                },
            }
        self.fitted_=True
        return self

    def _prediction_inputs(self,z,asset):
        if not self.fitted_:raise ValueError('Fit the Gaussian model before prediction')
        z,asset=_check_gaussian_inputs(z,asset)
        if z.shape[1]!=self.n_features_:
            raise ValueError('Prediction path length must match the training horizon')
        unknown=set(np.unique(asset))-set(self.asset_states_)
        if unknown:raise ValueError('Prediction contains an asset absent from training')
        return z,asset

    def log_bayes_factor(self,z,asset):
        z,asset=self._prediction_inputs(z,asset)
        result=np.empty(len(z))
        for name in np.unique(asset):
            mask=asset==name;state=self.asset_states_[name]
            midpoint=.5*(state['means'][0]+state['means'][1])
            result[mask]=(z[mask]-midpoint)@state['coefficient']
        return result

    def predict_proba(self,z,asset):
        z,asset=self._prediction_inputs(z,asset)
        odds=self.log_bayes_factor(z,asset)
        prior=np.array([self.asset_states_[name]['label_prior'] for name in asset])
        return expit(odds+np.log(prior)-np.log1p(-prior))

    def to_dict(self):
        if not self.fitted_:raise ValueError('Fit the Gaussian model before serialization')
        states={}
        for name,state in self.asset_states_.items():
            states[name]={key:value.tolist() if isinstance(value,np.ndarray) else deepcopy(value) for key,value in state.items()}
        return {'config':deepcopy(GAUSSIAN_CONFIG),'n_features':self.n_features_,
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'assets':states}

    @classmethod
    def from_dict(cls,payload):
        if payload.get('config')!=GAUSSIAN_CONFIG:
            raise ValueError('Serialized Gaussian configuration does not match this implementation')
        model=cls();model.n_features_=int(payload['n_features']);model.asset_states_={}
        for name,serialized in payload['assets'].items():
            state=deepcopy(serialized)
            for key in ['means','covariance','coefficient']:state[key]=np.asarray(state[key],dtype=float)
            if (state['means'].shape!=(2,model.n_features_)
                or state['covariance'].shape!=(model.n_features_,model.n_features_)
                or state['coefficient'].shape!=(model.n_features_,)
                or not 0<state['label_prior']<1
                or not all(np.isfinite(state[key]).all() for key in ['means','covariance','coefficient'])):
                raise ValueError('Invalid serialized Gaussian parameters')
            np.linalg.cholesky(state['covariance'])
            expected=np.linalg.solve(state['covariance'],state['means'][1]-state['means'][0])
            if not np.allclose(expected,state['coefficient'],rtol=1e-10,atol=1e-12):
                raise ValueError('Serialized discriminant does not match means and covariance')
            model.asset_states_[str(name)]=state
        if not model.asset_states_:raise ValueError('Serialized Gaussian model has no assets')
        model.fitted_=True
        return model
