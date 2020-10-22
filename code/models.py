'''
Most stuff in here inherits from GeneralizedLinearModel (really this doesn't
  necessarily have to be a GLM..), which provides functionality of
  retrain_with_weights() and fit(). 

Inheriting classes like PoissonRegressionModel mainly need to provide
  eval_objective()
'''
import autograd.numpy as np
import copy
import scipy
import autograd
import scipy
import time
import os

import utils
from operator import mul
import fitL1
import solvers

from collections import namedtuple

class GeneralizedLinearModel(object):
  def __init__(self, training_data, params_init, example_weights, 
               test_data=None, regularization=None):
    if regularization is None:
      regularization = lambda x: 0.0
    self.regularization = regularization
    
    self.training_data = training_data
    self.test_data = test_data
    self.example_weights = copy.deepcopy(example_weights)
    
    self.params = copy.deepcopy(params_init)
    
    self.dParams_dWeights = None
    self.is_a_glm = True

    self.L2Lambda = None


  def eval_objective(self, free_params):
    pass

  def get_params(self):
    return self.params.get_free()

  def dump_state(self, xk):
    '''
    callback to save the state to disk during optimization
    '''
    filename = 'state.txt'

    if not os.path.exists(filename):
      past = np.zeros((0,xk.shape[0]))
    else:
      past = np.loadtxt(filename)
      if past.ndim < 2:
        past = past.reshape(1,-1)
    np.savetxt(filename, np.append(past, xk.reshape(1,-1), axis=0))

  def fit(self,
          warm_start=True,
          label=None,
          save=False, 
          use_fit_L1=False,
          tol=1e-15,
          dump_state=False,
          **kwargs):
    '''
    Trains the model
    '''
    if False and hasattr(self, 'objective_gradient'):
      eval_objective_grad = self.objective_gradient
    else:
      eval_objective_grad = autograd.grad(self.eval_objective)
      
    eval_objective = lambda theta: self.eval_objective(theta)
    eval_objective_hess = autograd.hessian(self.eval_objective)
    eval_objective_hvp = autograd.hessian_vector_product(self.eval_objective)

    if use_fit_L1:
      fitL1.fit_L1(self, **kwargs)
    else:
      if dump_state:
        callback = self.dump_state
      else:
        callback = None
      opt_res = scipy.optimize.minimize(eval_objective,
                                      jac=eval_objective_grad,
                                      hessp=eval_objective_hvp,
                                      hess=eval_objective_hess,
                                      x0=copy.deepcopy(self.params.get_free()),
                                      callback=callback,
                                      method='trust-ncg',
                                      tol=tol,
                                      options={
                                        'initial_trust_radius':0.1,
                                        'max_trust_radius':1,
                                        'gtol':tol,
                                        'disp':False,
                                        'maxiter':100000
                                      })

      self.params.set_free(opt_res.x)
      if np.linalg.norm(opt_res.jac) > .01:
        print('Got grad norm', np.linalg.norm(opt_res.jac))



  # TODO: can we rewrite to avoid rewriting the instance var each time?
  def weighted_model_objective(self, example_weights, free_params):
    ''' The actual objective that we differentiate '''
    self.example_weights = example_weights
    return self.eval_objective(free_params)

  def compute_gradients(self, weights):
    if self.is_a_glm:
      self.compute_derivs()
      grads = (self.D1 * weights)[np.newaxis,:] * self.training_data.X.copy().T
    else:
      dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
      d2Obj_dParamsdWeights = autograd.jacobian(dObj_dParams, argnum=0)
      array_box_go_away = self.params.get_free().copy()
      cur_weights = self.example_weights.copy()

      grads = d2Obj_dParamsdWeights(some_example_weights,
                                    self.params.get_free())
      self.params.set_free(array_box_go_away)
    return grads

  def compute_dParams_dWeights(self, some_example_weights,
                               solver_method='cholesky',
                               non_fixed_dims=None,
                               rank=-1,
                               **kwargs):
    '''
    sets self.jacobian = dParams_dxn for each datapoint x_n
    rank = -1 uses a full-rank matrix solve (i.e. np.linalg.solve on the full
      Hessian). A positive integer uses a low rank approximation in
      inverse_hessian_vector_product  
      
    '''
    if non_fixed_dims is None:
      non_fixed_dims = np.arange(self.params.get_free().shape[0])
    if len(non_fixed_dims) == 0:
      self.dParams_dWeights = np.zeros((0,some_example_weights.shape[0]))
      return
    
    dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
    d2Obj_dParams2 = autograd.jacobian(dObj_dParams, argnum=1)
    d2Obj_dParamsdWeights = autograd.jacobian(dObj_dParams, argnum=0)
    
    # Have to re-copy this into self.params after every autograd call, as
    #  autograd turns self.params.get_free() into an ArrayBox (whereas we want
    #  it to be a numpy array)
    #array_box_go_away = self.params.get_free().copy()
    #cur_weights = self.example_weights.copy()

    start = time.time()
    grads = self.compute_gradients(some_example_weights)
    X = self.training_data.X

    if solver_method == 'cholesky':
      eval_reg_hess = autograd.hessian(self.regularization)
      tmp = self.params.get_free().copy()
      reg_hess = eval_reg_hess(self.params.get_free())
      reg_hess[-1,:] = 0.0
      reg_hess[:,-1] = 0.0
      self.params.set_free(tmp)
      self.dParams_dWeights = -solvers.ihvp_cholesky(grads,
                                                     X,
                                                     self.D2,
                                                regularizer_hessian=reg_hess)
    elif solver_method == 'agarwal':
      eval_reg_hess = autograd.hessian(self.regularization)
      tmp = self.params.get_free().copy()
      reg_hess = eval_reg_hess(self.params.get_free())
      reg_hess[-1,:] = 0.0
      reg_hess[:,-1] = 0.0
      self.params.set_free(tmp)
      self.dParams_dWeights = -solvers.ihvp_agarwal(grads,
                                                     X,
                                                    self.D2,
                                                    regularizer_hessian=reg_hess,
                                                    **kwargs)
    elif solver_method == 'lanczos':
      print('NOTE lanczos currently assumes l2 regularization')
      self.dParams_dWeights = -solvers.ihvp_exactEvecs(grads,
                                                       X,
                                                       self.D2,
                                                       rank=rank,
                                                       L2Lambda=self.L2Lambda)
    elif solver_method == 'tropp':
      print('NOTE tropp currently assumes l2 regularization')
      self.dParams_dWeights = -solvers.ihvp_tropp(grads,
                                                  X,
                                                  self.D2,
                                                  L2Lambda=self.L2Lambda,
                                                  rank=rank)
                                                  
      
      
    #self.params.set_free(array_box_go_away)
    #self.example_weights = cur_weights
    self.non_fixed_dims = non_fixed_dims

  def retrain_with_weights(self, new_example_weights,
                           doIJAppx=False, doNSAppx=False,
                           label=None,
                           is_cv=False,
                           non_fixed_dims=None,
                           **kwargs):
    '''
    in_place: updates weights and params based on the new data

    Can do things a bit more efficiently if it's cross-validation; you actually
    don't need to multiply (KxN) times (Nx1) vector; just select the components
    that have been left out
    ''' 
    if doIJAppx: # i.e. infinitesimal jackknife approx
      delta_example_weights = new_example_weights - self.example_weights
      if is_cv and False:
        left_out_inds = np.where(delta_example_weights == 0)
        new_params = self.params.get_free()
        new_params += self.dParams_dWeights[:,left_out_inds].sum(axis=1)
      else:
        if non_fixed_dims is None:
          new_params = self.params.get_free() + self.dParams_dWeights.dot(
            delta_example_weights)
        else:
          new_params = self.params.get_free()
          new_params[non_fixed_dims] += self.dParams_dWeights.dot(
            delta_example_weights)
    elif doNSAppx: # i.e., Newton step based approx
      if is_cv and self.is_a_glm: # Can do rank-1 update
        n = np.where(new_example_weights != 1)[0]
        new_params = self.params.get_free().copy()
        new_params[non_fixed_dims] += self.loocv_rank_one_updates[n,:].squeeze()
      else:
        self.compute_dParams_dWeights(new_example_weights,
                                      non_fixed_dims=non_fixed_dims)
        delta_example_weights = new_example_weights - self.example_weights
        new_params = self.params.get_free().copy()
        new_params[non_fixed_dims] += self.dParams_dWeights.dot(
          delta_example_weights)
    else: # non-approximate: re-fit the model
      curr_params = copy.copy(self.params.get_free())
      curr_example_weights = copy.copy(self.example_weights)
      self.example_weights = new_example_weights
      self.fit(**kwargs)
      new_params = copy.copy(self.params.get_free())
      
    self.params.set_free(new_params)
    return new_params
  
  def predict_probability(self, X):
    pass

  def get_error(self, test_data, metric):
    pass

  def get_single_datapoint_hessian(self, n):
    X = self.training_data.X
    Y = self.training_data.Y
    weights = self.example_weights
    self.training_data.X = X[n].reshape(1,-1)
    self.training_data.Y = Y[n]
    self.example_weights = np.ones(1)
    array_box_go_away = copy.copy(self.params.get_free())
    dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
    d2Obj_dParams2 = autograd.jacobian(dObj_dParams, argnum=1)
    hess_n = d2Obj_dParams2(self.example_weights, self.params.get_free())
    self.params.set_free(array_box_go_away)

    self.training_data.X = X
    self.training_data.Y = Y
    self.example_weights = weights
    return hess_n

  def get_single_datapoint_hvp(self, n, vec):
    '''
    Returns Hessian.dot(vec), where the Hessian is the Hessian of the objective
       function with just datapoint n
    '''    
    X = self.training_data.X
    Y = self.training_data.Y
    weights = self.example_weights
    self.training_data.X = X[n].reshape(1,-1)
    self.training_data.Y = Y[n]
    self.example_weights = np.ones(1)
    
    array_box_go_away = copy.copy(self.params.get_free())
    eval_hvp = autograd.hessian_vector_product(self.weighted_model_objective,
                                               argnum=1)
    hess_n_dot_vec = eval_hvp(self.example_weights, self.params.get_free(), vec)
    
    self.params.set_free(array_box_go_away)
    self.training_data.X = X
    self.training_data.Y = Y
    self.example_weights = weights
    return hess_n_dot_vec

  def get_all_data_hvp(self, vec):
    '''
    Returns Hessian.dot(vec), where the Hessian is the Hessian of the objective
       function with all the data.
    '''
    array_box_go_away = copy.copy(self.params.get_free())
    eval_hvp = autograd.hessian_vector_product(self.weighted_model_objective,
                                               argnum=1)
    hvp = eval_hvp(self.example_weights, self.params.get_free(), vec)
    
    self.params.set_free(array_box_go_away)
    return hvp

  def compute_hessian(self):
    dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
    d2Obj_dParams2 = autograd.jacobian(dObj_dParams, argnum=1)
    array_box_go_away = self.params.get_free().copy()
    hessian = d2Obj_dParams2(self.example_weights, self.params.get_free())
    self.params.set_free(array_box_go_away)
    self.hessian = hessian

  def compute_restricted_hessian_and_dParamsdWeights(self, dims, weights,
                                                     comp_dParams_dWeights=True):
    '''
    Computes the dims.shape[0] by dims.shape[0] Hessian only along the entries
    in dims (used when using l_1 regularization)
    '''
    theta0 = self.params.get_free()
    
    # Objective to differentiate just along the dimensions specified
    def lowDimObj(weights, thetaOnDims, thetaOffDims, invPerm):
      allDims = np.append(dims, offDims)
      thetaFull = np.append(thetaOnDims, thetaOffDims)[invPerm]
      return self.weighted_model_objective(weights, thetaFull)
    
    offDims = np.setdiff1d(np.arange(self.params.get_free().shape[0]), dims)
    thetaOnDims = theta0[dims]
    thetaOffDims = theta0[offDims]

    # lowDimObj will concatenate thetaOnDims, thetaOffDims, then needs to
    #  un-permute them into the original theta.
    allDims = np.append(dims, offDims)
    invPerm = np.zeros(theta0.shape[0], dtype=np.int32)
    for i, idx in enumerate(allDims):
      invPerm[idx] = i

    evalHess = autograd.hessian(lowDimObj, argnum=1)
    array_box_go_away = self.params.get_free().copy()

    restricted_hess = evalHess(weights,
                               thetaOnDims,
                               thetaOffDims,
                               invPerm)
    self.params.set_free(theta0)

    dObj_dParams = autograd.jacobian(lowDimObj, argnum=1)
    d2Obj_dParamsdWeights = autograd.jacobian(dObj_dParams, argnum=0)

    if comp_dParams_dWeights:
      restricted_dParamsdWeights = d2Obj_dParamsdWeights(weights,
                                                         thetaOnDims,
                                                         thetaOffDims,
                                                         invPerm)
      return restricted_hess, restricted_dParamsdWeights
    else:
      return restricted_hess
    
      
  def hessian_inverse_vector_product(self, vec, hessian_scaling,
                                     S1=None, S2=None, method='stochastic'):
    '''
    From Agarwal et. al. "Second-order stochastic optimization for machine
       learning in linear time." 2017. 

    Not clear that this provides good accuracy in a reasonable amount of time.
    '''
    N = self.training_data.X.shape[0]
    D = vec.shape[0]
    if S1 is None and S2 is None:
      S1 = int(np.ceil(np.sqrt(N)/10))
      S2 = int(np.ceil(10*np.sqrt(N)))

    hivpEsts = np.zeros((S1,D))
    for ii in range(S1):
      hivpEsts[ii] = vec
      for n in range(1,S2):
        idx = np.random.choice(N)
        #H_n_prod_prev = self.get_single_datapoint_hvp(idx, hivpEsts[ii]) * N
        #H_n_prod_prev /= hessian_scaling
        H_n_prod_prev = self.get_all_data_hvp(hivpEsts[ii]) / hessian_scaling
        hivpEsts[ii] = vec + hivpEsts[ii] - H_n_prod_prev
    return np.mean(hivpEsts, axis=0) / hessian_scaling

  def stochastic_hessian_inverse(self, hessian_scaling, S1=None, S2=None):
    '''
    From Agarwal et. al. "Second-order stochastic optimization for machine
       learning in linear time." 2017. 

    Not clear that this provides good accuracy in a reasonable amount of time.
    '''
    self.compute_derivs()
    X = self.training_data.X
    N = self.training_data.X.shape[0]
    D = self.params.get_free().shape[0]
    if S1 is None and S2 is None:
      S1 = int(np.sqrt(N)/10)
      S2 = int(10*np.sqrt(N))

    if self.regularization is not None:
      evalRegHess = autograd.hessian(self.regularization)
      paramsCpy = self.params.get_free().copy()
      regHess = evalRegHess(self.params.get_free())
      regHess[-1,-1] = 0.0
      self.params.set_free(paramsCpy)
      
    hinvEsts = np.zeros((S1,D,D))
    for ii in range(S1):
      hinvEsts[ii] = np.eye(D)
      for n in range(1,S2):
        idx = np.random.choice(N)
        H_n = np.outer(X[idx],X[idx]) * self.D2[idx] * N + regHess

        if np.linalg.norm(H_n) >= hessian_scaling*0.9999:
          print(np.linalg.norm(H_n))
        #H_n = self.get_single_datapoint_hessian(idx) * N
        H_n /= hessian_scaling
        hinvEsts[ii] = np.eye(D) + (np.eye(D) - H_n).dot(hinvEsts[ii])
    return np.mean(hinvEsts, axis=0) / hessian_scaling


  def compute_loocv_rank_one_updates(self,
                                     non_fixed_dims=None,
                                     **kwargs):
    '''
    When the model is a GLM and you're doing approximate LOOCV with the Newton
    step approximation, rank one matrix inverse updates allow you to use only
    O(D^3), rather than O(ND^3) computation.
    '''
    X = self.training_data.X
    N = X.shape[0]

    if non_fixed_dims is None:
      non_fixed_dims = np.arange(self.params.get_free().shape[0])
    if len(non_fixed_dims) == 0:
      self.loocv_rank_one_updates = np.zeros((N,0))
      return
      
        
    X_S = X[:,non_fixed_dims]
    hivps = self.inverse_hessian_vector_product(X.T, **kwargs).T
    X_S_hivps = np.einsum('nd,nd->n', X_S, hivps)
    updates = 1 + (self.D2 * X_S_hivps) / (1 - self.D2 * X_S_hivps)
    self.loocv_rank_one_updates = (updates*self.D1)[:,np.newaxis] * hivps


  def inverse_hessian_vector_product(self, b,
                                     solver_method='cholesky',
                                     rank=1,
                                     **kwargs):
    X = self.training_data.X
    if solver_method == 'cholesky':
      eval_reg_hess = autograd.hessian(self.regularization)
      tmp = self.params.get_free().copy()
      reg_hess = eval_reg_hess(self.params.get_free())
      reg_hess[-1,:] = 0.0
      reg_hess[:,-1] = 0.0
      self.params.set_free(tmp)
      return solvers.ihvp_cholesky(b,
                                   X,
                                   self.D2,
                                   regularizer_hessian=reg_hess)
    elif solver_method == 'agarwal':
      eval_reg_hess = autograd.hessian(self.regularization)
      tmp = self.params.get_free().copy()
      reg_hess = eval_reg_hess(self.params.get_free())
      reg_hess[-1,:] = 0.0
      reg_hess[:,-1] = 0.0
      self.params.set_free(tmp)
      return solvers.ihvp_agarwal(b,
                                  X,
                                  self.D2,
                                  regularizer_hessian=reg_hess,
                                  **kwargs)
    elif solver_method == 'lanczos':
      print('NOTE lanczos currently assumes l2 regularization')
      return solvers.ihvp_exactEvecs(b,
                                     X,
                                     self.D2,
                                     rank=rank,
                                     L2Lambda=self.L2Lambda)
    elif solver_method == 'tropp':
      print('NOTE tropp currently assumes l2 regularization')
      return solvers.ihvp_tropp(b,
                                X,
                                self.D2,
                                L2Lambda=self.L2Lambda,
                                rank=rank)
    

      
class PoissonRegressionModel(GeneralizedLinearModel):
  '''
  Poisson regression with:
           y_n ~ Poi( log(1 + e^{<x_n, \theta>}) )
  '''
  def __init__(self, *args, **kwargs):
    super(PoissonRegressionModel, self).__init__(*args, **kwargs)

  def eval_objective(self, free_params):
    self.params.set_free(free_params)
    params = self.params['w'].get()
    Y = self.training_data.Y
    params_x = np.dot(self.training_data.X, params)
    M = np.maximum(params_x, 0.0)
    lam = np.log(np.exp(0-M) + np.exp(params_x-M)) + M
    ret = Y*np.log(lam + 1e-15) - lam
    ll = (-(ret*self.example_weights).sum())

    return ll + self.regularization(params)

  def get_error(self, test_data, metric="mse"):
    if metric == "mse":
      X = test_data.X
      Y = test_data.Y
      params = self.params['w'].get()
      params_x = np.dot(X, params)
      stacked = np.stack([params_x, np.zeros(params_x.shape[0])], axis=0)
      lam = scipy.special.logsumexp(stacked, axis=0)
      Yhat = lam
      return np.mean((Yhat-Y)**2)

  def compute_derivs(self):
    '''
    lazy slow AD-based implementation ... should actually hand-compute
      these for any serious use.
    '''
    Y = self.training_data.Y
    z = self.training_data.X.dot(self.params.get_free())
    f = lambda z, Y: -(Y*np.log(np.log1p(np.exp(z))) - np.log1p(np.exp(z)))
    grad = autograd.grad(f, argnum=0)
    grad2 = autograd.grad(grad)

    self.D1 = np.zeros(Y.shape[0])
    self.D2 = np.zeros(Y.shape[0])
    for n in range(Y.shape[0]):
      self.D1[n] = grad(z[n], Y[n])
      self.D2[n] = grad2(z[n], Y[n])

      
class ExponentialPoissonRegressionModel(GeneralizedLinearModel):
  '''
  Poisson regression with:
           y_n ~ Poi( e^{<x_n, \theta>} )
  '''
  def __init__(self, *args, **kwargs):
    self.L1Lambda = None
    super(ExponentialPoissonRegressionModel, self).__init__(*args, **kwargs)
    
  def eval_objective(self, free_params):
    self.params.set_free(free_params)
    params = self.params['w'].get()
    Y = self.training_data.Y
    params_x_bias = np.dot(self.training_data.X, params)
    ret = Y*params_x_bias - np.exp(params_x_bias)
    ll = (-(ret*self.example_weights).sum())
    
    return ll + self.regularization(params)

  def fit(self, warm_start=True, label=None, save=False,
          use_glmnet=False,
          **kwargs):
    '''
    Note: use_glmnet only works with CV weights (i.e. all 0 or 1)
    '''
    if not use_glmnet:
      super(ExponentialPoissonRegressionModel, self).fit(warm_start,
                                                         label,
                                                         save,
                                                         **kwargs)

    elif use_glmnet:
      from glmnet_py import glmnet
      lambdau = np.array([self.L1Lambda,])
      inds = self.example_weights.astype(np.bool)
      x = self.training_data.X[inds,:].copy()
      y = self.training_data.Y[inds].copy().astype(np.float)
      fit = glmnet(x=x,
                   y=y,
                   family='poisson',
                   standardize=False,
                   lambdau=lambdau,
                   thresh=1e-20,
                   maxit=10e4,
                   alpha=1.0,
      )
             
  def compute_derivs(self):
    '''
    For use from fitL1.py.
    '''
    Y = self.training_data.Y
    params = self.params.get_free()
    #exp_params_X = np.exp(self.training_data.X.dot(self.params['w'].get()))
    exp_params_X = np.exp(self.training_data.X.dot(params))
    self.D1 = -(Y - exp_params_X)
    self.D2 = -(-exp_params_X)
      
  def get_error(self, test_data, metric="mse"):
    if metric == "mse":
      X = test_data.X
      Y = test_data.Y
      params = self.params['w'].get()
      params_x_bias = np.dot(X, params)
      lam = np.exp(params_x_bias)
      Yhat = lam
      return np.mean((Yhat-Y)**2)

  def compute_bounds(self):
    '''
    Used for low-rank CV paper. Assumes l2 regularization.

    Note these bounds are assuming the objective is **not** scaled
      by 1/N
    '''
    X = self.training_data.X
    D1 = self.D1
    D2 = self.D2
    lam = self.L2Lambda
    thetaHat = self.params.get_free()

    thetaThetanBnd = 1/(lam) * np.abs(D1) * np.linalg.norm(X, axis=1)
    exactBnd = np.abs(X.dot(thetaHat)) + thetaThetanBnd * np.linalg.norm(X, axis=1)
    MnBnd = np.exp(exactBnd)
    LipBnd = ((np.linalg.norm(X, axis=1)**2).sum() - np.linalg.norm(X, axis=1)**2) * MnBnd
    self.IJBnd = LipBnd / (lam**3)  * D1**2 * np.linalg.norm(X, axis=1)**3 + 1/lam**2 * D2 * np.abs(D1) * np.linalg.norm(X, axis=1)**4
    self.NSBnd = LipBnd / (lam**3) * D1**2 * np.linalg.norm(X, axis=1)**3

    

    
class LogisticRegressionModel(GeneralizedLinearModel):
  
  def __init__(self, *args, **kwargs):
    super(LogisticRegressionModel, self).__init__(*args, **kwargs)

  def fit(self, warm_start=True, label=None, save=False,
          use_glmnet=False,
          **kwargs):
    '''
    Note: use_glmnet only works with CV weights (i.e. all 0 or 1)
    '''
    
    if not use_glmnet:
      super(LogisticRegressionModel, self).fit(warm_start,
                                               label,
                                               save,
                                               **kwargs)

    elif use_glmnet:
      from glmnet_py import glmnet
      lambdau = np.array([self.L1Lambda / self.training_data.X.shape[0],])
      inds = self.example_weights.astype(np.bool)
      x = self.training_data.X[inds,:-1].copy()
      y = self.training_data.Y[inds].copy().astype(np.float)
      y[np.where(y==-1)] = 0.0
      fit = glmnet(x=x,
                   y=y,
                   family='binomial',
                   standardize=True,
                   lambdau=lambdau,
                   thresh=1e-10,
                   maxit=10e3,
                   alpha=1.0,
      )
      self.params.set_free(np.append(fit['beta'], 0))
      return

  def eval_objective(self, free_params):
    self.params.set_free(free_params)
    params = self.params['w'].get()
    X = self.training_data.X
    Y = self.training_data.Y
    paramsXY = -Y * (np.dot(X, params))
    M = np.maximum(paramsXY, 0)
    log_likelihood = -(np.log(np.exp(0-M) + np.exp(paramsXY-M)) + M)
    return ( -(log_likelihood*self.example_weights).sum() +
             self.regularization(params) )
  
  def predict_probability(self, X):
    return utils.sigmoid(X, self.params.get_free())

  def predict_target(self, X):
    probs = self.predict_probability(X)
    probs[np.where(probs > .5)] = 1
    probs[np.where(probs <= .5)] = -1
    return probs

  def compute_derivs(self):
    '''
    For use from fitL1.py
    '''
    Y = self.training_data.Y
    params = self.params.get_free()
    exp_params_XY = np.exp(Y *
              self.training_data.X.dot(params))
    self.D1 = -Y/ (1 + exp_params_XY)
    self.D2 = -Y*self.D1 - (self.D1)**2
    
  def get_error(self, test_data, metric='log_likelihood'):
    if metric == "accuracy":
      # change Y_Test to 01 if required
      return 1.0 * np.where(
        self.predict_target(test_data.X) != test_data.Y)[0].shape[0] / test_data.N
    elif metric == 'log_likelihood':
      train_data = self.training_data
      weights = self.example_weights
      self.training_data = test_data
      self.example_weights = np.ones(test_data.X.shape[0])
      nll = self.eval_objective(self.params.get_free())
      nll -= self.regularization(self.params.get_free())
      self.training_data = train_data
      self.example_weights = weights
      return nll / test_data.X.shape[0]

class LinearRegressionModel(GeneralizedLinearModel):
  def __init__(self, *args, **kwargs):
    super(LinearRegressionModel, self).__init__(*args, **kwargs)

  def eval_objective(self, free_params):
    '''
    Objective that we minimize; \sum_n w_n f(x_n, \theta) + ||\theta||_2
    '''
    self.params.set_free(free_params)
    params = self.params['w'].get()
    params_x = np.dot(self.training_data.X, params)
    sq_error = (self.training_data.Y - params_x)**2 * self.example_weights
    return sq_error.sum() + self.regularization(params[:-1])

  def get_error(self, test_data, metric="mse"):
    if metric == "mse":
      Yhat = np.dot(test_data.X, self.params.get_free())
      Y = test_data.Y
      return np.mean((Yhat - Y)**2)

  def compute_derivs(self):
    '''
    First and second derivatives of link function, used in fitL1.py
    '''
    Y = self.training_data.Y
    params_x = self.training_data.X.dot(self.params.get_free())
    self.D1 = -2*(Y - params_x)
    self.D2 = 2*np.ones(Y.shape[0])
    
  def fit(self, warm_start=True, label=None, save=False,
          use_glmnet=False, **kwargs):
    '''
    Note: this only works with CV weights (i.e. all 0 or 1)
    '''
    if not use_glmnet:
      super(LinearRegressionModel, self).fit(warm_start,
                                             label,
                                             save,
                                             **kwargs)

    elif use_glmnet:
      from glmnet_py import glmnet
      inds = self.example_weights.astype(np.bool)
      x = self.training_data.X[inds,:].copy()
      y = self.training_data.Y[inds].copy().astype(np.float)
      lambdau = np.array([self.L1Lambda/(2*x.shape[0]),])
      
      fit = glmnet(x=x[:,:-1],
                   y=y,
                   family='gaussian',
                   standardize=True,
                   lambdau=lambdau,
                   thresh=1e-10,
                   maxit=10e4,
                   alpha=1.0,
      )
      self.params.set_free(np.append(np.squeeze(fit['beta']), fit['a0']))

      
class ProbitRegressionModel(GeneralizedLinearModel):
  def __init__(self, *args, **kwargs):
    super(ProbitRegressionModel, self).__init__(*args, **kwargs)

  def eval_objective(self, free_params):
    self.params.set_free(free_params)
    params_no_bias = self.params['w'].get()[:-1]
    bias = self.params['w'].get()[-1]
    y_x_params = self.training_data.Y * (
      np.dot(self.training_data.X, params_no_bias) + bias)

    log_likelihood = \
                autograd.scipy.stats.norm.logcdf(y_x_params) * self.example_weights
    return -(log_likelihood).sum() + self.regularization(params_no_bias)

  def predict_probability(self, X):
    params_no_bias = self.params.get_free()[:-1]
    bias = self.params.get_free()[-1]
    return autograd.scipy.stats.norm.cdf(X.dot(params_no_bias) + bias)

  def predict_target(self, X):
    probs = self.predict_probability(X)
    probs[np.where(probs > .5)] = 1
    probs[np.where(probs <= .5)] = -1
    return probs

  def get_error(self, test_data, metric="log_likelihood"):
    if metric == "accuracy":
      # change Y_Test to 01 if required
      return np.where(
        self.predict_target(test_data.X) != test_data.Y)[0].shape[0] / test_data.N
    elif metric == 'log_likelihood':
      train_data = self.training_data
      weights = self.example_weights
      self.training_data = test_data
      self.example_weights = np.ones(test_data.X.shape[0])
      nll = self.eval_objective(self.params.get_free())
      self.training_data = train_data
      return nll / test_data.X.shape[0]



