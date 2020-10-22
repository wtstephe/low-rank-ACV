'''
solvers.py

All methods here are for computing H^-1 b, where H is the DxD Hessian and b is a
  D x N matrix.
'''
import time
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

def ihvp_cholesky(b, X, D2,
                  regularizer_hessian):
  hess = (D2[:,np.newaxis]*X).T.dot(X) + regularizer_hessian
  return np.linalg.solve(hess, b)


def ihvp_agarwal(b, X, D2,
                 hessian_scaling,
                 regularizer_hessian,
                 S1=None,
                 S2=None,
                 **kwargs):
  '''
  From Agarwal et. al. "Second-order stochastic optimization for machine
     learning in linear time." 2017.
  First get a stochastic estimate of the inverse Hessian, then dot it with the
    vectors b
  Not clear that this provides good accuracy in a reasonable amount of time.
  '''
  N = X.shape[0]
  D = X.shape[1]
  if S1 is None and S2 is None:
    S1 = int(np.sqrt(N)/10)
    S2 = int(10*np.sqrt(N))

  #if self.regularization is not None:
  #  evalRegHess = autograd.hessian(self.regularization)
  #  paramsCpy = self.params.get_free().copy()
  #  regHess = evalRegHess(self.params.get_free())
  #  regHess[-1,-1] = 0.0
  #  self.params.set_free(paramsCpy)

  hinvEsts = np.zeros((S1,D,D))
  for ii in range(S1):
    hinvEsts[ii] = np.eye(D)
    for n in range(1,S2):
      idx = np.random.choice(N)
      H_n = np.outer(X[idx],X[idx]) * D2[idx] * N + regularizer_hessian

      if np.linalg.norm(H_n) >= hessian_scaling*0.9999:
        from IPython import embed; np.set_printoptions(linewidth=150); embed()
        print(np.linalg.norm(H_n))
      #H_n = self.get_single_datapoint_hessian(idx) * N
      H_n /= hessian_scaling
      hinvEsts[ii] = np.eye(D) + (np.eye(D) - H_n).dot(hinvEsts[ii])

  Hinv = np.mean(hinvEsts, axis=0) / hessian_scaling
  return Hinv.dot(b)


def ihvp_exactEvecs(b, X, D2,
                    rank=1,
                    regularizer_hessianVP=None,
                    L2Lambda=None,
                    **kwargs):
  '''
  regularizer_hessianVP should be a function taking a D-dimensional vector x
   and returning \nabla^2 R(\theta) * x. By default, assumes the regularizer
   is L2Lambda*np.linalg.norm(theta[:-1])**2 (i.e., l2 regularization on
   everything but the bias term)

  Uses the Lanczos algorithm to compute the top rank eigenvectors
  '''
  N = X.shape[0]
  D = X.shape[1] - 1
  if regularizer_hessianVP is None:
    eD1 = np.zeros(D+1)
    eD1[D] = 1.0
    regularizer_hessianVP = lambda x: 2*L2Lambda*x - 2*L2Lambda*x[-1]*eD1
    
  Hdotv = scipy.sparse.linalg.LinearOperator(
    shape=(D+1,D+1),
    matvec=lambda v, X=X, D2=D2: ( np.dot(X.T, np.dot(D2[:,np.newaxis]*X, v)) +
                       regularizer_hessianVP(v) )
    )
  evals, evecs = scipy.sparse.linalg.eigsh(Hdotv, k=rank, which='LM')
  return evecs.dot(np.diag(1/evals)).dot(evecs.T).dot(b)


def tropp_sketch(X, D2, L2Lambda, K,
                 seed=1234,
                 omegaForm='normal',
                 preconditioner=None,
                 D1=None):
  '''
  Helper method for ihvp_tropp

  preconditioner is either a DxD numpy array or a function that computes
    preconditioner.dot(x)
  '''
  D = X.shape[1]
  N = X.shape[0]
  
  if preconditioner is None:
    preconditioner = lambda x: x
  elif type(preconditioner) == np.ndarray:
    pre = preconditioner.copy()
    preconditioner = lambda x: pre.dot(x)

  
  ## Form test vectors, omega
  np.random.seed(seed)
  if omegaForm == 'normal':
    # same thing as qr of np.random.normal(size=(D,K)), but drawing the random
    #  numbers in this order ensures that, e.g. the first 5 columns of omega are
    #  the same whether K = 5 or K = 200.
    omega = np.linalg.qr(preconditioner(np.random.normal(size=(K,D)).T))[0]
  elif omegaForm == 'X':
    # Alternative for omega: make the test vectors relate to the data, X
    Z = np.sqrt(np.abs(D1))[:,np.newaxis] * X
    omega = np.linalg.qr(preconditioner(Z.T.dot(np.random.normal(size=(N,K)))))[0]
  elif omegaForm == 'Zvecs':
    # This is for testing only -- don't use this option.
    Z = np.sqrt(np.abs(D1))[:,np.newaxis] * X
    U, S, V = np.linalg.svd(Z)
    omega = np.linalg.qr(preconditioner(V[:K].T))[0]

  ## Form the sketch
  sketch = omega * (2*L2Lambda)
  sketch[-1,:] = 0.0
  sketch += (D2[:,np.newaxis]*X).T.dot(X.dot(omega))
  #for n in range(N):
  #  sketch += np.outer(X[n] * D2[n], X[n].T.dot(omega))
  return sketch, omega

def tropp_summarize_sketch(sketch, omega, rank, eps=1e-7):
  nu = eps * np.linalg.norm(sketch.ravel())
  sketch += nu * omega
  B = omega.T.dot(sketch)
  try:
    C = np.linalg.cholesky((B + B.T)/2)
  except:
    from IPython import embed; embed()
  #E = np.linalg.solve(C, sketch.T).T # == sketch.dot(np.linalg.inv(C.T))
  E = scipy.linalg.solve_triangular(C, sketch.T, lower=True).T
  U, S, V = np.linalg.svd(E,
                          full_matrices=False) #E = U.dot(np.diag(S)).dot(V)
  U = U[:,:rank]
  S = S[:rank]

  return U, S**2 - nu


def ihvp_tropp_LRDiagonal(b, X, D2,
                          L2Lambda,
                          rank=1,
                          K=None,
                          non_fixed_dims=None,
                          seed=1234,
                          **kwargs):
  if K is None:
    K = 2*rank
  sketch, omega = tropp_sketch(X, D2, L2Lambda=0.0, K=K, seed=seed)
  vecsAppx, valsAppx = tropp_summarize_sketch(sketch, omega, rank)
  goodInds = np.where(valsAppx > 1e-8)[0]
  vecsAppx = vecsAppx[:,goodInds]
  valsAppx = valsAppx[goodInds]

  mask = np.eye(X.shape[1]) 
  mask[-1,-1] = 0.0
  LamInv = np.eye(X.shape[1]) * 1./(2*L2Lambda)

  print('TODO : do not explicitly form matrices')
  Ainv = vecsAppx.dot(np.diag(1/valsAppx)).dot(vecsAppx.T)
  from IPython import embed; np.set_printoptions(linewidth=150); embed()
  mid = np.linalg.solve(LamInv + mask.dot(Ainv).dot(mask),
                        mask.dot(Ainv))
  
  hivps = ( Ainv - Ainv.dot(mask).dot(mid) )
  return hivps, 0.0


def ihvp_tropp(b, X, D2,
               L2Lambda,
               rank=1,
               K=None,
               non_fixed_dims=None,
               seed=1234,
               omegaForm='normal',
               returnSchatten=False,
               **kwargs):
  '''
  Returns matrix with columns H^{-1}.dot(b[:,i]), where H is the Hessian
    evaluated
  Method is from Tropp et. al 2017, "Fixed-rank approximation of
    a positive-semidefinite matrix from streaming data"
  '''
  if K is None:
    K = 1*rank
  sketch, omega = tropp_sketch(X, D2, L2Lambda, K,
                               seed=seed,
                               omegaForm=omegaForm,
                               preconditioner=preconditioner)
  vecsAppx, valsAppx = tropp_summarize_sketch(sketch, omega, rank)
  goodInds = np.where(valsAppx > 1e-8)[0]
  
  # Approximately compute H^{-1}.dot(b)
  hivps = vecsAppx[:,goodInds].dot(np.diag(1/valsAppx[goodInds])).dot(vecsAppx[:,goodInds].T).dot(b)
  

  if not returnSchatten:
    return hivps
  else:
    # For diagnostic purposes, compute schatten 1 norm, ||H - \tilde H||_1
    D = X.shape[1]
    hessReg = np.eye(D) * 2 * L2Lambda
    hessReg[-1,-1] = 0.0
    hess = X.T.dot(np.diag(D2)).dot(X) + hessReg

    vals, vecs = np.linalg.eigh(hess)
    goodInds = np.where(vals > 1e-6)[0]   
    schatten1 = np.abs(np.linalg.eigvalsh(vecs[:,goodInds].dot(np.diag(vals[goodInds])).dot(vecs[:,goodInds].T) - vecsAppx.dot(np.diag(valsAppx)).dot(vecsAppx.T))).sum()
    goodInds = np.where(vals > 0)[0]
    hivpsExact = vecs[:,goodInds].dot(np.diag(1/vals[goodInds])).dot(vecs[:,goodInds].T).dot(X.T)
    return hivps, schatten1  

  

def tropp_agree(omega, vecsAppxExtra):
  '''
  If using the tropp approximation, returns the subspace on which
    H and Happx agree. If Q = K (i.e. you didn't truncate the
    eigendecomposition of Happx), then this is just omega.

  Note that this is currently not a computationally efficient function;
    it runs in O(D^3) time!
  '''
  if vecsAppxExtra.shape[1] == 0:
    return omega
  
  D = omega.shape[0]
  M = omega.T.dot(omega)

  # H and Happx agree on omega times the eigenvectors of M with unit
  #   eigenvalues.
  M -= omega.T.dot(vecsAppxExtra).dot(vecsAppxExtra.T.dot(omega))
  vals, vecs = np.linalg.eigh(M)
  inds = np.where(np.abs(vals - 1.0) < 1e-8)[0]
  return omega.dot(vecs[:,inds])

def tropp_error_bound(X, vecsAppx, omega, K, lam,
                      H=None, Happx=None, truncate=False, D2=None):
  '''
  Returns error on the estimate of the Q_n
  '''

  # Get orthonormal basis for space on which Happx's and H's 
  #  inverses agree
  start = time.time()
  agree = tropp_agree(omega, vecsAppx[:,K:])

  if D2 is not None:
    Xd2 = X * D2[:,np.newaxis]
    invAgree = np.linalg.qr(Xd2.T.dot(X.dot(agree)) + lam*agree)[0]
  else:
    invAgree = np.linalg.qr(H.dot(agree))[0]
  projX = np.linalg.norm(X - invAgree.dot(invAgree.T.dot(X.T)).T, 
                         axis=1)**2 
  estErr = 1/lam * (projX)

  if truncate:
    assert(D2 is not None)
    bnorms = np.linalg.norm(X, axis=1)**2 * D2
    maxErr = np.abs(bnorms / (lam + bnorms) * 1 / D2)
    estErr = np.minimum(estErr, maxErr)


  if H is not None and Happx is not None:
    specNorm = np.abs(np.linalg.eigvalsh(np.linalg.inv(H) -
                  np.linalg.inv(Happx))).max()
    tightEstErr = specNorm * (projX * np.abs(D1))
    return estErr, tightEstErr
  else:
    return estErr


def compute_Q(model, K, truncate=True, estErr=False):
  '''
  Estimates x_n^T H\inv x_n for all n.
  '''
  X = model.training_data.X
  lam = model.L2Lambda
  D1 = model.D1
  D2 = model.D2
  diagH = (X**2 * D2[:,np.newaxis]).sum(axis=0) + lam
  preXX = lambda x, diagH=diagH: (X.T.dot(X.dot(x))) / diagH[:,np.newaxis]
  
  start = time.time()
  sketch, omega = tropp_sketch(X, 
                               model.D2, 
                               0.0, 
                               K,
                               preconditioner=preXX,
                               omegaForm='normal',
                               D1=model.D1)
  print('Sketch time =', time.time() - start)
  start = time.time()
  vecsAppx, valsAppx = tropp_summarize_sketch(sketch, 
                                              omega, 
                                              K,)
  print('Summary time =', time.time() - start)

  # Compute \tilde Q_n using decomposition
  start = time.time()
  vappxX = vecsAppx.T.dot(X.T).T
  Xnorms = np.linalg.norm(X, axis=1)**2
  diagInv = valsAppx / (lam + valsAppx)
  Qappx = Xnorms / lam - np.einsum('nk,nk->n', vappxX * diagInv[np.newaxis,:], vappxX) / lam
  
  if truncate:
    bnorms = np.linalg.norm(X, axis=1)**2 * D2
    Qappx = np.minimum(Qappx, bnorms / (model.L2Lambda + bnorms) / D2)
  print('Qn time=', time.time() - start)

  start = time.time()
  if estErr:
    estErr = tropp_error_bound(X,
                               vecsAppx,
                               omega,
                               K,
                               lam,
                               truncate=truncate,
                               D2=model.D2)
  else:
    estErr = None
  print('Error est time=', time.time() - start)
  
  return Qappx, estErr



'''
Bad implementation, doesn't use Woodburry lemma
def compute_Q(model, K, truncate=True, estErr=False):
  #Estimates x_n^T H\inv x_n for all n.

  X = model.training_data.X
  lam = model.L2Lambda
  D1 = model.D1
  D2 = model.D2
  diagH = (X**2 * D2[:,np.newaxis]).sum(axis=0) + lam
  preXX = lambda x, diagH=diagH: (X.T.dot(X.dot(x))) / diagH[:,np.newaxis]
  
  start = time.time()
  sketch, omega = tropp_sketch(X, 
                               model.D2, 
                               0.0, 
                               K,
                               preconditioner=preXX,
                               omegaForm='normal',
                               D1=model.D1)
  print('Sketch time =', time.time() - start)
  start = time.time()
  vecsAppx, valsAppx = tropp_summarize_sketch(sketch, 
                                              omega, 
                                              K,)
  print('Summary time =', time.time() - start)

  # Compute \tilde Q_n using decomposition
  start = time.time()
  vappxX = vecsAppx.T.dot(X.T)
  print(time.time() - start)
  start = time.time() 
  XPerp = (X.T - vecsAppx.dot(vappxX)).T
  print('xperp:', time.time() - start)
  #from IPython import embed;np.set_printoptions(linewidth=80);embed()
  start = time.time()
  ihvpAppx = vecsAppx.dot(np.diag(1/(valsAppx+lam)).dot(vappxX)).T
  print(time.time() - start)
  start = time.time()  
  ihvpAppx += 1/lam * XPerp
  print(time.time() - start)
  start = time.time()  
  Qappx = np.einsum('nd,nd->n', X, ihvpAppx)
  print(time.time() - start)
  start = time.time()
  if truncate:
    bnorms = np.linalg.norm(X, axis=1)**2 * D2
    Qappx = np.minimum(Qappx, bnorms / (model.L2Lambda + bnorms) / D2)
  print('Qn time=', time.time() - start)

  start = time.time()
  if estErr:
    estErr = tropp_error_bound(X,
                               vecsAppx,
                               omega,
                               K,
                               lam,
                               truncate=truncate,
                               D2=model.D2)
  else:
    estErr = None
  print('Error est time=', time.time() - start)
  
  return Qappx, estErr
'''
