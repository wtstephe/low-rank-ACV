import autograd.numpy as np
import time
import scipy.sparse

import sys
sys.path.append('./..')
import expUtils
import solvers
import datasets
import models
import Params
import matplotlib.pyplot as plt
import retrainingPlans



seed = 1234
datasetType = datasets.SyntheticPoissonDatasetGenerator
modelType = models.ExponentialPoissonRegressionModel
N = 800
D = 500
lam = 1.0 * N
Xrank = 50
lowRankNoise = 0.001
nIter = 5

exactAss = np.zeros(nIter)
NSAss = np.zeros(nIter)
NSAssUpperBnd = np.zeros(nIter)
NSAssLowerBnd = np.zeros(nIter)
IJAss = np.zeros(nIter)
IJAssUpperBnd = np.zeros(nIter)
IJAssLowerBnd = np.zeros(nIter)

for ii in range(nIter):

  # Get data and fit model
  train, test = datasetType().get_dataset(Ntrain=N,
                                          D=D,
                                          Xrank=Xrank,
                                          lowRankNoise=lowRankNoise,
                                          data_seed=seed+ii,
                                          param_seed=seed+ii)
  regularization = lambda x: lam / 2 * np.linalg.norm(x)**2
  params = Params.Param(train.X.shape[1])
  params.set_free(np.random.normal(size=(train.X.shape[1])))
  model = modelType(training_data=train,
            params_init=params,
            example_weights=np.ones(train.X.shape[0]),
            regularization=regularization,)
  model.L2Lambda = lam
  model.fit()
  model.compute_derivs()
  model.compute_hessian()
  thetaHat = model.params.get_free().copy()
  X = train.X
  Y = train.Y
  D1 = model.D1
  D2 = model.D2
  iprods = np.linalg.solve(model.hessian, X.T).T

  # Comptue exact and approximate Q_n
  Q = np.einsum('nd,nd->n', iprods, X)
  Qappx, estQErr = solvers.compute_Q(model,
                                     K=Xrank+5,
                                     estErr=True)
  print(X.shape)

  exactCV, exactParams, sets = retrainingPlans.leave_k_out_cv(
    model, 
    k=1, 
    method="exact",
    B=100,
    hold_outs='stochastic')

  sets = np.array(sets).squeeze().astype(np.int32)
  exact = np.einsum('nd,nd->n', exactParams, X[sets])


  NS = np.zeros(sets.shape[0])
  IJ = np.zeros(sets.shape[0])
  NSAppx = np.zeros(sets.shape[0])
  IJAppx = np.zeros(sets.shape[0])
  IJAppxBnd = np.zeros(sets.shape[0])
  for idx, n in enumerate(sets):
    IJParams = thetaHat + D1[n] * iprods[n]
    NS[idx] = np.inner(thetaHat, X[n]) + D1[n] * Q[n] / (1-D2[n]*Q[n])
    IJ[idx] = np.inner(thetaHat, X[n]) + D1[n] * Q[n]
    IJAppx[idx] = np.inner(thetaHat, X[n]) + D1[n] * Qappx[n]
    IJAppxBnd[idx] = np.abs(D1[n]) * estQErr[n]

  model.compute_bounds()
  NSBnd = model.NSBnd
  IJBnd = model.IJBnd
  totalIJBnd = IJAppxBnd + IJBnd[sets]

  print('NS bound holds:', np.all(np.abs(NS - exact) < NSBnd[sets]))
  print('IJ bound holds:', np.all(np.abs(IJ - exact) < IJBnd[sets]))

  predErrsExact = (Y[sets] - np.exp(exact))**2
  predErrsNS = (Y[sets] - np.exp(NS))**2
  predErrsNSUpper = (Y[sets] - np.exp(NS + NSBnd[sets]))**2
  predErrsNSLower = (Y[sets] - np.exp(NS - NSBnd[sets]))**2
  predErrsNSUpperBnd = np.maximum(predErrsNSUpper, predErrsNSLower)
  predErrsNSLowerBnd = np.minimum(predErrsNSUpper, predErrsNSLower)
  sortInds = np.argsort(predErrsNS)
  predErrsNSUpperBnd[np.isinf(predErrsNSUpperBnd)] = 0.0
  predErrsNSLowerBnd[np.isinf(predErrsNSLowerBnd)] = 0.0

  predErrsExact = (Y[sets] - np.exp(exact))**2
  predErrsIJ = (Y[sets] - np.exp(IJ))**2
  predErrsIJUpper = (Y[sets] - np.exp(IJ + totalIJBnd))**2
  predErrsIJLower = (Y[sets] - np.exp(IJ - totalIJBnd))**2
  predErrsIJUpperBnd = np.maximum(predErrsIJUpper, predErrsIJLower)
  predErrsIJLowerBnd = np.minimum(predErrsIJUpper, predErrsIJLower)
  sortInds = np.argsort(predErrsIJ)
  predErrsIJUpperBnd[predErrsIJUpperBnd > 1e5] = 0.0
  predErrsIJLowerBnd[predErrsIJLowerBnd > 1e5] = 0.0

  exactAss[ii] = predErrsExact.mean()
  IJAss[ii] = predErrsIJ.mean()
  IJAssUpperBnd[ii] = predErrsIJUpperBnd.mean()
  IJAssLowerBnd[ii] = predErrsIJLowerBnd.mean()
  NSAss[ii] = predErrsNS.mean()
  NSAssUpperBnd[ii] = predErrsNSUpperBnd.mean()
  NSAssLowerBnd[ii] = predErrsNSLowerBnd.mean()

tickFontsize = 16
linewidth = 3.0
axlabelFontsize = 20
titleFontsize = 18
legendFontsize = 16

upTo = None

xs = np.arange(nIter) + 1
jitter = 0.15
sortInds = np.argsort(exactAss)
plt.figure(figsize=(5.5,5))
errBars = np.vstack([IJAssLowerBnd[sortInds],
                     IJAssUpperBnd[sortInds]])
plt.scatter(xs[:upTo], exactAss[sortInds][:upTo],
            label='Exact CV',
            c='b')
plt.errorbar(xs[:upTo]+jitter, IJAss[sortInds][:upTo],
             yerr=errBars[:,:upTo],
             linestyle='none',
             c='r',
             capsize=5.0)
plt.scatter(xs[:upTo]+jitter, IJAss[sortInds][:upTo],
            label = r'$\widetilde\mathrm{IJ}$',
            c='r')
plt.legend(fontsize=legendFontsize)
plt.xlabel('Trial number', fontsize=axlabelFontsize)
plt.ylabel('Mean prediction error $y_n$\'s', fontsize=axlabelFontsize)
plt.title('Exact CV vs bounds across trials',
          fontsize=titleFontsize)
plt.gca().tick_params(axis='both',
                      which='major',
                      labelsize=tickFontsize)
plt.tight_layout()
#plt.savefig('C:YOUR_FILEPATH/averageTrialErrorBounds-Poisson-IJ.png', bbox='tight')

s = 10
sortInds = np.argsort(predErrsExact)
plt.figure(figsize=(5.5,5))
plt.scatter(np.arange(sets.shape[0]),
         predErrsExact[sortInds],
         label='Exact CV',
            c='b',
            s=s)
plt.scatter(np.arange(sets.shape[0]),
         predErrsIJUpperBnd[sortInds],
         label=r'$\widetilde\mathrm{IJ}$ Upper Bnd.',
            c='r',
            s=s)
plt.scatter(np.arange(sets.shape[0]),
         predErrsIJLowerBnd[sortInds],
         label=r'$\widetilde\mathrm{IJ}$ Lower Bnd.',
            c='k',
            s=s)
plt.legend(fontsize=legendFontsize)
plt.xlabel('Datapoint, n', fontsize=axlabelFontsize)
plt.ylabel('Prediction error of $y_n$', fontsize=axlabelFontsize)
plt.title('Exact CV vs bounds across datapoints n',
          fontsize=titleFontsize)
plt.gca().tick_params(axis='both',
                      which='major',
                      labelsize=tickFontsize)
plt.tight_layout()
#plt.savefig('C:YOUR_FILEPATH/singleTrialErrorBounds-Poisson-IJ.png', bbox='tight')
plt.show()


from IPython import embed;np.set_printoptions(linewidth=80);embed()
