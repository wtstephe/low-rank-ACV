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


nTrials = 1
Ks = np.arange(5,100,1)

datasetType = datasets.SyntheticPoissonDatasetGenerator
modelType = models.PoissonRegressionModel
N = 200
D = 150
lam = 5.0
Xrank = 50
lowRankNoise = 0.00
train, test = datasetType().get_dataset(Ntrain=N, D=D, Xrank=Xrank, lowRankNoise=lowRankNoise)


regularization = lambda x: lam / 2 * np.linalg.norm(x)**2
params = Params.Param(train.X.shape[1])
params.set_free(np.random.normal(size=(D+1)))
model = modelType(training_data=train,
				  params_init=params,
				  example_weights=np.ones(N),
				  regularization=regularization,)
model.L2Lambda = lam
model.fit(tol=1e-3)
model.compute_derivs()
model.compute_hessian()
H = model.hessian
Hinv = np.linalg.inv(H)
X = train.X
Z = np.sqrt(np.abs(model.D1))[:,np.newaxis] * X
thetaHat = model.params.get_free()
thetaHatX = X.dot(thetaHat)

HinvAppx = np.diag(1/np.diag(H))
preZZ = lambda x, diag=np.diag(H): (Z.T.dot(Z.dot(x))) / diag[:,np.newaxis]
preZZHinv = lambda x: Hinv.dot(Z.T.dot(Z).dot(x))

fullDiag = True
truncate = False
appxs = [('normal', None, r'$\Omega =$ random', fullDiag, truncate),
 		     ('Zvecs', Hinv, r'$\Omega = H^{-1} U_{:K}$', fullDiag, truncate),
 		     ('normal', preZZ, r'$\Omega \approx H^{-1} U_{:K}$', fullDiag, truncate),
         ('normal', None, r'Bar, $\Omega =$ random', fullDiag, True),
 		     ('Zvecs', Hinv, r'Bar, $\Omega = H^{-1} U_{:K}$', fullDiag, True),
 		     ('normal', preZZ, r'Bar, $\Omega \approx H^{-1} U_{:K}$', fullDiag, True),
 	]

actualErrs = np.empty((nTrials, len(appxs), Ks.shape[0], N))
estErrs = np.empty((nTrials, len(appxs), Ks.shape[0], N))
tightEstErrs = np.empty((nTrials, len(appxs), Ks.shape[0], N))
timingsFull = np.empty((nTrials))
timingsAppx = np.empty((nTrials, len(appxs), Ks.shape[0]))

timingsExact = np.empty((nTrials,Ks.shape[0]))
actualErrsExact = np.empty((nTrials,Ks.shape[0],N))


for trial in range(nTrials):
  start = time.time()
  ihvp = np.linalg.solve(H, X.T).T
  Q = np.einsum('nd,nd->n', X, ihvp)
  IJ = thetaHatX + Q*model.D1
  timingsFull[trial] = time.time() - start

  print('---', trial, '----')
  for kk, K in enumerate(Ks):
    print(K)
    for aa, appx in enumerate(appxs):
      actualErrs[trial,aa,kk], estErrs[trial,aa,kk], appxTime = \
        expUtils.testLowRankAppx(K, K, appx[0], appx[1],
                                 model, H,
                                 thetaHatX=thetaHatX,
                                 compareTo=IJ,
                                 fullDiagonal=appx[3],
                                 truncate=appx[4],
                                 IJ=IJ)
      timingsAppx[trial,aa,kk] = 0.0
      #Qappx, estErr = solvers.compute_Q(model, K, 
      #                                  truncate=appx[4], estErr=True)
      #actualErrs[trial,aa,kk] = np.abs(model.D1 * (Qappx - Q))
      #estErrs[trial,aa,kk] = np.abs(model.D1 * estErr)
                                        


for aa in range(3):
	E = 100 * actualErrs[:,aa,:,:] / np.abs(IJ)[np.newaxis,np.newaxis,:]
	plt.scatter(Ks,
				E.mean(axis=(0,2)),
				label = '%s' % appxs[aa][2])

plt.legend(fontsize=16)
plt.xlabel('Approximation rank, K', fontsize=16)
plt.ylabel(r'Mean % error in approximation of IJ', fontsize=16)
plt.title(r'Choice of $\Omega$', fontsize=16)
plt.tight_layout()
plt.savefig('C:YOUR_FILEPATH/syntheticPoisson-IJ-preconditioningOptions.png', bbox='tight')


plt.figure()
aa = 2
plt.scatter(Ks,
				    actualErrs[:,aa,:,:].mean(axis=(0,2)),
            label = r'$x_n^T \widetilde H^{-1} x_n$ (actual error)')
plt.scatter(Ks,
				    estErrs[:,aa,:,:].mean(axis=(0,2)),
            label = r'$x_n^T \widetilde H^{-1} x_n$ (upper bound)')
aa = 5
plt.scatter(Ks,
				    actualErrs[:,aa,:,:].mean(axis=(0,2)),
				    label = r'$\widetilde Q_n$ (actual error)')
plt.scatter(Ks,
				    estErrs[:,aa,:,:].mean(axis=(0,2)),
				    label = r'$\widetilde Q_n$ (upper bound)')

plt.legend(fontsize=16)
plt.xlabel('Approximation rank, K', fontsize=16)
plt.ylabel(r'Mean error in approximation of IJ', fontsize=16)
plt.title(r'Use of Proposition 4', fontsize=16)
plt.tight_layout()
plt.savefig('C:YOUR_FILEPATH/syntheticPoisson-IJ-truncationYesNo.png', bbox='tight')
plt.show()


from IPython import embed; embed()
