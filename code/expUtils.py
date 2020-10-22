'''
utils for loading / plotting data from experiments. Many very specific
  functions -- it's probably easiest to just write your own!
'''

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import retrainingPlans
import solvers

def scatterAndRegress(X, Y,
                      title=None,
                      show=False,
                      c=None,
                      linestyle='-',
                      legendLabel=None,
                      returnHandle=False,
                      linewidth=None,
                      marker='.',
                      markerSize=None):
  p = np.polyfit(X, Y, 1)
  plt.scatter(X, Y, c=c, marker=marker, s=markerSize)
  if title is not None:
    plt.title(title)

  t = np.linspace(X.min(), X.max(), 10)
  label = '%s,  slope=%.2f' % (legendLabel, p[0])
  handle, = plt.plot(t, p[0]*t + p[1],
                     c=c,
                     linestyle=linestyle,
                     label=label,
                     linewidth=linewidth)
  if show:
    plt.legend()
    plt.show()

  if returnHandle:
    return handle, label

def loadResultsDict(outputPath):
  if os.path.exists(outputPath):
    f = open(outputPath, 'rb')
    results = pickle.load(f)
    f.close()
  else:
    results = {}
  return results

def readInResultsDict(filepath, Ns, Ds,
                      returnW0=False,
                      loadNS=True,
                      returnTimes=True):
  '''
  NtoD takes in Ntrain and outputs D (e.g. NtoD(Ntrain) = int(Ntrain/10))
  '''
  f = open(filepath, 'rb')
  results = pickle.load(f)
  f.close()

  NsFound = []
  DsFound = []
  errorsIJ = []
  errorsExact = []
  errorsTest = []
  errorsTrain = []
  medianParamErrorsIJ = []
  medianParamErrorsNS = []
  medianDeltas = []
  avgHessNorms = []
  numSupportChanges = []
  supportSize = []
  paramsExact = []
  paramsIJ = []
  w0s = []

  timesIJ = []
  timesExact = []
  timesNS = []

  if loadNS:
    errorsNS = []

  for Ntrain in Ns:
    for D in Ds:
      if Ntrain not in results.keys() or D not in results[Ntrain].keys():
        continue
      NsFound.append(Ntrain)
      DsFound.append(D)
      errorsIJ.append(results[Ntrain][D]['IJ_CV'])
      errorsExact.append(results[Ntrain][D]['exact_error'])
      errorsTest.append(results[Ntrain][D]['test_error'])
      errorsTrain.append(results[Ntrain][D]['train_error'])

      timesIJ.append(results[Ntrain][D]['timings']['IJ'])
      timesNS.append(results[Ntrain][D]['timings']['NS'])
      timesExact.append(results[Ntrain][D]['timings']['exact'])

      param_exact = np.array(results[Ntrain][D]['exact_params'])
      #paramsIJ = np.array(results[Ntrain][D]['IJ'])
      w0 = np.array(results[Ntrain][D]['w0'])
      w0s.append(w0)
      supportSize.append((w0 != 0).sum())
      paramsExact.append(param_exact)
      paramsIJ.append(results[Ntrain][D]['IJ'])
      
      if param_exact.shape[1] > 1:
        deltas = np.linalg.norm(param_exact - w0[:,np.newaxis,:], axis=2)
        paramErrors = np.linalg.norm(param_exact - paramsIJ[-1], axis=2)
        paramErrors[np.where(np.isinf(paramErrors))] = 0.0
        paramErrors[np.where(np.isnan(paramErrors))] = 0.0
        medianDeltas.append(np.median(deltas))
        medianParamErrorsIJ.append(np.mean(paramErrors))
        supportChanged = np.logical_xor(param_exact != 0,
                                        (w0 != 0)[:,np.newaxis,:])
        numSupportChanges.append(np.median(supportChanged.sum(axis=(1,2))))
        
      if loadNS:
        paramsNS = np.array(results[Ntrain][D]['NS'])
        errorsNS.append(results[Ntrain][D]['NS_CV'])
        if param_exact.shape[1] > 1:
          NSErrors = np.linalg.norm(param_exact - paramsNS, axis=2)
          medianParamErrorsNS.append(np.mean(NSErrors))



  return np.array(NsFound), np.array(DsFound), np.array(errorsIJ), np.array(errorsExact), np.array(errorsNS), np.array(errorsTest), np.array(errorsTrain), np.array(medianDeltas), np.array(medianParamErrorsIJ), np.array(medianParamErrorsNS), paramsExact, paramsIJ, w0s 
  if not returnW0:
    if loadNS:
      return np.array(NsFound), np.array(DsFound), np.array(errorsIJ), np.array(errorsExact), np.array(errorsNS), np.array(errorsTest), np.array(errorsTrain), np.array(medianDeltas), np.array(medianParamErrorsIJ) ,np.array(medianParamErrorsNS), np.array(avgHessNorms), paramsExact, np.array(timesExact), np.array(timesIJ), np.array(timesNS)
    else:
      return np.array(NsFound), np.array(DsFound), np.array(errorsIJ), np.array(errorsExact), np.array(errorsTest), np.array(errorsTrain), np.array(medianDeltas), np.array(medianParamErrorsIJ), np.array(avgHessNorms), paramsExact
  else:
    if not loadNS:
      return np.array(NsFound), np.array(DsFound), np.array(errorsIJ), np.array(errorsExact), np.array(errorsTest), np.array(errorsTrain), np.array(medianDeltas), np.array(medianParamErrorsIJ), np.array(avgHessNorms), paramsExact, paramsIJ, w0s
    else:
      return np.array(NsFound), np.array(DsFound), np.array(errorsIJ), np.array(errorsExact), np.array(errorsNS), np.array(errorsTest), np.array(errorsTrain), np.array(medianDeltas), np.array(medianParamErrorsIJ), np.array(avgHessNorms), paramsExact, paramsIJ, w0s
    

def runCVAndLogResults(results, model, train, test,
                       runExactCV=True, runNS=False, **kwargs):
  '''
  Block o' code that used to sit in errorScalingExps.py
  '''

  Ntrain = train.X.shape[0]
  D = train.X.shape[1]
  w0 = model.params.get_free().copy()
  print('heres my w0', w0)

  if 'save_key' in kwargs:
    Ntrain = kwargs[kwargs['save_key']]
  if Ntrain not in results.keys():
    results[Ntrain] = {}
  if D not in results[Ntrain].keys():
    results[Ntrain][D] = {}
  if 'IJ_CV' not in results[Ntrain][D].keys():
    results[Ntrain][D]['IJ_CV'] = []
    results[Ntrain][D]['IJ'] = []
    results[Ntrain][D]['NS_CV'] = []
    results[Ntrain][D]['NS'] = []
    results[Ntrain][D]['exact_error'] = []
    results[Ntrain][D]['exact_params'] = []
    results[Ntrain][D]['w0'] = []
    results[Ntrain][D]['test_error'] = []
    results[Ntrain][D]['train_error'] = []

    results[Ntrain][D]['timings'] = {}
    results[Ntrain][D]['timings']['exact'] = []
    results[Ntrain][D]['timings']['IJ'] = []
    results[Ntrain][D]['timings']['NS'] = []
    
    results[Ntrain][D]['held_out_sets'] = {}
  if 'NS_CV' not in results[Ntrain][D]:
    results[Ntrain][D]['NS_CV'] = []


  if kwargs['nCores'] > 0:
    cv_function = retrainingPlans.leave_k_out_cv_parallel
  else:
    cv_function = retrainingPlans.leave_k_out_cv

  start = time.time()

  error_IJ, paramsIJ, held_out_sets_IJ = cv_function(model,
                                                     method='IJ',
                                                     **kwargs)
  IJ_time = time.time() - start
  print('finished IJ appx')

  if runExactCV:
    print('doing exact cv')
    start = time.time()
    error_exact, param_exact, held_out_sets_exact = cv_function(model,
                                                                method='exact',
                                                                **kwargs)
    exact_time = time.time() - start
  else:
    error_exact = []
    param_exact = []
    exact_time = 0.0
  print('finished exact CV')

  if runNS:
    start = time.time()
    print('starting NS appx')
    error_NS, paramsNS, held_out_sets_NS = cv_function(model,
                                                       method='NS',
                                                       **kwargs)
    NS_time = time.time() - start
  else:
    error_NS = []
    paramsNS = []
    NS_time = 0.0
  print('finished NS appx')

  results[Ntrain][D]['IJ_CV'].append(error_IJ)
  results[Ntrain][D]['NS_CV'].append(error_NS)
  results[Ntrain][D]['IJ'].append(paramsIJ)
  results[Ntrain][D]['NS'].append(paramsNS)
  results[Ntrain][D]['exact_error'].append(error_exact)
  results[Ntrain][D]['exact_params'].append(param_exact)
  results[Ntrain][D]['w0'].append(w0)

  results[Ntrain][D]['timings']['exact'].append(exact_time)
  results[Ntrain][D]['timings']['IJ'].append(IJ_time)
  results[Ntrain][D]['timings']['NS'].append(NS_time)
  results[Ntrain][D]['held_out_sets']['NS'] = held_out_sets_NS
  results[Ntrain][D]['held_out_sets']['IJ'] = held_out_sets_IJ
  results[Ntrain][D]['held_out_sets']['exact'] = held_out_sets_exact

  if test is not None:
    results[Ntrain][D]['test_error'].append(model.get_error(test))
  results[Ntrain][D]['train_error'].append(model.get_error(train))


def makeName(datasetName,
             Xrank,
             upTo,
             L1str,
             lambdaCoeff,
             regularization,
             k,
             B,
             solveRank=None,
             alpha=None,
             lowRankNoise=None,
             tag=''):
  
  if regularization == 'smoothedL1':
    return '%s-%s-Xrank=%d-upTo=%d%s-lam=%f-regularization=%s-alpha=%f-k=%d-B=%d' %\
      (tag, datasetName, Xrank, upTo, L1str, lambdaCoeff, regularization, alpha,
       k, B)
  if Xrank > -1:
    return '%s-%s-Xrank=%d-lowRankNoise=%f-solveRank=%s-upTo=%d-%s-lam=%f-regularization=%s-k=%d-B=%d' %\
      (tag, datasetName, Xrank, lowRankNoise, solveRank, upTo, L1str, lambdaCoeff, regularization,
       k, B)
  else:
    return '%s-%s-Xrank=%d-upTo=%d%s-lam=%f-regularization=%s-k=%d-B=%d' %\
      (tag, datasetName, Xrank, upTo, L1str, lambdaCoeff, regularization,
       k, B)
    
    


def plotMultiple(datasetName=None,
                 Xrank=-1, upTo=-1, L1str='', lambdaCoeff=1.0, regularization='L2',
                 Xscaling='None', tol=1e-10, k=1, B=30, title=None,
                 yaxis='paramError', name=None, show=True):

  if name is None:
    name = makeName(datasetName, Xrank, upTo, L1str, lambdaCoeff, regularization,
                    Xscaling, tol, k, B)
    
  filename = 'output/error_scaling_experiments-%s.pkl' % name
  print(filename)

  NtoD = lambda N: 3
  (Ns, Ds,
   errorsIJ, errorsExact, errorsTest, errorsTrain,
   medianDeltas, medianParamErrorsIJ,
   avgHessNorms, paramsExact) = \
                                readInResultsDict(filename, NtoD)
  plt.rc('text', usetex=True)
  if yaxis == 'paramError':
    yData = medianParamErrorsIJ
  elif yaxis == 'CVError':
    yData = errorsExact - errorsIJ
  scatterAndRegress(np.log(Ns),
                    np.log(np.abs(yData)),
                    c='r',
                    show=False,
                    legendLabel='D=2')
  
  NtoD = lambda N: int(np.ceil(N/10)) + 1
  (Ns, Ds,
   errorsIJ, errorsExact, errorsTest, errorsTrain,
   medianDeltas, medianParamErrorsIJ,
   avgHessNorms, paramsExact) = \
                                readInResultsDict(filename, NtoD)
  if yaxis == 'paramError':
    yData = medianParamErrorsIJ
  elif yaxis == 'CVError':
    yData = errorsExact - errorsIJ
  scatterAndRegress(np.log(Ns),
                    np.log(np.abs(yData)),
                    c='b',
                    show=False,
                    legendLabel='D=N/10')
  if title is not None:
    plt.title(title, fontsize=18)
  

  if yaxis == 'paramError':
    #plt.ylabel(r'$\log ( E_w \| \theta^w - \tilde\theta^w \|_2 )$',
    #           fontsize=24)
    plt.ylabel('log Approximation Error',
               fontsize=20)

  if yaxis == 'CVError':
    plt.ylabel(r'$\log ( \mathrm{LOO} - \mathrm{ALOO} )$',
               fontsize=24)
    
  plt.xlabel(r'$\log N$', fontsize=24)
  plt.legend(fontsize=18)

  if show:
    plt.show()


  
def plotSupportRecovery(idx,
                 Xrank=-1, upTo=-1, L1str='', lambdaCoeff=1.0, regularization='L2',
                 Xscaling='None', tol=1e-10, k=1, B=30, title=None,
                        yaxis='paramError', name=None, show=True):

  if name is None:
    name = makeName(datasetName, Xrank, upTo, L1str, lambdaCoeff, regularization,
                    Xscaling, tol, k, B)
    
  filename = 'output/error_scaling_experiments-%s.pkl' % name
  print(filename)
  
  NtoD = lambda N: int(np.ceil(N/10)) + 1
  #NtoD = lambda N: int(2*N) + 1
  (Ns, Ds,
   errorsIJ, errorsExact, errorsTest, errorsTrain,
   medianDeltas, medianParamErrorsIJ,
   avgHessNorms, paramsExact) = \
                                readInResultsDict(filename, NtoD)
  ax = plt.gca()
  ax.imshow(paramsExact[idx][0] != 0, interpolation='nearest',
            vmin=0, vmax=1, aspect='auto')



def computeAverageSupportChange(w0, exactParams):
  '''
  w0 is a D length array, exactParams is a BxD array of leave-1-out params.
  Each decrease in support size gives a -1, each increase gives a +1.
  Returns the sum of all these changes divided by B
  '''
  count = 0
  B = exactParams.shape[0]
  S = np.where(w0 != 0)[0]
  for b in range(B):
    Sb = np.where(exactParams[b] != 0)[0]
    suppIncrease = np.setdiff1d(Sb, S)
    suppDecrease = np.setdiff1d(S, Sb)
    count += suppIncrease.shape[0] - suppDecrease.shape[0]
  return count / B
  
def loadManyLambda(lambdas, Ns, Ds,  doNS=False, **kwargs):

  Ds = Ds.copy() + 1 # Account for intercept

  tests = []
  trains = []
  IJCVs = []
  exactCvs = []
  paramsExacts = []
  paramsIJs = []
  NSCVs = []
  w0ss = []
  NsFounds = []
  DsFounds = []
  
  for lam in lambdas:
    name = makeName(lambdaCoeff=lam, **kwargs)
    filename = 'output/error_scaling_experiments-%s.pkl' % name
    if not os.path.exists(filename):
      print('fail, ', filename)
      continue

    (NsFound, DsFound,
     errorsIJ, errorsExact, errorsNS, errorsTest, errorsTrain,
     medianDeltas, medianParamErrorsIJ,
     avgHessNorms, paramsExact, paramsIJ, w0s) = \
                                  readInResultsDict(filename, Ns, Ds,
                                                    returnW0=True,
                                                    loadNS=True)


    tests.append(errorsTest)
    trains.append(errorsTrain)
    IJCVs.append(errorsIJ)
    exactCvs.append(errorsExact)
    w0ss.append(w0s)
    DsFounds.append(DsFound)
    NsFounds.append(NsFound)
    if doNS:
      NSCVs.append(errorsNS)


  # Sometimes different files will have different numbers of trials saved. E.g.,
  #  the first lambda will only have 1 trial, the second 4, the third only 3, etc.
  #  The following chops off these extra trials so all have the same number.
  numTrials = np.unique(np.array([train.shape[1] for train in trains]))
  if numTrials.shape[0] > 1:
    minNumTrials = numTrials.min()
    trains = [train[:,:minNumTrials] for train in trains]
    tests = [test[:,:minNumTrials] for test in tests]
    exactCvs = [exact[:,:minNumTrials] for exact in exactCvs]
    IJCVs = [appx[:,:minNumTrials] for appx in IJCVs]
    NSCVs = [appx[:,:minNumTrials] for appx in NSCVs]

  return np.array(trains), np.array(tests), np.array(exactCvs), np.array(IJCVs), np.array(NSCVs), NsFounds, DsFounds
    

def makeLambdaSelectionPlot(lams, Ns, Ds,
                            trialNum=0,
                            doNS=False,
                            doExactCV=True,
                            **kwargs):
  trains, tests, exactCvs, IJCVs, NSCVs, NsFound, DsFound = \
                                      loadManyLambda(lams, Ns, Ds,
                                                     doNS=doNS,
                                                     **kwargs)
  if trains.ndim > 2:
    trains = trains.squeeze(axis=1)
    tests = tests.squeeze(axis=1)
    exactCvs = exactCvs.squeeze(axis=1)
    IJCVs = IJCVs.squeeze(axis=1)
    if NSCVs.ndim > 2:
      NSCVs = NSCVs.squeeze(axis=1)

  trains = trains[:,trialNum]
  tests = tests[:,trialNum]
  IJCVs = IJCVs[:,trialNum]
  exactCvs = exactCvs[:,trialNum]
  lams = lams[:len(trains)]


  plt.plot(lams, trains, c='y', label='Train Loss')
  plt.plot(lams, tests, c='k', linestyle='--', label='Test Loss')
  plt.plot(lams, IJCVs, c='b', label='Appx (IJ)')

  if doExactCV:
    plt.plot(lams, exactCvs, c='r', label='Exact CV')
    print('Exact CV minimizing lambda:', lams[np.argmin(exactCvs)])
  if doNS:
    NSCVs = NSCVs[:,trialNum]
    plt.plot(lams, NSCVs.squeeze(),
             c='b', linestyle=':', label='Appx (NS)', linewidth=3.0)
  plt.gca().set_xlabel('$\lambda$', fontsize=24)
  plt.gca().set_ylabel('Error', fontsize=24)
  plt.legend(fontsize=14, loc=4)
  
  np.savetxt('lambdaCalibrationData-N=300-D=150.txt',
             np.stack([lams, trains, tests, IJCVs, NSCVs, exactCvs], axis=0))

  print('Appx_IJ CV minimizing lambda:', lams[np.argmin(IJCVs)])
  print('Appx IJ deriv:', (IJCVs[1] - IJCVs[0])/(lams[1]-lams[0]))


def convertOldSaveFormat(filename):
  f = open(filename, 'rb')
  results = pickle.load(f)
  f.close()

  newResults = {}
  for N in results.keys():
    newResults[N] = {}
    for D in results[N].keys():
      newResults[N][D] = {}
      
      newResults[N][D]['IJ_CV'] = results[N][D]['appx_error']
      newResults[N][D]['IJ'] = results[N][D]['appx_params']
      newResults[N][D]['NS_CV'] = results[N][D]['appx_slow_error']
      newResults[N][D]['NS'] = results[N][D]['appx_slow_params']

      newResults[N][D]['w0'] = results[N][D]['w0']
      newResults[N][D]['exact_error'] = results[N][D]['exact_error']
      newResults[N][D]['exact_params'] = results[N][D]['exact_params']
      newResults[N][D]['test_error'] = results[N][D]['test_error']
      newResults[N][D]['train_error'] = results[N][D]['train_error']
      
      newResults[N][D] = results[N][D].copy()
      newResults[N][D]['timings'] = {'IJ':results[N][D]['timings']['appx'],
                                     'NS':results[N][D]['timings']['appx_slow'],
                                     'exact':results[N][D]['timings']['exact']}

  f = open(filename+'.bak', 'wb')
  pickle.dump(results, f)
  f.close()

  f = open(filename, 'wb')
  pickle.dump(newResults, f)
  f.close()
  
      
def testLowRankAppx(
    Q, 
    K, 
    omegaForm, 
    preconditioner, 
    model, 
    H,
    thetaHatX,
    compareTo, 
    fullDiagonal=True,
    truncate=False,
    IJ=None):

  D1 = model.D1
  D2 = model.D2
  lam = model.L2Lambda
  X = model.training_data.X
  D = X.shape[1] - 1

  start = time.time()
  sketch, omega = solvers.tropp_sketch(X, 
                                       model.D2, 
                                       0.0, 
                                       Q,
                                       preconditioner=preconditioner,
                                       omegaForm=omegaForm,
                                       D1=model.D1)
  vecsAppx, valsAppx = solvers.tropp_summarize_sketch(sketch, 
                                                      omega, 
                                                      K,)
  if fullDiagonal:
    #Happx = (vecsAppx.dot(np.diag(valsAppx)).dot(vecsAppx.T) 
    #  + lam*np.eye(D+1))
    #ihvpAppx = np.linalg.solve(Happx, X.T).T
    #vappxX = vecsAppx.T.dot(X.T)
    #XPerp = (X.T - vecsAppx.dot(vappxX)).T
    #ihvpAppx = vecsAppx.dot(np.diag(1/(valsAppx+lam)).dot(vappxX)).T
    #ihvpAppx += 1/lam * XPerp
    vappxX = vecsAppx.T.dot(X.T).T
    Xnorms = np.linalg.norm(X, axis=1)**2
    diagInv = valsAppx / (lam + valsAppx)    
    Qappx = Xnorms / lam - np.einsum('nk,nk->n', vappxX * diagInv[np.newaxis,:], vappxX) / lam    
  else:
    ihvpAppx = vecsAppx.dot(np.diag(1/(lam+valsAppx))).dot(vecsAppx.T.dot(X.T)).T
  

  #Qappx = np.einsum('nd,nd->n', X, ihvpAppx)
  if truncate:
    # "Partial truncation"
    #ihvpAppx = vecsAppx.dot(np.diag(1/(lam+valsAppx))).dot(vecsAppx.T.dot(X.T)).T
    #bnorms = np.linalg.norm(X, axis=1)**2 * D2
    #bnormsProj = D2 * (np.linalg.norm(X, axis=1)**2 - 
    #                    np.linalg.norm(vecsAppx.T.dot(X.T).T, axis=1)**2)
    #Qappx = np.einsum('nd,nd->n', X, ihvpAppx) + bnormsProj / (bnormsProj + lam) / D2
    #Qappx = np.minimum(Qappx, bnorms / (model.L2Lambda + bnorms) / D2)

    # "Full truncation"
    bnorms = np.linalg.norm(X, axis=1)**2 * D2
    QappxTrunc = np.minimum(Qappx, bnorms / (model.L2Lambda + bnorms) / D2)

    IJappx = thetaHatX + Qappx * model.D1
    IJappxTrunc = thetaHatX + QappxTrunc * model.D1
    if np.any(np.abs(IJappx - compareTo) < np.abs(IJappxTrunc - compareTo)):
      from IPython import embed;np.set_printoptions(linewidth=80);embed()
    Qappx = QappxTrunc

  IJappx = thetaHatX + Qappx * model.D1
  estErr = solvers.tropp_error_bound(X,
                                     vecsAppx,
                                     omega,
                                     K,
                                     model.L2Lambda,
                                     H=H,
                                     truncate=truncate,
                                     D2=model.D2)
  estErr *= np.abs(D1)
  appxTime = time.time() - start


  return np.abs(compareTo - IJappx), estErr, appxTime


def computeNSIJAppx(model, K):
  model.compute_derivs()
  D1 = model.D1
  D2 = model.D2
  Qappx, estErr = solvers.compute_Q(model, K)
  thetaHatX = model.training_data.X.dot(model.params.get_free())

  NStilde = thetaHatX + D1 * Qappx / (1 - D2*Qappx)
  IJtilde = thetaHatX + D1 * Qappx
  return NStilde, IJtilde
  
