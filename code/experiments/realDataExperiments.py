import autograd.numpy as np
import autograd
import pickle
import time

import sys
sys.path.append('./..')
import datasets
import expUtils
import Params
import models
import solvers

def computeNSIJAppx(model, K):
  import solvers
  model.compute_derivs()
  D1 = model.D1
  D2 = model.D2
  Qappx, estErr = solvers.compute_Q(model, K, estErr=False)
  thetaHatX = model.training_data.X.dot(model.params.get_free())

  NStilde = thetaHatX + D1 * Qappx / (1 - D2*Qappx)
  IJtilde = thetaHatX + D1 * Qappx
  return NStilde, IJtilde

def writeToDict(model, output, results, K, name):
  thetaHat = output['w0'][0]
  X = model.training_data.X
  lam = model.L2Lambda

  start = time.clock()
  NSTilde, IJTilde = computeNSIJAppx(model, 1000)
  AppxTime = time.clock() - start

  sets = np.array(output['held_out_sets']['exact']).squeeze()
  B = sets.shape[0]
  ExactCV = np.einsum('nd,nd->n',
                        X[sets],
                        output['exact_params'][0])

  start = time.clock()

  H = X.T.dot(np.diag(model.D2)).dot(X) + lam*np.eye(X.shape[1])
  ihvp = np.linalg.solve(H, X.T).T
  Qexact = np.einsum('nd,nd->n', ihvp, X)
  IJ = X.dot(model.params.get_free()) + model.D1 * Qexact
  NS = X.dot(model.params.get_free()) + model.D1 * Qexact / (1 - model.D2 * Qexact)
  NSIJExactTime = time.clock() - start

  timingExact = X.shape[0] * (output['timings']['exact'][0] / B)

  results = {}
  results[name] = {}
  results[name]['IJTilde'] = IJTilde[sets]
  results[name]['IJ'] = IJ[sets]
  results[name]['NSTilde'] = NSTilde[sets]
  results[name]['NS'] = NS[sets]
  results[name]['exactCV'] = ExactCV
  results[name]['timings'] = {}
  results[name]['timings']['ACVTilde'] = AppxTime
  results[name]['timings']['ACVExact'] = NSIJExactTime
  results[name]['timings']['exactCV'] = timingExact

  return results


############ 20k x 20k rcv1
rcv1Fname = '../output/error_scaling_experiments-None-RCV1DatasetGenerator-Xrank=-1-upTo=-1-lam=50000.000000-regularization=L2-k=1-B=20.pkl'
rcv1DataPath = '../../data/rcv1'
f = open(rcv1Fname, 'rb')
rcv1Dict = pickle.load(f)[20000][20001]
f.close()

lam = 100000 * 2
trainRCV1, _ = datasets.RCV1DatasetGenerator().get_dataset(filepath=rcv1DataPath)


params = Params.Param(trainRCV1.X.shape[1])
model = models.LogisticRegressionModel(trainRCV1,
                                       params,
                                       example_weights=np.ones(trainRCV1.X.shape[0]),
                                       test_data=None,
                                       regularization=None)


thetaHat = rcv1Dict['w0'][0]
model.regularization = lambda x: lam*np.linalg.norm(x)**2
model.params.set_free(thetaHat)
model.L2Lambda = lam
f = open('realDataResults.pkl', 'rb')
otherResults = pickle.load(f)
f.close()
results = writeToDict(model, rcv1Dict, {}, 1000, 'rcv1')
otherResults['rcv1'] = results['rcv1']
f = open('realDataResults.pkl', 'wb')
otherResults = pickle.dump(otherResults, f)
f.close()



######### Blog
f = open('../output/error_scaling_experiments-None-BlogFeedbackDatasetGenerator-Xrank=-1-upTo=-1-lam=50000.000000-regularization=L2-k=1-B=50.pkl', 'rb')
blogDict = pickle.load(f)[20000][18221]
f.close()
lam = 100000 * 2
trainBlog, _ = datasets.BlogFeedbackDatasetGenerator().get_dataset(filepath='../../data/BlogFeedback')
params = Params.Param(trainBlog.X.shape[1])
model = models.PoissonRegressionModel(trainBlog,
                                       params,
                                       example_weights=np.ones(trainBlog.X.shape[0]),
                                       test_data=None,
                                       regularization=None)
thetaHat = blogDict['w0'][0]
model.regularization = lambda x: lam*np.linalg.norm(x)**2
model.params.set_free(thetaHat)
model.L2Lambda = lam

f = open('realDataResults.pkl', 'rb')
otherResults = pickle.load(f)
f.close()
results = writeToDict(model, blogDict, {}, 1000, 'blog')
otherResults['blog'] = results['blog']
f = open('realDataResults.pkl', 'wb')
pickle.dump(otherResults, f)
f.close()

########## P53
f = open('../output/error_scaling_experiments-None-P53DatasetGenerator-Xrank=-1-upTo=-1-lam=20000.000000-regularization=L2-k=1-B=20.pkl', 'rb')
p53Dict = pickle.load(f)[8000][5409]
f.close()
lam = 20000 * 2
trainP53, _ = datasets.P53DatasetGenerator(filepath='../../data/p53_new_2012').get_dataset()
params = Params.Param(trainP53.X.shape[1])
model = models.LogisticRegressionModel(trainP53,
                                       params,
                                       example_weights=np.ones(trainP53.X.shape[0]),
                                       test_data=None,
                                       regularization=None)
thetaHat = p53Dict['w0'][0]
model.regularization = lambda x: lam*np.linalg.norm(x)**2
model.params.set_free(thetaHat)
model.L2Lambda = lam

f = open('realDataResults.pkl', 'rb')
otherResults = pickle.load(f)
f.close()
results = writeToDict(model, p53Dict, results, 500, 'p53')
otherResults['p53'] = results['p53']
f = open('realDataResults.pkl', 'wb')
otherResults = pickle.dump(otherResults, f)
f.close()





print('you sure you want to quit, this has all real data loaded?')
from IPython import embed;np.set_printoptions(linewidth=80);embed()
print('really sure?')
from IPython import embed;np.set_printoptions(linewidth=80);embed()
