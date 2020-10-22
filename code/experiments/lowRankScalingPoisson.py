import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

sys.path.append('./..')
import expUtils

legendSize = 14
labelSize = 18
tickFontsize = 13


Ns = np.around(np.logspace(np.log10(400),
                                np.log10(5000),
                                8), 0)
Ns = Ns.astype(np.int32)
plt.figure(num=None, figsize=(6,4.30))


# Exact low dimension
# python cmdExperiments.py --dataset SyntheticPoissonDatasetGenerator --model PoissonRegressionModel --regularization L2 --lambdaScaling const --lambdaCoeff 10.0 --upTo 30 --minNtrain 400 --maxNtrain 5000 --numNtrains 8 --NtoD const --D 40 --Xrank 40 --nCores 0
#fname = '../output/error_scaling_experiments-None-SyntheticPoissonDatasetGenerator-Xrank=40-lowRankNoise=0.000000-solveRank=-1-upTo=30--lam=10.000000-regularization=L2-k=1-B=30.pkl'
fname = '../output/error_scaling_experiments-None-SyntheticPoissonDatasetGenerator-Xrank=40-lowRankNoise=0.000000-solveRank=-1-upTo=30--lam=100.000000-regularization=L2-k=1-B=30.pkl'
Ds = [41,]
NsFound, DsFound, errorsIJ, errorsExact, errorsNS, errorsTest, errorsTrain, medianDeltas, medianParamErrorsIJ, medianParamErrorsNS, paramsExact, paramsIJ, w0s = expUtils.readInResultsDict(fname, Ns, Ds)
expUtils.scatterAndRegress(np.log(NsFound),
                           np.log(medianParamErrorsIJ),
                           legendLabel='D = 40',
                           c='r',
                           linewidth=4,
                           marker='s',
                           markerSize=80)



#ALR
fname = '../output/error_scaling_experiments-None-SyntheticPoissonDatasetGenerator-Xrank=40-lowRankNoise=0.000000-solveRank=-1-upTo=30--lam=100.000000-regularization=L2-k=1-B=30.pkl'
#fname = '../output/error_scaling_experiments-None-SyntheticPoissonDatasetGenerator-Xrank=40-lowRankNoise=0.000000-solveRank=-1-upTo=30--lam=10.000000-regularization=L2-k=1-B=30.pkl'
Ds = np.ceil(Ns / 10).astype(np.int32) + 1
NsFound, DsFound, errorsIJ, errorsExact, errorsNS, errorsTest, errorsTrain, medianDeltas, medianParamErrorsIJ, medianParamErrorsNS, paramsExact, paramsIJ, w0s = expUtils.readInResultsDict(fname, Ns, Ds)
expUtils.scatterAndRegress(np.log(NsFound),
                           np.log(medianParamErrorsIJ),
                           legendLabel='D = N/10, rank = 40',
                           c='k',
                           linestyle='--',
                           linewidth=4,
                           marker='o')

# Full rank
fname = '../output/error_scaling_experiments-None-SyntheticPoissonDatasetGenerator-Xrank=-1-upTo=30-lam=10.000000-regularization=L2-k=1-B=30.pkl'
NsFound, DsFound, errorsIJ, errorsExact, errorsNS, errorsTest, errorsTrain, medianDeltas, medianParamErrorsIJ, medianParamErrorsNS, paramsExact, paramsIJ, w0s = expUtils.readInResultsDict(fname, Ns, Ds)
expUtils.scatterAndRegress(np.log(NsFound),
                           np.log(medianParamErrorsIJ),
                           legendLabel='D = N/10',
                           c='b',
                           linestyle='-',
                           linewidth=4,
                           marker='o')


print(NsFound)

plt.legend(fontsize=legendSize)
plt.ylabel('Log average IJ error', fontsize=labelSize)
plt.xlabel('Log(N)', fontsize=labelSize)
plt.title('Error vs. dataset size $N$', fontsize=labelSize)
plt.gca().tick_params(axis='both',
                      which='major',
                      labelsize=tickFontsize)
plt.tight_layout()
plt.savefig('C:YOUR_FILEPATH/errorScalingPoisson_fixed.png')
plt.show()






