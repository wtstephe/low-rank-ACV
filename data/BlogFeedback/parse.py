# Load in the blog data and save (a subset of) all pairwise features

import numpy as np


np.random.seed(12354)
X = np.genfromtxt('blogData_train.csv', delimiter=',')

Nsaved = 20000
Dsaved = 20000

Nsubset = np.random.choice(X.shape[0], replace=False, size=Nsaved)
Y = X[:,-1].astype(np.int32)
X = X[:,:-1]
X = X[Nsubset]
Y = Y[Nsubset]

subsetSize = Dsaved - X.shape[1] - 1
ids1 = np.random.choice(X.shape[1], replace=True, size=subsetSize)
ids2 = np.random.choice(X.shape[1], replace=True, size=subsetSize)
Xfull = np.zeros((X.shape[0], X.shape[1]+subsetSize+1))
Xfull[:,:X.shape[1]] = X
Xfull[:,-1] = 1.0


for ii in range(subsetSize):
	Xfull[:,X.shape[1]+ii] = X[:,ids1[ii]] * X[:,ids2[ii]]

print(Xfull.shape, Y.shape)
np.savetxt('Y-N=%d.txt' % Y.shape[0], Y)
np.savetxt('X-N=%d-D=%d.txt' % (Xfull.shape[0], Xfull.shape[1]), Xfull)



