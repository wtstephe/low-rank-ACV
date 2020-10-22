'''Parses the file rcv1_train.binary taken from:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
'''

import numpy as np

Nsaved = 20000
Dsaved = 20000
svdRatio = .5

f = open('rcv1_train.binary', 'r')

activeLocs = {}
values = {}
N = 0

for line in f.readlines():
  pieces = line.split(' ')

  for loc in pieces[1:]:
    idx, value = loc.split(':')
    idx = int(idx)
    if idx in activeLocs:
      activeLocs[idx] += 1
    else:
      activeLocs[idx] = 1

    if idx not in values:
      values[idx] = []
    values[idx].append(float(value))
  N += 1

f.close()

items = list(activeLocs.items())
ids = np.array([item[0] for item in items])
counts = np.array([item[1] for item in items])
sortinds = np.argsort(counts)
ids = ids[sortinds]
counts = counts[sortinds]

goodIds = ids[-Dsaved:]
Y = []
X = np.zeros((N,goodIds.shape[0]))
n = 0
f = open('rcv1_train.binary', 'r')

for line in f.readlines():
  pieces = line.split(' ')
  Y.append(int(pieces[0]))
  for loc in pieces[1:]:
    idx, value = loc.split(':')
    idx = int(idx)
    if idx not in goodIds:
      continue
    else:
      newIdx = np.where(idx == goodIds)[0][0]
      X[n,newIdx] = float(value)
  n += 1

f.close()
Y = np.array(Y)
np.random.seed(1234)
subsetInds = np.random.choice(X.shape[0], size=Nsaved, replace=False)
Ysave = Y[subsetInds]
Xsave = X[subsetInds,:]
print(Xsave.shape, Ysave.shape)
np.savetxt('X-N=%d-D=%d.txt' % (Nsaved, Dsaved), Xsave)
np.savetxt('Y-N=%d-D=%d.txt' % (Nsaved, Dsaved), Ysave)

from IPython import embed; np.set_printoptions(linewidth=150); embed()

# Make non-low-rank version
U, S, V = np.linalg.svd(Xsave)
lowInds = np.where(S < svdRatio * S[0])[0]
S2 = S.copy()
S2[lowInds] = (svdRatio * S[0] + \
               np.abs(np.random.normal(size=lowInds.shape[0],
                                       scale=np.sqrt((1-svdRatio)*S[0]))))
Xsave2 = U[:,:S.shape[0]].dot(np.diag(S2)).dot(V)
np.savetxt('fullRank-X-N=%d-D=%d.txt' % (Nsaved, Dsaved), Xsave2)
np.savetxt('Y-N=%d-D=%d.txt' % (Nsaved, Dsaved), Ysave)



                        
  
