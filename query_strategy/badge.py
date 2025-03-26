from .base_strategy import BaseStrategy
import torch.nn.functional as F
import pdb
from scipy import stats
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from sklearn.cluster import KMeans
from torch.autograd import Variable
from copy import deepcopy

def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        #print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll


class BadgeSampling(BaseStrategy):
    '''
    (ICLR2020)DEEP BATCH ACTIVE LEARNING BYDIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS
    '''
    def query(self, n):
        self.net = self.net[0]

        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        # probs, embeddings, _ = self.predict_probs_and_embed(unlabel_loader)
        # total_loader = self.build_total_loader()
        gradEmbedding = self.get_grad_embedding(unlabel_loader)
        chosen = init_centers(gradEmbedding, n)
        return unlabel_ind[chosen]

    def get_grad_embedding(self, loader):
        embDim = self.net.get_embedding_dim()
        self.net.eval()
        nLab = self.net.n_label
        embedding = np.zeros([len(loader.dataset), embDim * nLab])
        # loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            # shuffle=False, **self.args['loader_te_args'])
        idx = 0
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out = self.net(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idx][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idx][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                    idx += 1
            return embedding