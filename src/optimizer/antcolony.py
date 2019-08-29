import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.optimizer.optimizer import Optimizer
from src.optimizer.pareto import Pareto


### Ant colony optimization ###
class ACO(Optimizer):
    def __init__(self, fct, xbounds, ybounds, cbounds=[], nparticles=10, q=0.1, eps=0.1, colonySize=10, archiveSize=10, **kwargs):
        super().__init__(fct, xbounds, ybounds, cbounds=cbounds, **kwargs)

        self.colonySize = colonySize
        self.archiveSize = archiveSize
        self.xbest = np.zeros((0, self.xdim))
        self.ybest = np.zeros((0, self.ydim))
        self.pbest = np.zeros((0, 1))
        self.q = q
        self.eps = eps

    ### Iterate ###
    def iterate(self, itermax):

        x = Optimizer._dimensionalize(np.random.rand(self.colonySize, self.xdim), self.xlb, self.xub )
        
        ### Start iterating ###
        for n in range(itermax):
            self.currentIteration += 1
            print(" Episode : {}".format(self.currentIteration))

            ### Evaluate it ###
            y, c, p = self.evaluate(x)

            ### Append value to so far best seen designs ###
            xNorm = Optimizer._nondimensionalize(np.vstack((x, self.xbest)), self.xlb, self.xub )
            yNorm = Optimizer._nondimensionalize(np.vstack((y, self.ybest)), self.ylb, self.yub )
            p = np.vstack((p, self.pbest))

            ### Pareto Ranks ###
            ranks = Pareto.computeParetoRanks(yNorm+p)
            idxs = np.argsort(ranks)

            ### Sort according to ranks ###
            xNorm, yNorm, ranks, p = xNorm[idxs, :], yNorm[idxs, :], ranks[idxs], p[idxs]

            self.xbest = Optimizer._dimensionalize(xNorm[:self.archiveSize,:], self.xlb, self.xub )
            self.ybest = Optimizer._dimensionalize(yNorm[:self.archiveSize,:], self.ylb, self.yub )
            self.pbest = p[:self.archiveSize,:]
            ranks = ranks[:self.archiveSize]

            ### Calculate weights ###
            omega = 1/(self.q*self.archiveSize*np.sqrt(2*np.pi))*np.exp((-ranks**2+2*ranks-1.0)/(2*self.q**2*self.archiveSize**2))
            p = omega/np.sum(omega)

            ### Build new solution ###
            x = np.zeros((self.colonySize, self.xdim))

            for l in range(self.colonySize):
                for i in range(self.xdim):
                    j = np.random.choice(np.arange(self.archiveSize), p=p)
                    Sji = self.xbest[j,:][i]
                    sigma = self.eps*np.sum(np.abs(self.xbest[j,i]-self.xbest[:,i]))

                    ### Sample from normal distribution ###
                    x[l,i] = Sji + sigma*np.random.normal()
                    if x[l,i]>self.xub[i]:
                        x[l,i]=self.xub[i]
                    elif x[l,i]<self.xlb[i]:
                        x[l,i]=self.xlb[i]

            ### Store ###
            Optimizer.store([self.currentIteration, self.xbest, self.ybest])

    ### Restart algorithm ###
    def restart(self):
        [self.currentIteration, self.xbest, self.ybest] = Optimizer.load()
