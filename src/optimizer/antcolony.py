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
        self.X = Optimizer._dimensionalize(np.random.rand(colonySize, self.nparas), self.xlb, self.xub)
        self.K = np.zeros((0, self.ntrgts +1 + self.nparas))
        self.q = q
        self.eps = eps

    ### Iterate ###
    def iterate(self, itermax):

        ### Start iterating ###
        for n in range(itermax):
            self.currentIteration += 1
            print(" Episode : {}".format(self.currentIteration))

            ### Evaluate it ###
            Y, C, P = self.evaluate(self.X)

            ### Add to archive ###
            self.K = np.vstack((self.K, np.hstack((Y, P, self.X))))


            ### Rank Solution ###
            ranks = Pareto.computeParetoRanks(self.K[:,:self.ntrgts +1])
            
            ### Sort according to ranks ###
            self.K = self.K[np.argsort(ranks)[:self.archiveSize],:]
            ranks = ranks[np.argsort(ranks)][:self.archiveSize]

            self.Xbest = self.K[:self.archiveSize, self.ntrgts +1:]
            self.Ybest = self.K[:self.archiveSize, :self.ntrgts]

            ### Calculate weights ###
            omega = 1/(self.q*self.archiveSize*np.sqrt(2*np.pi))*np.exp((-ranks**2+2*ranks-1.0)/(2*self.q**2*self.archiveSize**2))
            p = omega/np.sum(omega)

            ### Build new solution ###
            self.X = np.zeros((self.colonySize,self.nparas))

            for l in range(self.colonySize):
                for i in range(self.nparas):
                    j = np.random.choice(np.arange(self.archiveSize), p=p)
                    Sji = self.Xbest[j,:][i]
                    sigma = self.eps*np.sum(np.abs(self.Xbest[j,i]-self.Xbest[:,i]))

                    ### Sample from normal distribution ###
                    self.X[l,i] = Sji + sigma*np.random.normal()
                    if self.X[l,i]>self.xub[i]:
                        self.X[l,i]=self.xub[i]
                    elif self.X[l,i]<self.xlb[i]:
                        self.X[l,i]=self.xlb[i]

            ### Store ###
            Optimizer.store([self.currentIteration, self.X, self.K])

    ### Restart algorithm ###
    def restart(self):
        [self.currentIteration, self.X, self.K] = Optimizer.load()
