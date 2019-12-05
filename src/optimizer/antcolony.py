import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.optimizer.optimizer import Optimizer
from src.optimizer.pareto import Pareto


### Ant colony optimization ###
class ACO(Optimizer):
    def __init__(self, fct, xbounds, ybounds, cbounds=[], nparticles=10, q=0.1, eps=0.1, 
                 colonySize=10, archiveSize=10, parallel=False, optidir='./.myopti', args=()):
        super().__init__(fct, xbounds, ybounds, cbounds=cbounds, optidir=optidir, parallel=parallel,  args=args)

        self.colonySize = colonySize
        self.archiveSize = archiveSize
        self.xranked, self.xbest = np.zeros((0, self.xdim)), np.zeros((0, self.xdim))
        self.yranked, self.ybest = np.zeros((0, self.ydim)), np.zeros((0, self.ydim))
        self.cranked, self.cbest = np.zeros((0, self.cdim)), np.zeros((0, self.cdim))
        self.pranked = np.zeros((0, 1))
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
            xNorm = Optimizer._nondimensionalize(np.vstack((x, self.xranked)), self.xlb, self.xub )
            yNorm = Optimizer._nondimensionalize(np.vstack((y, self.yranked)), self.ylb, self.yub )
            c = np.vstack((c, self.cranked))
            p = np.vstack((p, self.pranked))

            ### Pareto Ranks ###
            ranks = Pareto.computeParetoRanks(yNorm+p)
            idxs = np.argsort(ranks)

            ### Sort according to ranks ###
            xNorm, yNorm, ranks, p, c = xNorm[idxs, :], yNorm[idxs, :], ranks[idxs], p[idxs], c[idxs, :]

            self.xranked = Optimizer._dimensionalize(xNorm[:self.archiveSize,:], self.xlb, self.xub )
            self.yranked = Optimizer._dimensionalize(yNorm[:self.archiveSize,:], self.ylb, self.yub )
            self.cranked = c[:self.archiveSize,:]
            self.pranked = p[:self.archiveSize,:]
            ranks = ranks[:self.archiveSize]

            ### Store best values ###
            self.xbest = self.xranked[ranks[:self.archiveSize]<1,:]
            self.ybest = self.yranked[ranks[:self.archiveSize]<1,:]
            self.cbest = self.cranked[ranks[:self.archiveSize]<1,:] 

            ### Calculate weights ###
            omega = 1/(self.q*self.archiveSize*np.sqrt(2*np.pi))*np.exp((-ranks**2+2*ranks-1.0)/(2*self.q**2*self.archiveSize**2))
            p = omega/np.sum(omega)

            ### Build new solution ###
            x = np.zeros((self.colonySize, self.xdim))

            for l in range(self.colonySize):
                for i in range(self.xdim):
                    j = np.random.choice(np.arange(self.archiveSize), p=p)
                    Sji = self.xranked[j,:][i]
                    sigma = self.eps*np.sum(np.abs(self.xranked[j,i]-self.xranked[:,i]))

                    ### Sample from normal distribution ###
                    x[l,i] = Sji + sigma*np.random.normal()
                    if x[l,i]>self.xub[i]:
                        x[l,i]=self.xub[i]
                    elif x[l,i]<self.xlb[i]:
                        x[l,i]=self.xlb[i]

            ### Store ###
            self.store([self.currentIteration, self.xranked, self.yranked])

    ### Restart algorithm ###
    def restart(self):
        [self.currentIteration, self.xranked, self.yranked] = self.load()
