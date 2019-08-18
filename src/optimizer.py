import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import functools

from src.particle import Particle
from src.pareto import Pareto
from src.database import Database

### Generic Optimizer class ###
class Optimizer(object):

    ### Constructor ###
    def __init__(self, fct, xbounds, ybounds, cbounds=[], epsDominanceBins=None, **kwargs):

        self.fct = fct
        self.currentIteration = 0
        self.nparas = len(xbounds)
        self.ntrgts = len(ybounds)
        self.ncstrs = len(cbounds)

        self.xlb, self.xub = np.asarray([x[0] for x in xbounds]), np.asarray([x[1] for x in xbounds])
        self.ylb, self.yub = np.asarray([x[0] for x in ybounds]), np.asarray([x[1] for x in ybounds])
        self.clb, self.cub = np.asarray([x[0] for x in cbounds]), np.asarray([x[1] for x in cbounds])

        self.Xbest, self.Ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))

        ### Sanity checks ###
        assert np.any(self.xlb<self.xub), "X: Lower bound must be smaller than upper bound"
        assert np.any(self.ylb<self.yub), "Y: Lower bound must be smaller than upper bound"
        if self.clb.shape[0]>0:
            assert np.any(self.clb<self.cub), "C: Lower bound must be smaller than upper bound"

        ### Database ###
        self.paralabels=["para_{}".format(p) for p in range(self.nparas)]
        self.trgtlabels=["trgt_{}".format(p) for p in range(self.ntrgts)]
        self.cstrlabels=["cstr_{}".format(p) for p in range(self.ncstrs)]

        ### Eps dominace ###
        self.epsDominanceBins = epsDominanceBins

        ### Keywordarguments ###
        self.kwargs = kwargs


    ### Evaluate function ###
    def evaluate(self, X):
        ### Evaluate toolchain ###
        output = self.fct(X, **self.kwargs)
        Y = output[0].reshape(X.shape[0], self.ntrgts)
        C = output[1].reshape(X.shape[0], self.ncstrs) if self.ncstrs>0 else np.zeros((X.shape[0], self.ncstrs))


        ### Build penalty function ###
        Py = Optimizer.boundaryCheck(Y, self.ylb, self.yub)
        Px = Optimizer.boundaryCheck(X, self.xlb, self.xub)
        Pc = Optimizer.boundaryCheck(C, self.clb, self.cub)
        P  = (Py+Px+Pc)

        ### Return to optimizer ###
        return Y, C, P

    ### Initialize ###
    def initialize(self):
        self.currentIteration = 0

    ### Check boundary violation and penalizis it ###
    @staticmethod
    def boundaryCheck(Y, ylb, yub):
        Y = Optimizer._nondimensionalize(Y, ylb, yub)
        index = (Y<0) | (Y>1.0)
        Y[~index] = 0.0
        return np.sum(Y**2,axis=1).reshape(-1,1) 


    ### Epsilon Dominance ###
    def epsDominance(self):
        bins = np.linspace(0,1, self.epsDominanceBins)
        binDistance, index2delete = {}, []

        for n in range(self.Ybest.shape[0]):
            Ydim = Optimizer._nondimensionalize(self.Ybest[n,:], self.ylb, self.yub)
            
            inds = np.digitize(Ydim, bins)
            
            inds_key = '-'.join(map(str,inds))
            dist = sum([(Ydim[i]-bins[inds[i]-1])**2 for i in range(self.ntrgts)])

            if not inds_key in list(binDistance.keys()):
                binDistance[inds_key] = [dist, n]
            else:
                if binDistance[inds_key][0] < dist:
                    index2delete.append(n)
                else:
                    index2delete.append(binDistance[inds_key][1])
                    binDistance[inds_key][0] = dist
                    binDistance[inds_key][1] = n

        self.Ybest = np.delete(self.Ybest,index2delete,axis=0)
        self.Xbest = np.delete(self.Xbest,index2delete,axis=0)
        

    @staticmethod
    def _nondimensionalize(x, lb, ub):
        return (x - lb) / (ub - lb)

    @staticmethod
    def _dimensionalize(x, lb, ub):
        return lb + x * (ub - lb)

    ### Restart ###
    @staticmethod
    def load(filename=".optimizerBackup.pkl"):
        with open(filename,'rb') as f1:
            datalist = pickle.load(f1)
        return datalist

    ### Backup ###
    @staticmethod
    def store(datalist, filename=".optimizerBackup.pkl"):
        with open(filename,'wb') as f1:
            pickle.dump(datalist,f1)





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



### Particle Swarm Optimizer ###
class Swarm(Optimizer):

    def __init__(self, fct, xbounds, ybounds, cbounds=[], nparticles=10, nichingDistanceY=0.1, nichingDistanceX=0.1, 
                 epsDominanceBins=6, minimumSwarmSize=10, **kwargs):
        super().__init__(fct, xbounds, ybounds, cbounds=cbounds, epsDominanceBins=epsDominanceBins, **kwargs)


        ### Initialize swarm ###
        self.particles = [Particle(self.nparas, pid, vinitScale=0.5) for pid in range(nparticles)]
        self.Xbest, self.Ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))
        self.minimumSwarmSize = minimumSwarmSize
        self.particleReductionRate = 0.99

        ### Niching parameters ###
        self.delta_y = nichingDistanceY
        self.delta_x = nichingDistanceX
        self.epsDominanceBins = epsDominanceBins

    ### swarm size ###
    @property
    def swarmSize(self):
        return len(self.particles)
    

    ### Iterate ###
    def iterate(self, itermax):

        ### Start iterating ###
        for n in range(itermax):
            self.currentIteration += 1
            print(" Episode : {}".format(self.currentIteration))

            ### Remove particles ###
            SwarmSizeTarget = int(self.particleReductionRate*self.swarmSize) if self.swarmSize > self.minimumSwarmSize else self.minimumSwarmSize
            while self.swarmSize > SwarmSizeTarget:
                self.particles.pop()

            ### Evaluate the new particle's position ###
            X = np.asarray([Optimizer._dimensionalize(particle.x, self.xlb, self.xub) for particle in self.particles])

            ### Evaluate it ###
            Y, C, P = self.evaluate(X)

            ### Update particle targets ###
            for i, particle in enumerate(self.particles):
                particle.y = Optimizer._nondimensionalize(Y[i, :], self.ylb, self.yub)
                particle.p = P[i, :]

            ### Determine new pbest ###
            [particle.updatePersonalBest() for particle in self.particles]

            ### Determine new gbest ###
            Xpareto, Ypareto, Ppareto, pvals = self.findParetoMembers(X, Y, P)

            for i, particle in enumerate(self.particles):
                index = np.random.choice(np.arange(Xpareto.shape[0]), p=pvals)
                particle.ygbest = Optimizer._nondimensionalize(Ypareto[index, :], self.ylb, self.yub)
                particle.xgbest = Optimizer._nondimensionalize(Xpareto[index, :], self.xlb, self.xub)
                particle.pgbest = Ppareto[index, :]

            ### Apply eps dominance ###
            self.epsDominance()

            ### Update particles ###
            [particle.update() for particle in self.particles]

            ### Store ###
            Optimizer.store([self.currentIteration, self.particles, self.Xbest, self.Ybest])


    ### Restart algorithm ###
    def restart(self, resetParticles=False):
        [self.currentIteration, self.particles, self.Xbest, self.Ybest] = Optimizer.load()
        if resetParticles:
            [particle.reset() for particle in self.particles]


    ### Update leader particle without external archive ###
    def findParetoMembers(self, X, Y, P):
        ### Only designs without penality violation ###
        validIndex = P[:,0] == 0.0
        
        ### Some designs are valid ###
        if P[validIndex].shape[0] > 0:

            # Calculate pareto ranks
            X = np.vstack((self.Xbest, X[validIndex,:]))
            Y = np.vstack((self.Ybest, Y[validIndex,:]))

            paretoIndex, dominatedIndex = Pareto.computeParetoOptimalMember(Y)
            Xpareto, Ypareto = X[paretoIndex,:], Y[paretoIndex,:]
            Ppareto = np.zeros((Xpareto.shape[0],1))
            ### Update best particle archive ###                                                
            self.Xbest, self.Ybest = Xpareto, Ypareto

            ### Probality of being a leader ###
            py = Swarm.neighbors(Ypareto, self.delta_y)
            px = Swarm.neighbors(Xpareto, self.delta_x)
            pvals = py*px

        ### All designs have a penalty ==> Move to point with lowest violation ###
        else:
            index = np.argmin(P, axis=0)
            Ypareto, Xpareto = Y[index, :].reshape(1,self.ntrgts), X[index, :].reshape(1,self.nparas)
            Ppareto = np.zeros((Xpareto.shape[0],1))
            pvals = np.ones(1)

        return  Xpareto, Ypareto, Ppareto, pvals/pvals.sum()


    @staticmethod
    def neighbors(X, delta):
        D,N = np.zeros((X.shape[0],X.shape[0])), np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i,j] = np.linalg.norm(X[i,:]-X[j,:])

        N[D<delta] = 1.0
        del D
        return 1.0/np.exp(N.sum(axis=0))

