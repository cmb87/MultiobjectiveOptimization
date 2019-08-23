import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.optimizer.optimizer import Optimizer
from src.optimizer.pareto import Pareto



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


class Particle(object):
    """docstring for Particle"""
    def __init__(self, D, particleID, vinitScale=0.1, mutateRate=0.03):

        self.particleID = particleID

        self.w = 0.5
        self.c1 = 0.5
        self.c2 = 0.5
        self.D = D

        self.x = np.random.rand(self.D)
        self.v = vinitScale*(-1.0 + 2.0*np.random.rand(self.D))
        self.xpbest = self.x.copy()
        self.xgbest = None

        ### Objectives ###
        self.y = None
        self.ypbest = None
        self.ygbest = None

        ### penalty ###
        self.p = None
        self.ppbest = None
        self.pgbest = None

        ### Reset Counter ###
        self.resetCtr = 0
        self.resetLimit = 12

        ### Mutation rate ###
        self.mutateRate = mutateRate

    ### reset ###
    def reset(self):
        self.x = np.random.rand(self.D)
        self.v = 1.0*(-1.0 + 2.0*np.random.rand(self.D))
        self.resetCtr = 0

    ### find pbest ###
    def updatePersonalBest(self):

        ### Initialization ###
        if self.ypbest is None:
            self.ypbest = self.y
            self.xpbest = self.x
            self.ppbest = self.p
            self.resetCtr = 0
            return
        ### penalty violation ###
        if np.abs(self.p) < np.abs(self.ppbest):
            self.ypbest = self.y
            self.xpbest = self.x
            self.ppbest = self.p
            self.resetCtr = 0
            return
        ### Pareto dominance ###
        if Pareto.dominates(self.y, self.ypbest):
            self.ypbest = self.y
            self.xpbest = self.x
            self.ppbest = self.p
            self.resetCtr = 0
            return

        ### Special scenario (for fixing the swarm after resetting) ###
        if np.abs(self.ppbest) == 0.0 and np.abs(self.p)>0.0:
            self.resetCtr = 0
            return

        ### If non of the above cases match ###
        self.resetCtr += 1
        return

    ### Update the Particle's position ###
    def update(self):

        r1 = np.random.rand()
        r2 = np.random.rand()

        ### Update particle velocity ###
        self.v = self.w * self.v + self.c1*r1*(self.xpbest - self.x) + self.c2*r2*(self.xgbest - self.x)
        self.x += self.v

        ### Mutation ###
        if np.random.rand() < self.mutateRate: 
            self.x = np.random.rand(self.D)

        ### If pbest hasn't changed ###
        if self.resetCtr > self.resetLimit:
            self.reset()

        ### Design space violation ###
        self.x[self.x>1.0] = 1.0
        self.x[self.x<0.0] = 0.0
