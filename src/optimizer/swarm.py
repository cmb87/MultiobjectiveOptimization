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
        self.particles = [Particle(self.xdim, pid, vinitScale=0.5) for pid in range(nparticles)]
        self.xbest, self.ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))
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
            x = np.asarray([Optimizer._dimensionalize(particle.x, self.xlb, self.xub) for particle in self.particles])

            ### Evaluate it ###
            y, c, p = self.evaluate(x)

            ### Update particle targets ###
            for i, particle in enumerate(self.particles):
                particle.y = Optimizer._nondimensionalize(y[i, :], self.ylb, self.yub) + p[i,:]

            ### Determine new pbest ###
            [particle.updatePersonalBest() for particle in self.particles]

            ### Determine new gbest ###
            paretoIndex, _ = Pareto.computeParetoOptimalMember(Optimizer._nondimensionalize(y, self.ylb, self.yub)+p)

            for i, particle in enumerate(self.particles):
                idx = np.random.choice(paretoIndex, 1, replace=False)[0]
                particle.xgbest = Optimizer._nondimensionalize(x[idx, :], self.xlb, self.xub)
                particle.ygbest = Optimizer._nondimensionalize(y[idx, :], self.ylb, self.yub)

            ### Apply eps dominance ###
            self.epsDominance()

            ### Update particles ###
            [particle.update() for particle in self.particles]

            ### Store ###
            Optimizer.store([self.currentIteration, self.particles, self.xbest, self.ybest])


    ### Restart algorithm ###
    def restart(self, resetParticles=False):
        [self.currentIteration, self.particles, self.xbest, self.ybest] = Optimizer.load()
        if resetParticles:
            [particle.reset() for particle in self.particles]




### Particle ###
class Particle(object):
    """docstring for Particle"""
    def __init__(self, xdim, particleID, vinitScale=0.1, mutateRate=0.03):

        self.particleID = particleID
        self.w = 0.5
        self.c1 = 0.5
        self.c2 = 0.5
        self.xdim = xdim

        self.x = np.random.rand(self.xdim)
        self.v = vinitScale*(-1.0 + 2.0*np.random.rand(self.xdim))
        self.xpbest = self.x.copy()
        self.xgbest = None

        ### Objectives ###
        self.y = None
        self.ypbest = None
        self.ygbest = None

        ### Reset Counter ###
        self.resetCtr = 0
        self.resetLimit = 12

        ### Mutation rate ###
        self.mutateRate = mutateRate

    ### reset ###
    def reset(self):
        self.x = np.random.rand(self.xdim)
        self.v = 1.0*(-1.0 + 2.0*np.random.rand(self.xdim))
        self.resetCtr = 0

    ### find pbest ###
    def updatePersonalBest(self):

        ### Initialization ###
        if self.ypbest is None:
            self.ypbest = self.y
            self.xpbest = self.x
            self.resetCtr = 0
            return

        ### Pareto dominance ###
        if Pareto.dominates(self.y, self.ypbest):
            self.ypbest = self.y
            self.xpbest = self.x
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
            self.x = np.random.rand(self.xdim)

        ### If pbest hasn't changed ###
        if self.resetCtr > self.resetLimit:
            self.reset()

        ### Design space violation ###
        self.x[self.x>1.0] = 1.0
        self.x[self.x<0.0] = 0.0
