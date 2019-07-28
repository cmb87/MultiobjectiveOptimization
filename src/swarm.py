import numpy as np
import matplotlib.pyplot as plt

from src.particle import Particle
from src.pareto import Pareto


class Swarm(object):
    """docstring for Particle"""
    TARGETDIRECTION = {"maximize": 1, "minimize": -1,
                       "max": 1, "min": -1, 1: 1, -1: -1, "1": 1, "-1": 1}

    def __init__(self, tfct, xlb, xub, ylb, yub, cfct=None, clb=None, cub=None,
                 vlb=None, vub=None, nparticles=10, itermax=10, targetdirection=None):

        self.tfct = tfct
        self.cfct = cfct
        self.xlb, self.xub = np.asarray(xlb), np.asarray(xub)
        self.ylb, self.yub = np.asarray(ylb), np.asarray(yub)
        self.clb, self.cub = np.asarray(clb), np.asarray(cub)

        self.vlb = -np.ones(self.xlb.shape[0]) if vlb is None else np.asarray(vlb)
        self.vub = np.ones(self.xlb.shape[0]) if vub is None else np.asarray(vub)

        self.particles = [Particle(pid) for pid in range(nparticles)]
        self.itermax = itermax

        self.targetdirection = -np.ones(len(self.ylb)) if targetdirection is None else np.asarray(targetdirection)
        self.Xbest, self.Ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))

    ### Swarmsize ###
    @property
    def swarmsize(self):
        return len(self.particles)

    ### Initialize ###
    def initialize(self):
        ### Initialize positions and velocity ###
        for particle in self.particles:
            x0 = self.xlb + np.random.rand(self.xlb.shape[0]) * (self.xub - self.xlb)
            v0 = self.vlb + np.random.rand(self.xlb.shape[0]) * (self.vub - self.vlb)
            particle.v = v0
            particle.x = Swarm._nondimensionalize(x0, self.xlb, self.xub)

        ### Evaluate the particle's position ###
        X = self._particles2array()
        Y = self.evaluateParticles(X)

        ### Update the particle targets ###
        self._array2particles(Y)

        ### Set new personal best value to initial y value ###
        for i, particle in enumerate(self.particles):
            particle.ypbest = particle.y
            particle.xpbest = particle.x

        ### Set global leaders ###
        self._updateLeadersWithoutExternalRepo(X, Y)

    ### Iterate ###
    def iterate(self, visualize=False):
        for n in range(self.itermax):

            ### Update particle positions ###
            [particle.update() for particle in self.particles]

            ### Evaluate the particle's position ###
            X = self._particles2array()
            Y = self.evaluateParticles(X)

            ### Update particle targets ###
            self._array2particles(Y)

            ### Determine new pbest ###
            [particle.updatePersonalBest() for particle in self.particles]

            ### Determine new gbest ###
            self._updateLeadersWithoutExternalRepo(X, Y)

            ### Visualize ###
            print(X.min(axis=0),X.max(axis=0))

    ### Update leader particle without external archive ###
    def _updateLeadersWithoutExternalRepo(self, X, Y):
        ### Add best points to evaluation ###
        X, Y = np.vstack((self.Xbest, X)), np.vstack((self.Ybest, Y))
        # Calculate pareto ranks
        Xpareto, Ypareto, paretoIndex, _, _, _ = Pareto.computeParetoOptimalMember(X, Y, targetdirection=self.targetdirection)
        pvals = Xpareto.shape[0] * [1.0 / Xpareto.shape[0]]
        ### Assign each particle a new leader ###
        for i, particle in enumerate(self.particles):
            index = np.random.choice(np.arange(Xpareto.shape[0]), p=pvals)
            particle.ygbest = self.targetdirection * Swarm._nondimensionalize(Ypareto[index, :], self.ylb, self.yub)
            particle.xgbest = Swarm._nondimensionalize(Xpareto[index, :], self.xlb, self.xub)

        self.Xbest, self.Ybest = Xpareto, Ypareto

    ### Evaluate all particles ###
    def evaluateParticles(self, X):
        return self.tfct(X).reshape(-1, 1)

    ### Convert particle postition to numpy array ###
    def _particles2array(self):
        return np.asarray([Swarm._dimensionalize(particle.x, self.xlb, self.xub) for particle in self.particles])

    ### Convert particle postition to numpy array ###
    def _array2particles(self, Y):
        for i, particle in enumerate(self.particles):
            particle.y = self.targetdirection * Swarm._nondimensionalize(Y[i, :], self.ylb, self.yub)

    @staticmethod
    def _nondimensionalize(x, lb, ub):
        return (x - lb) / (ub - lb)

    @staticmethod
    def _dimensionalize(x, lb, ub):
        return lb + x * (ub - lb)

    ### Compare with external archive ###
    def compareWithArchive(self):
        pass

    ### Store to archive ###
    def store(self):
        pass

    ### add to database ###
    def archiveStore(self):
        pass
