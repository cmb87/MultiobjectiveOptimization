import numpy as np
from src.particle import Particle
from src.pareto import Pareto


class Swarm(object):
    """docstring for Particle"""
    TARGETDIRECTION = {"maximize": 1, "minimize": -1,
                       "max": 1, "min": -1, 1: 1, -1: -1, "1": 1, "-1": 1}

    def __init__(self, tfct, xlb, xub, ylb, yup, vlb=None, vub=None, nparticles=10, itermax=10, targetdirection=None):

        self.tfct = tfct

        self.xlb = np.asarray(xlb)
        self.xub = np.asarray(xub)
        self.ylb = np.asarray(ylb)
        self.yub = np.asarray(yub)

        self.vlb = -np.ones(xlb.shape[0]) if vlb is None else np.asarray(vlb)
        self.vub = np.ones(xlb.shape[0]) if vub is None else np.asarray(vub)

        self.particles = [Particle(pid) for pid in nparticles]
        self.itermax = itermax

        self.targetdirection = -np.ones(len(self.ylb)) if targetdirection is None else np.asarray(targetdirection)

    ### Swarmsize ###
    @property
    def swarmsize(self):
        return len(self.particles)

    ### Initialize ###
    def initialize(self):

        ### Initialize positions and velocity ###
        for particle in self.particles:
            x0 = self.vlb + np.random.rand(self.xlb.shape[0]) * (self.vub - self.vlb)
            v0 = self.xlb + np.random.rand(self.xlb.shape[0]) * (self.xub - self.xlb)
            particle.v = v0
            particle.x = Swarm._nondimensionalize(x0, self.xlb, self.xub)

        ### Evaluate the particle's position ###
        Y = self.evaluateParticles()

        ### Update the particle targets ###
        self.updateParticleTargets(Y)

        ### Set new personal best value to initial y value ###
        for i, particle in enumerate(self.particles):
            particle.ypbest = particle.y
            particle.xpbest = particle.x

        ### Set global leaders ###
        X = self._particles2array()
        Xpareto, Ypareto, paretoIndex, _, _, _ = Pareto.computeParetoOptimalMember(X, Y, targetdirection=self.targetdirection)

    ### Iterate ###
    def iterate(self):
        for n in range(self.itermax):

            ### Update particle positions ###
            [particle.update() for particle in self.particles]

            ### Evaluate the particle's position ###
            Y = self.evaluateParticles()

            ### Update particle targets ###
            self._array2particles(Y)

            ### Determine new pbest ###
            [particle.updatePersonalBest() for particle in self.particles]

            ### Determine new gbest ###

    ### Evaluate all particles ###
    def evaluateParticles(self):
        X = self._particles2array()
        return self.fct(X)

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
