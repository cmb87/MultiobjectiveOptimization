import numpy as np
import matplotlib.pyplot as plt
import json

from src.pareto import Parteo


class Particle(object):
    """docstring for Particle"""

    def __init__(self, tfct, ndim, tdim, particleID=None):

        self.particleID = None
        self.w = 1.0
        self.c1 = 0.5
        self.c2 = 0.5

        self.particleNeigbors = []

        self.x = np.zeros(ndim)
        self.v = np.zeros(ndim)
        self.pbest = np.zeros(ndim)
        self.gbest = np.zeros(ndim)
        self.y = np.zeros(tdim)

        self.tfct = tfct

    ### Initialize ###
    def initialize(self, xlb, xub, vlb=None, vub=None):
        if vlb is None or vub is None:
            vlb = -np.ones(self.x.shape)
            vub = np.ones(self.x.shape)

        self.v = vlb + np.random.rand() * (vub - vlb)
        self.x = xlb + np.random.rand() * (xub - xlb)

        ### Evaluate the particle's position ###
        self.y = Particle.evaluate(self.tfct, self.x)
        ### Set new personal best value ###
        self.pbest = self.y.copy()

    ### find leader ###
    def findLeader(self):

        for particle in self.particleNeigbors:
            if Pareto.dominates(particle.pbest, self.y):

                ### Update the Particle's position ###
    def update(self):

        r1 = np.random.rand()
        r2 = np.random.rand()

        ### Update particle velocity ###
        self.v = self.W * self.v + self.c1 * r1 * (self.pbest - self.x) + self.c2 * r2 * (self.gbest - self.x)
        self.x = self.x + self.v

        ### Add mutation ###

        ### Evaluate new particle's fitness ###
        self.y = Particle.evaluate(self.tfct, self.x)

        ### Evaluate if we have become better ###
        if Pareto.dominates(self.pbest, self.y):
            self.pbest = self.y.copy()

    ### Evaluate the particle ###
    @staticmethod
    def evaluate(fct, x):
        return fct(x)

    ### Compare with external archive ###
    def compareWithArchive(self):
        pass

    ### Store to archive ###
    def store(self):
        pass
