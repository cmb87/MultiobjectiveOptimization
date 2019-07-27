import numpy as np
import matplotlib.pyplot as plt
import json

from src.pareto import Pareto


class Particle(object):
    """docstring for Particle"""

    def __init__(self, particleID, tfct, xlb, xub, ylb, yup, vlb=None, vub=None):

        self.particleID = particleID
        self.tfct = tfct

        self.w = 1.0
        self.c1 = 0.5
        self.c2 = 0.5

        self.x = None
        self.v = None
        self.xpbest = None
        self.xgbest = None

        self.y = None
        self.ypbest = None
        self.ygbest = None

    ### find pbest ###
    def updatePersonalBest(self):
        if Pareto.dominates(self.y, self.ypbest):
            self.ypbest = self.y
            self.xpbest = self.x

    ### Update the Particle's position ###
    def update(self):

        r1 = np.random.rand()
        r2 = np.random.rand()

        ### Update particle velocity ###
        self.v = self.W * self.v + self.c1 * r1 * (self.xpbest - self.x) + self.c2 * r2 * (self.xgbest - self.x)
        self.x = self.x + self.v
