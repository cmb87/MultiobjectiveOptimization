import numpy as np
import matplotlib.pyplot as plt
import json

from src.pareto import Pareto


class Particle(object):
    """docstring for Particle"""
    def __init__(self, D, particleID, vinitScale=0.1):

        self.particleID = particleID

        self.w = 0.5
        self.c1 = 0.5
        self.c2 = 0.5
        self.D = D

        self.x = np.random.rand(self.D)
        self.v = vinitScale*(-1.0 + 2.0*np.random.rand(self.D))
        self.xpbest = self.x.copy()
        self.xgbest = None

        ### Note: The last objective (-1) is the penalty ###
        self.y = None
        self.ypbest = None
        self.ygbest = None

    ### find pbest ###
    def updatePersonalBest(self):

        ### penalty violation ###
        if np.abs(self.y[-1]) < np.abs(self.ypbest[-1]):
            self.ypbest = self.y
            self.xpbest = self.x

        ### Pareto dominance ###
        elif Pareto.dominates(self.y, self.ypbest):
            self.ypbest = self.y
            self.xpbest = self.x


    ### Update the Particle's position ###
    def update(self):

        r1 = np.random.rand()
        r2 = np.random.rand()

        ### Update particle velocity ###
        self.v = self.w * self.v + self.c1*r1*(self.xpbest - self.x) + self.c2*r2*(self.xgbest - self.x)
        self.x += self.v

        ### Design space violation ###
        self.x[self.x>1.0] = 1.0
        self.x[self.x<0.0] = 0.0
