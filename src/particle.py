import numpy as np
import matplotlib.pyplot as plt
import json

from src.pareto import Pareto
from src.database import Database

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
