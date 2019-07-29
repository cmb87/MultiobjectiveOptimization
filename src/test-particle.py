import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from src.particle import Particle
from src.swarm import Swarm
from src.pareto import Pareto

### Rosenbrock test function ###


def rosenbrock(X):
    if X.ndim == 2:
        x, y = X[:, 0], X[:, 1]
    else:
        x, y = X[0], X[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


xlb, xub = [-2, -2], [2, 2]
ylb, yub = [0], [500]
tfct = rosenbrock

### Initialize swarm ####
swarm = Swarm(tfct, xlb, xub, ylb, yub, vlb=None, vub=None, nparticles=10, itermax=100, targetdirection=None)
swarm.initialize()
swarm.iterate()
