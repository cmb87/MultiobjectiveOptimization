import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pareto import Pareto

NPOINTS = 50


X = np.random.rand(NPOINTS, 2)
Y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
#Xpareto, Ypareto, paretoIndex, Xdominated, Ydominated, dominatedIndex = Pareto.computeParetoOptimalMember(X,X)

ranked = Pareto.computeParetoRanks(X, X, targetdirection=[-1, 1])
Xranked, Yranked, rank = Pareto.rankedToRanks(ranked)

plt.scatter(Xranked[:, 0], Xranked[:, 1], c=rank)
plt.show()
