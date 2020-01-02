import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pipeline.graph import OptimizationGraph
from src.optimizer.pareto import Pareto
from src.optimizer.antcolony import ACO

from test_functions import rosenbrock
from test_functions import rosenbrockContour
from test_functions import rosenbrockContourConstrained
from test_functions import binhAndKorn
from test_functions import animateSwarm, animateSwarm2


if __name__ == "__main__":


    if True:
        itermax = 40
        xbounds = [(-5,10),(0,15)]
        ybounds = [(0,13500)]
        cbounds = []

        def branin(X, a, b, c, r, s, t):
            return a*(X[:,1]-b*X[:,0]**2+c*X[:,0]-r)**2 + s*(1-t)*np.cos(X[:,0])+s, None

        aco = ACO(branin, xbounds, ybounds, cbounds, colonySize=10, q=0.3, eps=0.2, args=(1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)))
        aco.initialize()
        aco.iterate(itermax)

        aco.restart()
        aco.iterate(5)

        print(aco.xbest)
        print(aco.ybest)

    if False:
        ### Define toolchain ###
        graph = OptimizationGraph(xdim=2, rdim=2, tindex=[0], cindex=[], xlabels=["x", "y"], rlabels=["z", "c"])
        graph.singleProcessChain(rosenbrock)

        ### Optimizer ###
        itermax = 20
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = []

        aco = ACO(graph.run, xbounds, ybounds, cbounds, colonySize=10, q=0.3, eps=0.2)

        aco.initialize()
        aco.iterate(20)

        aco.restart()
        aco.iterate(10)
        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)

    if False:
        ### Define toolchain ###
        graph = OptimizationGraph(xdim=2, rdim=4, tindex=[0,1], cindex=[2,3], xlabels=["x", "y"])
        graph.singleProcessChain(binhAndKorn)

        ### Optimizer ###
        itermax = 30
        xbounds = [(0,5),(0,3)]
        ybounds = [(0,60),(0,60)]
        cbounds = [(0, 25),(7.7, 1e+2)]


        aco = ACO(graph.run, xbounds, ybounds, cbounds, colonySize=10, q=0.1, eps=0.1)
        aco.initialize()
        aco.iterate(itermax)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm2(Xcine, Ycine, xbounds=xbounds, ybounds=ybounds)


    if True:
        ### Define toolchain ###
        graph = OptimizationGraph(xdim=2, rdim=2, tindex=[0], cindex=[1], xlabels=["x", "y"], rlabels=["z", "c"])
        graph.singleProcessChain(rosenbrock)


        ### Optimizer ###
        itermax = 50
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = [(0, 1.3)]

        aco = ACO(graph.run, xbounds, ybounds, cbounds, colonySize=10, q=2.1, eps=0.1)

        aco.initialize()
        aco.iterate(itermax)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False)

        