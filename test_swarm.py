import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pipeline.graph import OptimizationGraph
from src.optimizer.pareto import Pareto
from src.optimizer.swarm import Swarm

from test_functions import rosenbrock
from test_functions import rosenbrockContour
from test_functions import rosenbrockContourConstrained
from test_functions import binhAndKorn
from test_functions import animateSwarm, animateSwarm2



if __name__ == "__main__":

    if False:

        ### Define toolchain ###
        graph = OptimizationGraph(xdim=2, rdim=2, tindex=[0], cindex=[], xlabels=["x", "y"], rlabels=["z", "c"])
        graph.singleProcessChain(rosenbrock)

        ### Optimizer stuff ###
        itermax = 40
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = []

        swarm = Swarm(graph.run, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10)

        swarm.initialize()
        swarm.iterate(itermax)

        swarm.restart(resetParticles=False)
        swarm.iterate(5)

        ### Postprocessing ###
        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)
        #tc.postprocess()


    if False:

        ### Define toolchain ###
        graph = OptimizationGraph(xdim=2, rdim=4, tindex=[0,1], cindex=[2,3], xlabels=["x", "y"])
        graph.singleProcessChain(binhAndKorn)

         ### Optimizer stuff ###
        itermax = 30
        xbounds = [(0,5),(0,3)]
        ybounds = [(0,60),(0,60)]
        cbounds = [(0, 25),(7.7, 1e+2)]


        swarm = Swarm(graph.run, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10)
        swarm.initialize()
        swarm.iterate(itermax)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm2(Xcine, Ycine, xbounds=xbounds, ybounds=ybounds)
        #graph.postprocess()


    if True:
        ### Define toolchain ###
        graph = OptimizationGraph(xdim=2, rdim=2, tindex=[0], cindex=[1], xlabels=["x", "y"], rlabels=["z", "c"])
        graph.singleProcessChain(rosenbrock)

        ### Optimizer stuff ###
        itermax = 20
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = [(0, 1.3)]

        swarm = Swarm(graph.run, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10)

        swarm.initialize()
        swarm.iterate(itermax)

        #swarm.restart(resetParticles=True)
        #swarm.iterate(5)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False)
        #tc.postprocess()
        

