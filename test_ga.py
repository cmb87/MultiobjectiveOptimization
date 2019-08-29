import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pipeline.graph import OptimizationGraph
from src.optimizer.pareto import Pareto
from src.optimizer.genetic import GA

from test_functions import rosenbrock
from test_functions import rosenbrockContour
from test_functions import rosenbrockContourConstrained
from test_functions import binhAndKorn
from test_functions import animateSwarm, animateSwarm2



if __name__ == "__main__":

    if True:
        d = 32
        ### Define toolchain ###
        graph = OptimizationGraph(xdim=d, rdim=1, tindex=[0], cindex=[])
        graph.singleProcessChain(lambda x: np.sum(x**2, axis=1)/d) # 

        ### Optimizer stuff ###
        itermax = 1000
        xbounds = d*[[-4,4]]
        ybounds = [(0,13500)]
        cbounds = []

        ga = GA(graph.run, xbounds, ybounds, cbounds, npop=10)

        ga.initialize()
        ga.iterate(itermax)



        # ### Postprocessing ###
        #Xcine, Ycine = graph.postprocessAnimate()
        #animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)
        graph.postprocess()



    if True:
        ### Define toolchain ###
        graph = OptimizationGraph(xdim=2, rdim=2, tindex=[0], cindex=[1], xlabels=["x", "y"], rlabels=["z", "c"])
        graph.singleProcessChain(rosenbrock)

        ### Optimizer stuff ###
        itermax = 20
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = [(0, 1.3)]

        ga = GA(graph.run, xbounds, ybounds, cbounds, npop=20)
        ga.initialize()
        ga.iterate(itermax)


        #swarm.restart(resetParticles=True)
        #swarm.iterate(5)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False)
        #tc.postprocess()
        

