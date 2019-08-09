import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pareto import Pareto
from src.optimizer import Swarm
from test_functions import rosenbrock
from test_functions import rosenbrockConstrained
from test_functions import rosenbrockContour
from test_functions import rosenbrockContourConstrained
from test_functions import binhAndKorn
from test_functions import animateSwarm, animateSwarm2

from src.database import Database


if __name__ == "__main__":

    if False:
        itermax = 40
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = []

        swarm = Swarm(rosenbrock, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10, nparallel=1)

        swarm.initialize()
        swarm.iterate(120)

        #swarm.restart(resetParticles=True)
        #swarm.iterate(5)

        Xcine, Ycine = swarm.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)
        swarm.postprocess()
        Xbest, Ybest, lastIteration = swarm.postprocessReturnBest()
        print(Ybest)


    if False:
        itermax = 100
        xbounds = [(0,5),(0,3)]
        ybounds = [(0,60),(0,60)]
        cbounds = [(0, 25),(7.7, 1e+2)]


        swarm = Swarm(binhAndKorn, xbounds, ybounds, cbounds, nichingDistanceX=0.1, nichingDistanceY=0.1, epsDominanceBins=8, nparallel=1)
        swarm.initialize()
        swarm.iterate(itermax)

        Xcine, Ycine = swarm.postprocessAnimate()
        animateSwarm2(Xcine, Ycine, xbounds=xbounds, ybounds=ybounds)

        swarm.postprocess(resdir='../', store=True, xlabel=["x1", "x2"])


    if True:
        itermax = 10
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = [(0, 1.3)]

        swarm = Swarm(rosenbrockConstrained, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10, nparallel=1)

        swarm.initialize()
        swarm.iterate(itermax)

        #swarm.restart(resetParticles=True)
        #swarm.iterate(5)

        Xcine, Ycine = swarm.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False)
        swarm.postprocess()
        

