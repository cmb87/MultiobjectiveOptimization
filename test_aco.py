import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pareto import Pareto
from src.optimizer import ACO
from test_functions import rosenbrock
from test_functions import rosenbrockConstrained
from test_functions import rosenbrockContour
from test_functions import rosenbrockContourConstrained
from test_functions import binhAndKorn
from test_functions import animateSwarm, animateSwarm2

from src.database import Database


if __name__ == "__main__":

    if True:
        itermax = 20
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = []

        aco = ACO(rosenbrock, xbounds, ybounds, cbounds, colonySize=10, q=0.5, eps=0.1)

        aco.initialize()
        aco.iterate(20)

        aco.restart()
        aco.iterate(10)
        #Xcine, Ycine = aco.postprocessAnimate()
        #animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)

    if False:
        itermax = 100
        xbounds = [(0,5),(0,3)]
        ybounds = [(0,60),(0,60)]
        cbounds = [(0, 25),(7.7, 1e+2)]


        aco = ACO(binhAndKorn, xbounds, ybounds, cbounds, colonySize=10, q=0.1, eps=0.1)
        aco.initialize()
        aco.iterate(itermax)

        Xcine, Ycine = aco.postprocessAnimate()
        animateSwarm2(Xcine, Ycine, xbounds=xbounds, ybounds=ybounds)


    if False:
        itermax = 50
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = [(0, 1.3)]

        aco = ACO(rosenbrockConstrained, xbounds, ybounds, cbounds, colonySize=10, q=2.1, eps=0.1)

        aco.initialize()
        aco.iterate(itermax)

        Xcine, Ycine = aco.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False)

        