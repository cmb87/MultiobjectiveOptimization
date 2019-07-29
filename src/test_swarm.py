import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pareto import Pareto
from src.optimizer import Swarm
from src.test_functions import rosenbrock
from src.test_functions import rosenbrockConstrained
from src.test_functions import rosenbrockContour
from src.test_functions import rosenbrockContourConstrained
from src.test_functions import binhAndKorn
from src.test_functions import animateSwarm, animateSwarm2


if True:
    itermax = 80
    xbounds = [(0,5),(0,3)]
    ybounds = [(0,150),(0,150)]
    cbounds = [(0, 25),(7.7, 1e+5)]


    swarm = Swarm(binhAndKorn, xbounds, ybounds, cbounds, itermax=itermax, optimizationDirection=["minimize", "minimize"])
    swarm.initialize()
    Xcine, Ycine = swarm.iterate(visualize=True)

    animateSwarm2(Xcine, Ycine, xbounds=xbounds, ybounds=ybounds)



if False:
    itermax = 20
    xbounds = [(-4,4),(-4,4)]
    ybounds = [(0,13500)]
    cbounds = [(0, 1.3)]

    swarm = Swarm(rosenbrockConstrained, xbounds, ybounds, cbounds, itermax=itermax, optimizationDirection=["minimize"])
    swarm.initialize()
    Xcine, Ycine = swarm.iterate(visualize=True)

    animateSwarm(Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False)

