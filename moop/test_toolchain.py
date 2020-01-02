import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.toolchain import Toolchain
from src.pareto import Pareto
from src.optimizer import Swarm
from test_functions import rosenbrock
from test_functions import rosenbrockContour
from test_functions import rosenbrockContourConstrained
from test_functions import binhAndKorn
from test_functions import animateSwarm, animateSwarm2

from src.database import Database


if __name__ == "__main__":

    if True:

        ### Define toolchain ###
        tc = Toolchain(2, 2, trgtIndex=[0])
        tc.chain = rosenbrock

        ### Optimizer stuff ###
        itermax = 40
        xbounds = [(-4,4),(-4,4)]
        ybounds = [(0,13500)]
        cbounds = []

        swarm = Swarm(tc.execute, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10)

        swarm.initialize()
        swarm.iterate(itermax)

        swarm.restart(resetParticles=False)
        swarm.iterate(5)

        ### Postprocessing ###
        #Xcine, Ycine = tc.postprocessAnimate()
        #animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)
        #tc.postprocess()

        ### 
        tc.storeToolchain()
        tc2 = Toolchain.find_by_id(1)
        print(tc2)

        tc3 = Toolchain.find_all()
        print(tc3)