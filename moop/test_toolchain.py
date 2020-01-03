import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pareto import Pareto
from src.optimizer import Swarm
from src.toolchain import Toolchain
from test_functions import rosenbrockToolchain
from test_functions import rosenbrockContour
from test_functions import animateSwarm, animateSwarm2
from src.database import Database

### Define toolchain ###
tc = Toolchain(2, 1, trgtIndex=[0])
tc.chain = rosenbrockToolchain


### Optimizer settings ###
itermax = 20
xbounds = [(-4,4),(-4,4)]
ybounds = [(0,13500)]
cbounds = []

swarm = Swarm(tc.execute, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10)
swarm.initialize()
swarm.iterate(itermax)

swarm.restart(resetParticles=False)
swarm.iterate(5)

### Toolchain postprocessing ###
Xcine, Ycine = tc.postprocessAnimate()

animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)
tc.postprocess()
# Xbest, Ybest, lastIteration = swarm.postprocessReturnBest()
# print(Ybest)