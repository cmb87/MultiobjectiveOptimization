import logging

import numpy as np
from .pipeline.graph import OptimizationGraph
from .optimizer.swarm import Swarm
from .optimizer.testfunctions import (
    rosenbrock,
    binhAndKorn,
    animateSwarm,
    animateSwarm2,
    rosenbrockContour,
    rosenbrockContourConstrained,
)

# Set logging formats
logging.basicConfig(
    level=logging.INFO,
    format=("[%(filename)8s] [%(levelname)4s] :  %(funcName)s - %(message)s"),
)


if __name__ == "__main__":

    # ========================================================

    CASE1 = True
    CASE2 = True
    CASE3 = True
    CASE4 = True

    #  wdw dwd

    # ========================================================
    if CASE1:
        logging.info(50 * "=")
        logging.info("CASE1 - Start Testing")
        logging.info(50 * "=")

        itermax = 40
        xbounds = [(-5, 10), (0, 15)]
        ybounds = [(0, 13500)]
        cbounds = []

        def branin(X, a, b, c, r, s, t):
            return (
                a * (X[:, 1] - b * X[:, 0] ** 2 + c * X[:, 0] - r) ** 2
                + s * (1 - t) * np.cos(X[:, 0])
                + s,
                None,
            )

        swarm = Swarm(
            branin,
            xbounds,
            ybounds,
            cbounds,
            nparticles=10,
            minimumSwarmSize=10,
            args=(1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)),
        )
        swarm.initialize()
        swarm.iterate(itermax)

        swarm.restart()
        swarm.iterate(5)

        logging.info(swarm.xbest)
        logging.info(swarm.ybest)

        logging.info("CASE1 - passed :)")

    # ========================================================
    if CASE2:
        logging.info(50 * "=")
        logging.info("CASE2 - Start Testing")
        logging.info(50 * "=")

        # Define toolchain
        graph = OptimizationGraph(
            xdim=2,
            rdim=2,
            tindex=[0],
            cindex=[],
            xlabels=["x", "y"],
            rlabels=["z", "c"],
        )
        graph.singleProcessChain(rosenbrock)

        # Optimizer stuff
        itermax = 40
        xbounds = [(-4, 4), (-4, 4)]
        ybounds = [(0, 13500)]
        cbounds = []

        swarm = Swarm(
            graph.run, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10
        )

        swarm.initialize()
        swarm.iterate(itermax)

        swarm.restart(resetParticles=False)
        swarm.iterate(5)

        # Postprocessing
        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)
        # tc.postprocess()

        logging.info("CASE2 - passed :)")

    # ========================================================
    if CASE3:
        logging.info(50 * "=")
        logging.info("CASE3 - Start Testing")
        logging.info(50 * "=")

        # Define toolchain
        graph = OptimizationGraph(
            xdim=2, rdim=4, tindex=[0, 1], cindex=[2, 3], xlabels=["x", "y"]
        )
        graph.singleProcessChain(binhAndKorn)

        # Optimizer stuff
        itermax = 30
        xbounds = [(0, 5), (0, 3)]
        ybounds = [(0, 60), (0, 60)]
        cbounds = [(0, 25), (7.7, 1e2)]

        swarm = Swarm(
            graph.run, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10
        )
        swarm.initialize()
        swarm.iterate(itermax)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm2(Xcine, Ycine, xbounds=xbounds, ybounds=ybounds)
        # graph.postprocess()

        logging.info("CASE3 - passed :)")

    # ========================================================
    if CASE4:
        logging.info(50 * "=")
        logging.info("CASE4 - Start Testing")
        logging.info(50 * "=")
        # Define toolchain

        graph = OptimizationGraph(
            xdim=2,
            rdim=2,
            tindex=[0],
            cindex=[1],
            xlabels=["x", "y"],
            rlabels=["z", "c"],
        )
        graph.singleProcessChain(rosenbrock)

        # Optimizer stuff
        itermax = 20
        xbounds = [(-4, 4), (-4, 4)]
        ybounds = [(0, 13500)]
        cbounds = [(0, 1.3)]

        swarm = Swarm(
            graph.run, xbounds, ybounds, cbounds, nparticles=10, minimumSwarmSize=10
        )

        swarm.initialize()
        swarm.iterate(itermax)

        # swarm.restart(resetParticles=True)
        # swarm.iterate(5)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm(
            Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False
        )
        # tc.postprocess()
        logging.info("CASE4 - passed :)")
