import logging

import numpy as np
from .pipeline.graph import OptimizationGraph
from .optimizer.genetic import GA
from .optimizer.testfunctions import (
    rosenbrock,
    animateSwarm,
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

        ga = GA(
            branin,
            xbounds,
            ybounds,
            cbounds,
            npop=10,
            args=(1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)),
        )
        ga.initialize()
        ga.iterate(itermax)

        ga.restart()
        ga.iterate(5)

        logging.info(ga.xbest)
        logging.info(ga.ybest)

        logging.info("CASE1 - passed :)")

    # ========================================================
    if CASE2:
        logging.info(50 * "=")
        logging.info("CASE2 - Start Testing")
        logging.info(50 * "=")

        d = 32
        # Define toolchain
        graph = OptimizationGraph(xdim=d, rdim=1, tindex=[0], cindex=[])
        graph.singleProcessChain(lambda x: np.sum(x ** 2, axis=1) / d)  #

        # Optimizer stuff
        itermax = 3
        xbounds = d * [[-4, 4]]
        ybounds = [(0, 13500)]
        cbounds = []

        ga = GA(graph.run, xbounds, ybounds, cbounds, npop=10)

        ga.initialize()
        ga.iterate(itermax)

        # # Postprocessing
        # Xcine, Ycine = graph.postprocessAnimate()
        # animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)
        # graph.postprocess()

        del graph, ga

        logging.info("CASE2 - passed :)")

    # ========================================================
    if CASE3:
        logging.info(50 * "=")
        logging.info("CASE3 - Start Testing")
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

        ga = GA(graph.run, xbounds, ybounds, cbounds, npop=20)
        ga.initialize()
        ga.iterate(itermax)

        # swarm.restart(resetParticles=True)
        # swarm.iterate(5)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm(
            Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False
        )
        # tc.postprocess()
