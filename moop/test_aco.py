import logging

import numpy as np
from .pipeline.graph import OptimizationGraph
from .optimizer.antcolony import ACO
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

        aco = ACO(
            branin,
            xbounds,
            ybounds,
            cbounds,
            colonySize=10,
            q=0.3,
            eps=0.2,
            args=(1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)),
        )
        aco.initialize()
        aco.iterate(itermax)

        aco.restart()
        aco.iterate(5)

        logging.info(aco.xbest)
        logging.info(aco.ybest)
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

        # Optimizer
        itermax = 20
        xbounds = [(-4, 4), (-4, 4)]
        ybounds = [(0, 13500)]
        cbounds = []

        aco = ACO(graph.run, xbounds, ybounds, cbounds, colonySize=10, q=0.3, eps=0.2)

        aco.initialize()
        aco.iterate(20)

        aco.restart()
        aco.iterate(10)

        # Animation
        Xcine, Ycine = graph.postprocessAnimate()

        animateSwarm(Xcine, Ycine, rosenbrockContour, xbounds=xbounds, store=False)

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

        # Optimizer
        itermax = 30
        xbounds = [(0, 5), (0, 3)]
        ybounds = [(0, 60), (0, 60)]
        cbounds = [(0, 25), (7.7, 1e2)]

        aco = ACO(graph.run, xbounds, ybounds, cbounds, colonySize=10, q=0.1, eps=0.1)

        aco.initialize()
        aco.iterate(itermax)

        Xcine, Ycine = graph.postprocessAnimate()
        animateSwarm2(Xcine, Ycine, xbounds=xbounds, ybounds=ybounds)

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

        # Assign optimization function to graph
        graph.singleProcessChain(rosenbrock)

        # Optimizer
        itermax = 50
        xbounds = [(-4, 4), (-4, 4)]
        ybounds = [(0, 13500)]
        cbounds = [(0, 1.3)]

        aco = ACO(graph.run, xbounds, ybounds, cbounds, colonySize=10, q=2.1, eps=0.1)

        aco.initialize()
        aco.iterate(itermax)

        Xcine, Ycine = graph.postprocessAnimate()

        animateSwarm(
            Xcine, Ycine, rosenbrockContourConstrained, xbounds=xbounds, store=False
        )

        logging.info("CASE4 - passed :)")
