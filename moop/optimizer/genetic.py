"""Genetic algorithm module
"""
import logging
from typing import Callable, Union

import numpy as np
from .pareto import Pareto
from .optimizer import Optimizer


class GA(Optimizer):
    def __init__(
        self,
        fct: Callable,
        xbounds: list,
        ybounds: list,
        cbounds: list = [],
        callback: Union[Callable, None] = None,
        npop: int = 20,
        nichingDistanceY: float = 0.1,
        nichingDistanceX: float = 0.1,
        parallel: bool = False,
        epsDominanceBins: int = 6,
        optidir: str = "./.myopti",
        args: tuple = (),
    ) -> None:
        """Constructor of Genetic Algorithm
        Parameters
        ----------
        fct : Callable
            Function to be optimized
        xbounds : list
            Design space bounds [[x1low, x1high], [x2low, x2high], ...]
        ybounds : list
            Region of interest [[y1low, y1high], [y2low, y2high], ...]
        cbounds : list, optional
            Constraint bounds [[c1low, c1high], [c2low, c2high], ...]
        callback : Union[Callable, None], optional
            Callback function, executed every iteration. Must be of form
            f(x, y, c, nIter)
        npop : int, optional
            Population Count
        nichingDistanceY : float, optional
            Niching distance for target space
        nichingDistanceX : float, optional
            Niching distance for design space
        parallel : bool, optional
            Parallel execution on/off
        epsDominanceBins : int, optional
            Number of bins for epsDominance calculation
        optidir : str, optional
            The directory where intermediate break points are stored (
            to restart the optimization)
        args : tuple, optional
            Args to pass for the optimization function (fct)
        """
        super().__init__(
            fct,
            xbounds,
            ybounds,
            cbounds=cbounds,
            callback=callback,
            epsDominanceBins=epsDominanceBins,
            parallel=parallel,
            optidir=optidir,
            args=args,
        )

        self.xranked = np.zeros((0, self.xdim))
        self.yranked = np.zeros((0, self.ydim))
        self.cranked = np.zeros((0, self.cdim))

        self.xbest = np.zeros((0, self.xdim))
        self.ybest = np.zeros((0, self.ydim))
        self.cbest = np.zeros((0, self.cdim))

        self.pranked = np.zeros((0, 1))
        self.npop = npop

    def iterate(self, itermax: int, mut: float = 0.5, crossp: float = 0.7) -> None:
        """Iterate

        Parameters
        ----------
        itermax : int
            Maximum number of iterations
        mut : float, optional
            Mutation probability
        crossp : float, optional
            Crossover probability
        """
        x = Optimizer._dimensionalize(
            np.random.rand(self.npop, self.xdim), self.xlb, self.xub
        )

        # Start iterating
        for n in range(itermax):
            self.currentIteration += 1
            logging.info(" Episode : {}".format(self.currentIteration))

            # Evaluate it
            y, c, p = self.evaluate(x)

            # Append value to so far best seen designs
            xNorm = Optimizer._nondimensionalize(
                np.vstack((x, self.xranked)), self.xlb, self.xub
            )
            yNorm = Optimizer._nondimensionalize(
                np.vstack((y, self.yranked)), self.ylb, self.yub
            )
            c = np.vstack((c, self.cranked))
            p = np.vstack((p, self.pranked))

            # Pareto Ranks
            ranks = Pareto.computeParetoRanks(yNorm + p)
            idxs = np.argsort(ranks)

            # Sort by rank
            xNorm, yNorm = xNorm[idxs, :], yNorm[idxs, :]
            ranks, p, c = ranks[idxs], p[idxs], c[idxs, :]

            # Ranked
            self.xranked = Optimizer._dimensionalize(
                xNorm[: self.npop, :], self.xlb, self.xub
            )
            self.yranked = Optimizer._dimensionalize(
                yNorm[: self.npop, :], self.ylb, self.yub
            )
            self.cranked = c[: self.npop, :]
            self.pranked = p[: self.npop, :]

            # Eps Dominance
            index2delete = self.epsDominance(self.ybest)
            self.xranked = np.delete(self.xranked, index2delete, axis=0)
            self.yranked = np.delete(self.yranked, index2delete, axis=0)
            self.pranked = np.delete(self.pranked, index2delete, axis=0)
            self.cranked = np.delete(self.cranked, index2delete, axis=0)

            ranksBest = np.delete(ranks[:self.npop], index2delete, axis=0)


            # Store best values

            self.xbest = self.xranked[ranksBest < 1, :]
            self.ybest = self.yranked[ranksBest < 1, :]
            self.cbest = self.cranked[ranksBest < 1, :]

            # Filter by eps dominance
            logging.debug("mean", x.mean(), "std", x.std())

            # Differential Evolution
            x = np.zeros((self.npop, self.xdim))
            for j in range(self.npop):
                idxs = [i for i in range(xNorm.shape[0]) if i != j]
                a, b, c = xNorm[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mut * (b - c), 0, 1)
                cross_points = np.random.rand(self.xdim) < crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.xdim)] = True

                x[j, :] = Optimizer._dimensionalize(
                    np.where(cross_points, mutant, xNorm[j, :]), self.xlb, self.xub
                )

            # Store
            self.store(
                [self.currentIteration, self.xranked, self.yranked, self.pranked]
            )

    def restart(self) -> None:
        """Restart algorithm
        """
        [self.currentIteration, self.xranked, self.yranked, self.pranked] = self.load()
