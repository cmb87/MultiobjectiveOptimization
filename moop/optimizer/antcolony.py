"""Ant colony optimization
"""
import logging
from typing import Callable, Union

import numpy as np
from .pareto import Pareto
from .optimizer import Optimizer


class ACO(Optimizer):
    def __init__(
        self,
        fct: Callable,
        xbounds: list,
        ybounds: list,
        cbounds: list = [],
        callback: Union[Callable, None] = None,
        nparticles: int = 10,
        q: float = 0.1,
        eps: float = 0.1,
        colonySize: int = 10,
        archiveSize: int = 10,
        parallel: int = False,
        optidir: str = "./.myopti",
        args: tuple = (),
    ) -> None:
        """Ant colony optimization
        
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
        nparticles : int, optional
            Description
        q : float, optional
            Description
        eps : float, optional
            Description
        colonySize : int, optional
            Colony size of the ants
        archiveSize : int, optional
            Description
        parallel : int, optional
            Description
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
            optidir=optidir,
            parallel=parallel,
            args=args,
        )

        self.colonySize = colonySize
        self.archiveSize = archiveSize
        self.xranked = np.zeros((0, self.xdim))
        self.yranked = np.zeros((0, self.ydim))
        self.cranked = np.zeros((0, self.cdim))

        self.xbest = np.zeros((0, self.xdim))
        self.ybest = np.zeros((0, self.ydim))
        self.cbest = np.zeros((0, self.cdim))
        self.pranked = np.zeros((0, 1))
        self.q = q
        self.eps = eps

    def iterate(self, itermax: int) -> None:
        """Iterate

        Parameters
        ----------
        itermax : int
            Maximal number of iterations
        """
        x = Optimizer._dimensionalize(
            np.random.rand(self.colonySize, self.xdim), self.xlb, self.xub
        )

        # Start iterating
        for n in range(itermax):
            self.currentIteration += 1
            logging.info("Episode : {}".format(self.currentIteration))

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

            # Sort according to ranks
            xNorm, yNorm = xNorm[idxs, :], yNorm[idxs, :]
            ranks, p, c = ranks[idxs], p[idxs], c[idxs, :]

            self.xranked = Optimizer._dimensionalize(
                xNorm[: self.archiveSize, :], self.xlb, self.xub
            )
            self.yranked = Optimizer._dimensionalize(
                yNorm[: self.archiveSize, :], self.ylb, self.yub
            )
            self.cranked = c[: self.archiveSize, :]
            self.pranked = p[: self.archiveSize, :]
            ranks = ranks[: self.archiveSize]

            # Store best values
            self.xbest = self.xranked[ranks[: self.archiveSize] < 1, :]
            self.ybest = self.yranked[ranks[: self.archiveSize] < 1, :]
            self.cbest = self.cranked[ranks[: self.archiveSize] < 1, :]

            # Calculate weights
            omega = (
                1.0
                / (self.q * self.archiveSize * np.sqrt(2 * np.pi))
                * np.exp(
                    (-(ranks ** 2) + 2 * ranks - 1.0)
                    / (2 * self.q ** 2 * self.archiveSize ** 2)
                )
            )

            # Norm it
            p = omega / np.sum(omega)

            # Build new solution
            x = np.zeros((self.colonySize, self.xdim))

            for l in range(self.colonySize):
                for i in range(self.xdim):
                    j = np.random.choice(np.arange(self.archiveSize), p=p)
                    Sji = self.xranked[j, :][i]
                    sigma = self.eps * np.sum(
                        np.abs(self.xranked[j, i] - self.xranked[:, i])
                    )

                    # Sample from normal distribution
                    x[l, i] = Sji + sigma * np.random.normal()
                    if x[l, i] > self.xub[i]:
                        x[l, i] = self.xub[i]
                    elif x[l, i] < self.xlb[i]:
                        x[l, i] = self.xlb[i]

            # Store
            self.store([self.currentIteration, self.xranked, self.yranked])

    # Restart algorithm
    def restart(self) -> None:
        [self.currentIteration, self.xranked, self.yranked] = self.load()
