"""Generic optimizer
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import functools
from datetime import datetime

import logging

from typing import Callable, Union, Tuple

from .optimizer.pareto import Pareto


# Generic Optimizer class
class Optimizer:

    def __init__(
        self,
        fct: Callable,
        xbounds: list,
        ybounds: list,
        cbounds: list = [],
        epsDominanceBins: Union[None, int]=None,
        optidir: str = './.myOpti',
        args: tuple = (),
        parallel: bool = False
    ) -> None:
        """Generic optimizer class. Gets inherited by the different
        specific optimizers

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
        epsDominanceBins : Union[None, int], optional
            Number of bins for epsDominance calculation
        optidir : str, optional
            The directory where intermediate break points are stored (
            to restart the optimization)
        args : tuple, optional
            Description
        parallel : bool, optional
            Args to pass for the optimization function (fct)
        """
        self.fct = fct
        self.currentIteration = 0
        self.parallel = parallel

        self.xdim = len(xbounds)
        self.ydim = len(ybounds)
        self.cdim = len(cbounds)

        self.xlb = np.asarray([x[0] for x in xbounds])
        self.ylb = np.asarray([x[0] for x in ybounds])
        self.clb = np.asarray([x[0] for x in cbounds])

        self.xub = np.asarray([x[1] for x in xbounds])
        self.yub = np.asarray([x[1] for x in ybounds])
        self.cub = np.asarray([x[1] for x in cbounds])

        self.xbest = np.zeros((0, self.cdim))
        self.ybest = np.zeros((0, self.ydim))
        self.cbest = np.zeros((0, self.xdim))

        # Sanity checks
        assert np.any(self.xlb < self.xub),
        "X: Lower bound must be smaller than upper bound"
        assert np.any(self.ylb < self.yub),
        "Y: Lower bound must be smaller than upper bound"
        if self.clb.shape[0] > 0:
            assert np.any(self.clb < self.cub),
            "C: Lower bound must be smaller than upper bound"

        # Database
        self.paralabels = ["para_{}".format(p) for p in range(self.xdim)]
        self.trgtlabels = ["trgt_{}".format(p) for p in range(self.ydim)]
        self.cstrlabels = ["cstr_{}".format(p) for p in range(self.cdim)]

        # Eps dominace
        self.epsDominanceBins = epsDominanceBins

        # Store opti dir and name
        self.optidir = optidir

        # Keywordarguments
        self.args = args


    def evaluate(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the optimization function

        Parameters
        ----------
        X : np.ndarray
            Design Vectors [nDesigns, nXdim]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Description
        """

        output = self.fct(X, *self.args)

        Y = output[0].reshape(X.shape[0], self.ydim)
        C = output[1].reshape(X.shape[0], self.cdim) \
            if self.cdim > 0 else np.zeros((X.shape[0], self.cdim))

        # Build penalty function
        Py = Optimizer.boundaryCheck(Y, self.ylb, self.yub)
        Px = Optimizer.boundaryCheck(X, self.xlb, self.xub)
        Pc = Optimizer.boundaryCheck(C, self.clb, self.cub)

        # Assemble all penalties
        P = (Py + Px + Pc)

        # Return to optimizer
        return Y, C, P

    def initialize(self) -> None:
        """Initialize
        """
        os.makedirs(self.optidir, exist_ok=True)
        self.currentIteration = 0

    @staticmethod
    def boundaryCheck(
        Y: np.ndarray,
        ylb: np.ndarray,
        yub: np.ndarray
    ) -> np.ndarray:
        """Check boundary violation and penalizis it

        Parameters
        ----------
        Y : np.ndarray
            array
        ylb : np.ndarray
            lower bound
        yub : np.ndarray
            upper bound

        Returns
        -------
        np.ndarray
            Array with violation penalty
        """
        Y = Optimizer._nondimensionalize(Y, ylb, yub)
        index = (Y < 0) | (Y > 1.0)
        Y[~index] = 0.0
        return np.sum(Y**2, axis=1).reshape(-1, 1)


    def epsDominance(self) -> None:
        """Epsilon Dominance and delete dominated points from ranked designs
        """
        bins = np.linspace(0, 1, self.epsDominanceBins)
        binDistance, index2delete = {}, []

        for n in range(self.yranked.shape[0]):
            Ydim = Optimizer._nondimensionalize(
                self.yranked[n, :],
                self.ylb,
                self.yub
            )

            inds = np.digitize(Ydim, bins)

            inds_key = '-'.join(map(str, inds))
            dist = sum(
                [(Ydim[i] - bins[inds[i] - 1])**2 for i in range(self.ydim)]
            )

            # Check if design is in bin or not
            if inds_key in list(binDistance.keys()):
                if binDistance[inds_key][0] < dist:
                    index2delete.append(n)
                else:
                    index2delete.append(binDistance[inds_key][1])
                    binDistance[inds_key][0] = dist
                    binDistance[inds_key][1] = n
            else:
                binDistance[inds_key] = [dist, n]

        self.yranked = np.delete(self.yranked, index2delete, axis=0)
        self.xranked = np.delete(self.xranked, index2delete, axis=0)

    @staticmethod
    def _nondimensionalize(
        x: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray
    ) -> np.ndarray:
        """Nondimensionalize x by lower and upper bound

        Parameters
        ----------
        x : np.ndarray
            Description
        lb : np.ndarray
            Description
        ub : np.ndarray
            Description
        """

        return (x - lb) / (ub - lb)

    @staticmethod
    def _dimensionalize(
        x: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray
    ) -> np.ndarray:
        """Dimensionalize x by lower and upper bound

        Parameters
        ----------
        x : np.ndarray
            Description
        lb : np.ndarray
            Description
        ub : np.ndarray
            Description

        No Longer Returned
        ------------------
        param
            Description
        """
        return lb + x * (ub - lb)

    def load(self) -> None:
        """Restart

        Returns
        -------
        None
            Description
        """
        if not os.path.isdir(self.optidir):
            return
        elif len(os.listdir(self.optidir)) == 0:
            return
        with open(self.returnLatest(), 'rb') as f1:
            datalist = pickle.load(f1)
        return datalist

    def store(self, datalist: list) -> None:
        """Backup

        Parameters
        ----------
        datalist : list
            list of dataitems to store
        """
        with open(os.path.join(self.optidir,
                               datetime.now().strftime("%Y%m%d-%H%M%S") +
                               '.pkl'), 'wb') as f1:
            pickle.dump(datalist, f1)

    def returnLatest(self) -> None:
        """List backupfiles and return latest file

        Returns
        -------
        None
            Description
        """
        return os.path.join(self.optidir, sorted(os.listdir(self.optidir))[-1])
