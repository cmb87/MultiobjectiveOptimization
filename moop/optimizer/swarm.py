"""Particle Swarm Optimizer
"""
import logging
from typing import Callable

import numpy as np
from .pareto import Pareto
from .optimizer import Optimizer


class Swarm(Optimizer):
    def __init__(
        self,
        fct: Callable,
        xbounds: list,
        ybounds: list,
        cbounds: list = [],
        nparticles: int = 10,
        nichingDistanceY: float = 0.1,
        nichingDistanceX: float = 0.1,
        epsDominanceBins: int = 6,
        minimumSwarmSize: int = 10,
        parallel: bool = False,
        optidir: str = "./.myopti",
        args: tuple = (),
    ) -> None:
        """Particle Swarm Optimizer

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
        nparticles : int, optional
            Description
        nichingDistanceY : float, optional
            Description
        nichingDistanceX : float, optional
            Description
        epsDominanceBins : int, optional
            Description
        minimumSwarmSize : int, optional
            Description
        parallel : bool, optional
            Description
        optidir : str, optional
            The directory where intermediate break points are stored (
            to restart the optimization)
        args : tuple, optional
            Description
        """
        super().__init__(
            fct,
            xbounds,
            ybounds,
            cbounds=cbounds,
            epsDominanceBins=epsDominanceBins,
            parallel=parallel,
            optidir=optidir,
            args=args,
        )

        # Initialize swarm
        self.particles = [
            Particle(self.xdim, pid, vinitScale=0.5) for pid in range(nparticles)
        ]
        self.minimumSwarmSize = minimumSwarmSize
        self.particleReductionRate = 0.99
        self.xbest = np.zeros((0, self.xdim))
        self.ybest = np.zeros((0, self.ydim))
        self.cbest = np.zeros((0, self.cdim))

        # Niching parameters
        self.delta_y = nichingDistanceY
        self.delta_x = nichingDistanceX
        self.epsDominanceBins = epsDominanceBins

    # swarm size
    @property
    def swarmSize(self):
        return len(self.particles)

    def iterate(self, itermax: int) -> None:
        """Iterate

        Parameters
        ----------
        itermax : int
            Number of iterations
        """
        for n in range(itermax):
            self.currentIteration += 1
            logging.info(" Episode : {}".format(self.currentIteration))

            # Remove particles
            SwarmSizeTarget = (
                int(self.particleReductionRate * self.swarmSize)
                if self.swarmSize > self.minimumSwarmSize
                else self.minimumSwarmSize
            )

            while self.swarmSize > SwarmSizeTarget:
                self.particles.pop()

            # Evaluate the new particle's position
            x = np.asarray(
                [
                    Optimizer._dimensionalize(particle.x, self.xlb, self.xub)
                    for particle in self.particles
                ]
            )

            # Evaluate it
            y, c, p = self.evaluate(x)

            # Update particle targets
            for i, particle in enumerate(self.particles):
                particle.y = (
                    Optimizer._nondimensionalize(y[i, :], self.ylb, self.yub) + p[i, :]
                )

            # Determine new pbest
            [particle.updatePersonalBest() for particle in self.particles]

            # Determine new gbest
            paretoIndex, _ = Pareto.computeParetoOptimalMember(
                Optimizer._nondimensionalize(y, self.ylb, self.yub) + p
            )

            # Allocate best particles
            self.xbest = x[paretoIndex, :]
            self.ybest = y[paretoIndex, :]
            self.cbest = c[paretoIndex, :]

            for i, particle in enumerate(self.particles):
                idx = np.random.choice(paretoIndex, 1, replace=False)[0]
                particle.xgbest = Optimizer._nondimensionalize(
                    x[idx, :], self.xlb, self.xub
                )

                particle.ygbest = Optimizer._nondimensionalize(
                    y[idx, :], self.ylb, self.yub
                )

            # Apply eps dominance
            index2delete = self.epsDominance(self.ybest)
            self.xbest = np.delete(self.xbest, index2delete, axis=0)
            self.ybest = np.delete(self.ybest, index2delete, axis=0)
            self.cbest = np.delete(self.cbest, index2delete, axis=0)

            # Update particles
            [particle.update() for particle in self.particles]

            # Store
            self.store(
                [
                    self.currentIteration,
                    self.particles,
                    self.xbest,
                    self.ybest,
                    self.cbest,
                ]
            )

    def restart(self, resetParticles: bool = False) -> None:
        """Restart algorithm

        Parameters
        ----------
        resetParticles : bool, optional
            Description
        """

        # Get files from pickle
        [
            self.currentIteration,
            self.particles,
            self.xbest,
            self.ybest,
            self.cbest,
        ] = self.load()

        # Reset particles
        if resetParticles:
            [particle.reset() for particle in self.particles]


# Particle
class Particle:
    def __init__(
        self,
        xdim: int,
        particleID: int,
        vinitScale: float = 0.1,
        mutateRate: float = 0.03,
    ) -> None:
        """Constructor of a particle

        Parameters
        ----------
        xdim : int
            Description
        particleID : int
            Description
        vinitScale : float, optional
            Initial velocity
        mutateRate : float, optional
            Mutation rate
        """
        self.particleID = particleID
        self.w = 0.5
        self.c1 = 0.5
        self.c2 = 0.5
        self.xdim = xdim

        self.x = np.random.rand(self.xdim)
        self.v = vinitScale * (-1.0 + 2.0 * np.random.rand(self.xdim))
        self.xpbest = self.x.copy()
        self.xgbest = None

        # Objectives
        self.y = None
        self.ypbest = None
        self.ygbest = None

        # Reset Counter
        self.resetCtr = 0
        self.resetLimit = 12

        # Mutation rate
        self.mutateRate = mutateRate

    def reset(self) -> None:
        """Reset particle
        """
        self.x = np.random.rand(self.xdim)
        self.v = 1.0 * (-1.0 + 2.0 * np.random.rand(self.xdim))
        self.resetCtr = 0

    def updatePersonalBest(self) -> None:
        """Update personal best particle

        Returns
        -------
        None
            Description
        """
        # Initialization
        if self.ypbest is None:
            self.ypbest = self.y
            self.xpbest = self.x
            self.resetCtr = 0
            return None

        # Pareto dominance
        if Pareto.dominates(self.y, self.ypbest):
            self.ypbest = self.y
            self.xpbest = self.x
            self.resetCtr = 0
            return None

        # If non of the above cases match
        self.resetCtr += 1
        return None

    def update(self) -> None:
        """Update the Particle's position
        """
        r1 = np.random.rand()
        r2 = np.random.rand()

        # Update particle velocity
        self.v = (
            self.w * self.v
            + self.c1 * r1 * (self.xpbest - self.x)
            + self.c2 * r2 * (self.xgbest - self.x)
        )
        self.x += self.v

        # Mutation
        if np.random.rand() < self.mutateRate:
            self.x = np.random.rand(self.xdim)

        # If pbest hasn't changed
        if self.resetCtr > self.resetLimit:
            self.reset()

        # Design space violation
        self.x[self.x > 1.0] = 1.0
        self.x[self.x < 0.0] = 0.0
