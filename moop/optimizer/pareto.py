"""Pareto Module
"""
from typing import Tuple, Union

import numpy as np


class Pareto:
    @staticmethod
    def cull(pts: list, dominates: list) -> Tuple[list, list]:
        """Evaluate Pareto rank

        Parameters
        ----------
        pts : list
            Description
        dominates : list
            Description

        Returns
        -------
        Tuple(list, list)
            Description
        """
        dominated, cleared = [], []
        remaining = pts

        while remaining:
            candidate = remaining[0]
            new_remaining = []
            for other in remaining[1:]:
                [new_remaining, dominated][Pareto.dominates(candidate, other)].append(
                    other
                )

            if not any(Pareto.dominates(other, candidate) for other in new_remaining):
                cleared.append(candidate)  # PARETO POINT
            else:
                dominated.append(candidate)
            remaining = new_remaining
        return cleared, dominated

    @staticmethod
    def dominates(row: list, rowCandidate: list) -> list:
        """Domination function

        Parameters
        ----------
        row : list
            Description
        rowCandidate : list
            Description

        Returns
        -------
        list
            Description
        """

        return all(r <= rc for r, rc in zip(row, rowCandidate))

    @staticmethod
    def computeParetoOptimalMember(
        Y: np.ndarray, index: Union[None, list] = None
    ) -> Tuple[list, list]:
        """Compute Rank 0 members from given set of target labels

        Parameters
        ----------
        Y : np.ndarray
            Target labels
        index : Union[None, list], optional
            Description
        """

        index = np.arange(0, Y.shape[0]) if index is None else np.asarray(index)

        Ypareto, Ydominated = Pareto.cull(Y.tolist(), Pareto.dominates)
        paretoIndex = [
            index[np.all(Y == ypareto, axis=1)][0] for i, ypareto in enumerate(Ypareto)
        ]
        dominatedIndex = [i for i in index if i not in paretoIndex]

        return paretoIndex, dominatedIndex

    @staticmethod
    def computeParetoRanks(Y: np.ndarray) -> np.ndarray:
        """Compute all Pareto Ranks

        Parameters
        ----------
        Y : np.ndarray
            Description

        Returns
        -------
        np.ndarray
            Description
        """
        indexunranked = np.arange(0, Y.shape[0])
        ranks = 99 * np.ones(Y.shape[0])
        paretoRank = 0
        while len(indexunranked) > 0:
            paretoIndex, indexunranked = Pareto.computeParetoOptimalMember(
                Y[indexunranked, :], index=indexunranked
            )
            ranks[paretoIndex] = paretoRank
            paretoRank += 1

        return ranks
