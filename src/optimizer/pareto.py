import numpy as np


### Pareto Class ###
class Pareto(object):

    @staticmethod
    def cull(pts, dominates):
        dominated, cleared = [], []
        remaining = pts

        while remaining:
            candidate = remaining[0]
            new_remaining = []
            for other in remaining[1:]:
                [new_remaining, dominated][Pareto.dominates(candidate, other)].append(other)
            if not any(Pareto.dominates(other, candidate) for other in new_remaining):
                cleared.append(candidate)   # PARETO POINT
            else:
                dominated.append(candidate)
            remaining = new_remaining
        return cleared, dominated

    @staticmethod
    def dominates(row, rowCandidate):
        return all(r <= rc for r, rc in zip(row, rowCandidate))


    @staticmethod
    def computeParetoOptimalMember(Y, index=None):
        index = np.arange(0,Y.shape[0]) if index is None else np.asarray(index)

        Ypareto, Ydominated = Pareto.cull(Y.tolist(), Pareto.dominates)
        paretoIndex = [index[np.all(Y == ypareto, axis=1)][0]    for i, ypareto    in enumerate(Ypareto)]
        dominatedIndex = [i for i in index if not i in paretoIndex]

        return paretoIndex, dominatedIndex


    @staticmethod
    def computeParetoRanks(Y):
        indexunranked = np.arange(0,Y.shape[0])
        ranks = 99*np.ones(Y.shape[0])
        paretoRank = 0
        while len(indexunranked) > 0:
            paretoIndex, indexunranked = Pareto.computeParetoOptimalMember(Y[indexunranked,:], index=indexunranked)
            ranks[paretoIndex] = paretoRank
            paretoRank += 1

        return ranks

