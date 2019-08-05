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
        index = np.arange(0,Y.shape[0]) if index is None else index

        Ypareto, Ydominated = Pareto.cull(Y.tolist(), Pareto.dominates)

        paretoIndex = [index[np.where(Y == ypareto)[0][0]] for i, ypareto in enumerate(Ypareto)]
        dominatedIndex = [index[np.where(Y == ydominated)[0][0]] for i, ydominated in enumerate(Ydominated)]

        return paretoIndex, dominatedIndex


    @staticmethod
    def computeParetoRanks(Y):
        indexunranked = np.arange(0,Y.shape[0])
        ranks = np.zeros(Y.shape[0])
        paretoRank = 0
        while len(indexunranked) > 0:
            paretoIndex, indexunranked = Pareto.computeParetoOptimalMember(Y[indexunranked], index=indexunranked)
            ranks[paretoIndex] = paretoRank
            paretoRank += 1
        return ranks

