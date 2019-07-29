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
        return all(r >= rc for r, rc in zip(row, rowCandidate))

    @staticmethod
    def computeParetoOptimalMember(X, Y, targetdirection=None):
        tdir = np.asarray(Y.shape[1] * [-1]) if targetdirection is None else np.asarray(targetdirection)
        assert len(tdir) == Y.shape[1], "Target direction must match feature count"

        Y = tdir * Y.copy()
        Ypareto, Ydominated = Pareto.cull(Y.tolist(), Pareto.dominates)

        paretoIndex = [np.where(Y == ypareto)[0][0] for i, ypareto in enumerate(Ypareto)]
        dominatedIndex = [np.where(Y == ydominated)[0][0] for i, ydominated in enumerate(Ydominated)]

        Xpareto = np.asarray([X[p, :] for i, p in enumerate(paretoIndex)])
        Xdominated = np.asarray([X[p, :] for i, p in enumerate(dominatedIndex)])

        Ypareto = tdir * np.asarray(Ypareto)
        if len(Ydominated) > 0:
            Ydominated = tdir * np.asarray(Ydominated)

        return Xpareto, Ypareto, paretoIndex, Xdominated, Ydominated, dominatedIndex

    @staticmethod
    def computeParetoRanks(X, Y, targetdirection=None):

        Xunranked, Yunranked = X.copy(), Y.copy()
        paretoRank = 0
        ranked = {}

        while Xunranked.shape[0] > 1:
            Xpareto, Ypareto, paretoIndex, Xunranked, Yunranked, _ = Pareto.computeParetoOptimalMember(Xunranked, Yunranked, targetdirection=targetdirection)
            ranked[paretoRank] = {"Xpareto": Xpareto, "Ypareto": Ypareto, "ParetoIndex": paretoIndex}
            paretoRank += 1

        return ranked

    @staticmethod
    def rankedToRanks(ranked):
        feats = ranked[0]["Xpareto"].shape[1]
        trgts = ranked[0]["Ypareto"].shape[1]

        Xranked, Yranked = np.zeros((0, feats)), np.zeros((0, trgts))
        rank = np.zeros(0)

        for key, pareto in ranked.items():
            Xranked = np.vstack((Xranked, pareto["Xpareto"]))
            Yranked = np.vstack((Yranked, pareto["Ypareto"]))
            rank = np.hstack((rank, np.asarray(pareto["Xpareto"].shape[0] * [key])))

        return Xranked, Yranked, rank
