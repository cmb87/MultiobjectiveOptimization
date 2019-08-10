import math

from mutual_information import mutual_information


def pre_calc_entropy(data, F):
    # Calculates entropies used for normalizing the mutual information later on.
    # To make sure that we get compatible values we exploit that H(x) == I(x,x)
    # even so it is computationally a bit more expensive.
    Hs = {}
    for f in F:
        Hs[f] = H(data[f])
    return Hs


def amifs(data, F, C, max_features, Hs=None, MIs=None):
    """
    Method described in this paper:
    https://www.researchgate.net/publication/4116665_AMIFS_Adaptive_feature_selection_by_using_mutual_information

    :param data:  pandas DataFrame

    :param F: list, set  -  initial set of input features (columns of DataFrame), will be modified

    :param C: string - target column in data

    :param max_features: int

    :param Hs: dict[string -> float] (optional) - entropy of the features.
        Pre-calculated entropies can be passed to avoid recalculation,
        for example, when using the function for various targets on the same data.

    :param MIs: dict[(string, string) -> float] (optional) - mutual information between features.
        Pre-calculated results can be passed to avoid recalculation,
        for example, when using the function for various targets on the same data.

    :return:
    """
    F = set(F)
    S = []
    ICs = {}  # mutual information to target

    if Hs is None:
        Hs = pre_calc_entropy(data, F)

    if MIs is None:
        MIs = {}

    # Ignore inputs with zero entropy (no variation across whole dataset)
    constant = [f for f in F if Hs[f] < 1e-10]
    F -= set(constant)

    # Gets best single feature with max mutual information to target
    max_I = -math.inf
    best_feature = None
    for f_i in F:
        ICs[f_i] = I(data[C], data[f_i])
        if ICs[f_i] > max_I:
            best_feature = f_i
            max_I = ICs[f_i]

    S.append(best_feature)
    F.remove(best_feature)
    fi_score = {best_feature: max_I}

    # Greedy selection of further features
    while len(S) < max_features:
        best_feature = None
        max_score = -math.inf
        for f_i in F:
            score = ICs[f_i]
            for f_s in S:
                if not ((f_i, f_s) in MIs):
                    MIs[f_i, f_s] = I(data[f_i], data[f_s])
                    MIs[f_s, f_i] = MIs[f_i, f_s]
                score -= MIs[f_i, f_s] / (len(S) * min(Hs[f_i], Hs[f_s]))
            fi_score[f_i] = score
            if score > max_score:
                best_feature = f_i
                max_score = score

        S.append(best_feature)
        F.remove(best_feature)
    return S, ICs, MIs, fi_score


def I(x, y):
    mi = mutual_information(x.values, y.values, bins=(10, 15), percentile_range=(.1, 99.9))
    return mi


def H(x):
    return I(x, x)
