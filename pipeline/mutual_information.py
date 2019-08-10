import numpy as np

from sklearn.metrics import mutual_info_score


def mutual_information(x, y, bins=range(10, 15), percentile_range=None):
    """
    Returns mutual information between x and y.
    x and y are assumed to have continuous (float) values.
    Joint probabilities between x and y are estimated with a 2D histogram.
    :param x: array_like, shape (N,)
    :param y:  array_like, shape (N,)
    :param bins: int or range of int
    :param percentile_range: None or tuple (p0, p1) with p0 and p1 between 0 and 100 (incl.) as
        lower and upper limits to exclude outliers from the histogram binning,
        e.g. (1, 99) to define the range from the 1% to the 99% percentile.
    :return: float, NaN in case of errors
    """

    use = np.isfinite(x) & np.isfinite(y)

    if not np.any(use):
        return np.NaN

    if percentile_range is not None:
        xmin, xmax = np.nanpercentile(x, percentile_range)
        ymin, ymax = np.nanpercentile(y, percentile_range)
        _range = [[xmin, xmax], [ymin, ymax]]
        if (xmin >= xmax) or (ymin >= ymax):
            return np.NaN
    else:
        _range = None

    if isinstance(bins, int):
        c_xy = np.histogram2d(x[use], y[use], bins=bins, range=_range)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi
    else:
        mi = np.empty(len(bins), )
        for i, b in enumerate(bins):
            c_xy = np.histogram2d(x[use], y[use], bins=b, range=_range)[0]
            mi[i] = mutual_info_score(None, None, contingency=c_xy)
        return np.mean(mi)


def conditional_information(x, y, z, bins=range(10, 15), percentile_range=None):
    """

    Calculates I(X;Y|Z).  All variables assumed to have float values.

    Reference:

    @article{Brown:2012:CLM:2188385.2188387,
     author = {Brown, Gavin and Pocock, Adam and Zhao, Ming-Jie and Luj\'{a}n, Mikel},
     title = {Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection},
     journal = {J. Mach. Learn. Res.},
     issue_date = {3/1/2012},
     volume = {13},
     month = jan,
     year = {2012}
    }

    :return: float: I(X;Y|Z), mutual information between X and Y, given Z.
        That is, the information still shared between X and Y after the value of a third variable, Z, is revealed.

    """

    use = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(use):
        return np.NaN

    # Boolean indexing creates copies
    x = x[use]
    y = y[use]
    z = z[use]

    if percentile_range is not None:
        zmin, zmax = np.percentile(z, percentile_range)
    else:
        zmin = np.min(z)
        zmax = np.max(z)
    if zmin >= zmax:
        return np.NaN

    if isinstance(bins, int):
        bins = [bins]

    # For all numbers of bins do a binning of z
    ci = np.empty(len(bins, ))
    for i, b in enumerate(bins):
        edges = np.linspace(zmin, zmax, b)
        indices = np.digitize(z, edges, right=False)
        mi = 0.0
        for k in range(1, len(edges) + 1):
            in_bin = indices == k
            if np.any(in_bin):
                pz = np.count_nonzero(in_bin) / len(z)
                mi += pz * mutual_information(x[in_bin], y[in_bin], bins=bins, percentile_range=percentile_range)
        ci[i] = mi
    return np.mean(ci)
