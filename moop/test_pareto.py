import numpy as np
import matplotlib.pyplot as plt
from .optimizer.pareto import Pareto

Y = np.random.rand(100, 2)

ranks = Pareto.computeParetoRanks(Y)
plt.scatter(Y[:, 0], Y[:, 1], c=ranks)
plt.show()


if False:
    paretoIndex, dominatedIndex = Pareto.computeParetoOptimalMember(Y)
    Ypareto = Y[paretoIndex, :]

    plt.plot(Y[:, 0], Y[:, 1], "o")
    plt.plot(Ypareto[:, 0], Ypareto[:, 1], "ro")
    plt.show()
