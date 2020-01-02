import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool
import sys
import os
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.pareto import Pareto



Y = np.random.rand(100,2)

ranks = Pareto.computeParetoRanks(Y)
plt.scatter(Y[:,0],Y[:,1],c=ranks)
plt.show()

sys.exit()
paretoIndex, dominatedIndex = Pareto.computeParetoOptimalMember(Y)
Ypareto = Y[paretoIndex,:]

plt.plot(Y[:,0],Y[:,1],'o')
plt.plot(Ypareto[:,0],Ypareto[:,1],'ro')
plt.show()
