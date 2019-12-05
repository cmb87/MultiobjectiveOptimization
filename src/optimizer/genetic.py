import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import functools

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from src.optimizer.pareto import Pareto
from src.optimizer.optimizer import Optimizer

class GA(Optimizer):

    ### Constructor ###
    def __init__(self, fct, xbounds, ybounds, cbounds=[], npop=20, nichingDistanceY=0.1, nichingDistanceX=0.1, parallel=False, 
                 epsDominanceBins=6, optidir='./.myopti', args=()):
        super().__init__(fct, xbounds, ybounds, cbounds=cbounds, epsDominanceBins=epsDominanceBins, parallel=parallel, optidir=optidir, args=args)

        self.xranked, self.xbest = np.zeros((0, self.xdim)), np.zeros((0, self.xdim))
        self.yranked, self.ybest = np.zeros((0, self.ydim)), np.zeros((0, self.ydim))
        self.cranked, self.cbest = np.zeros((0, self.cdim)), np.zeros((0, self.cdim))
        self.pranked = np.zeros((0, 1))
        self.npop = npop

    ### Iterate ###
    def iterate(self, itermax, mut=0.5, crossp=0.7):

        x = Optimizer._dimensionalize(np.random.rand(self.npop, self.xdim), self.xlb, self.xub)

        ### Start iterating ###
        for n in range(itermax):
            self.currentIteration += 1
            print(" Episode : {}".format(self.currentIteration))

            ### Evaluate it ###
            y, c, p = self.evaluate(x)

            ### Append value to so far best seen designs ###
            xNorm = Optimizer._nondimensionalize(np.vstack((x, self.xranked)), self.xlb, self.xub )
            yNorm = Optimizer._nondimensionalize(np.vstack((y, self.yranked)), self.ylb, self.yub )
            c = np.vstack((c, self.cranked))
            p = np.vstack((p, self.pranked))

            ### Pareto Ranks ###
            ranks = Pareto.computeParetoRanks(yNorm+p)
            idxs = np.argsort(ranks)

            ### Sort by rank ###
            xNorm, yNorm, ranks, p, c = xNorm[idxs, :], yNorm[idxs, :], ranks[idxs], p[idxs], c[idxs, :]

            self.xranked = Optimizer._dimensionalize(xNorm[:self.npop,:], self.xlb, self.xub )
            self.yranked = Optimizer._dimensionalize(yNorm[:self.npop,:], self.ylb, self.yub )
            self.cranked = c[:self.npop, :]
            self.pranked = p[:self.npop, :]

            ### Store best values ###
            self.xbest = self.xranked[ranks[:self.npop]<1,:]
            self.ybest = self.yranked[ranks[:self.npop]<1,:]
            self.cbest = self.cranked[ranks[:self.npop]<1,:]

            print("mean",x.mean(), "std",x.std())
            
            ### Differential Evolution ###
            x = np.zeros((self.npop, self.xdim))
            for j in range(self.npop):
                idxs = [i for i in range(xNorm.shape[0]) if i != j]
                a, b, c = xNorm[np.random.choice(idxs, 3, replace = False)]
                mutant = np.clip(a + mut * (b - c), 0, 1)
                cross_points = np.random.rand(self.xdim) < crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.xdim)] = True
                x[j,:] = Optimizer._dimensionalize(np.where(cross_points, mutant, xNorm[j,:]), self.xlb, self.xub)

            #print(x)
            ### Store ###
            self.store([self.currentIteration, self.xranked, self.yranked, self.pranked])
            

    ### Restart algorithm ###
    def restart(self):
        [self.currentIteration, self.xranked, self.yranked, self.pranked] = self.load()

    ### Epsilon Dominance ###
    def epsDominance(self):
        bins = np.linspace(0,1, self.epsDominanceBins)
        binDistance, index2delete = {}, []

        for n in range(self.yranked.shape[0]):
            Ydim = Optimizer._nondimensionalize(self.yranked[n,:], self.ylb, self.yub)
            
            inds = np.digitize(Ydim, bins)
            
            inds_key = '-'.join(map(str,inds))
            dist = sum([(Ydim[i]-bins[inds[i]-1])**2 for i in range(self.ydim)])

            if not inds_key in list(binDistance.keys()):
                binDistance[inds_key] = [dist, n]
            else:
                if binDistance[inds_key][0] < dist:
                    index2delete.append(n)
                else:
                    index2delete.append(binDistance[inds_key][1])
                    binDistance[inds_key][0] = dist
                    binDistance[inds_key][1] = n

        self.yranked = np.delete(self.yranked,index2delete,axis=0)
        self.xranked = np.delete(self.xranked,index2delete,axis=0)
        


if __name__ == "__main__":
    def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
        dimensions = len(bounds)
        pop = np.random.rand(popsize, dimensions)
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([fobj(ind) for ind in pop_denorm])

        best_idx = np.argmin(fitness)
        #print(best_idx, np.argmin(fitness))
        best = pop_denorm[best_idx]
        for i in range(its):
            for j in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
                mutant = np.clip(a + mut * (b - c), 0, 1)
                cross_points = np.random.rand(dimensions) < crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutant, pop[j])
               
                trial_denorm = min_b + trial * diff
                f = fobj(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm

            yield best, fitness[best_idx]

    for d in [8]:
        print("Running d={}".format(d))
        it = list(de(lambda x: sum(x**2)/d, [(-100, 100)] * d, its=3000))    
        x, f = zip(*it)
        plt.plot(f, label='d={}'.format(d))
    plt.legend()
    plt.show()