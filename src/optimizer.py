import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import functools
from multiprocessing import Process
from multiprocessing import Pipe 

from src.particle import Particle
from src.pareto import Pareto
from src.database import Database


class my_decorator(object):
    def __init__(self, target):
        self.target = target

    def __call__(self, x, i, send_end, kwargs):
        output = self.target(x, **kwargs)
        return send_end.send([i,output])


### Generic Optimizer class ###
class Optimizer(object):

    TABLENAME = "Optimization"
    TABLEBEST = "OptimizationBest"

    ### Constructor ###
    def __init__(self, fct, xbounds, ybounds, cbounds=[], nichingDistanceY=0.1,
                  nichingDistanceX=0.1, epsDominanceBins=6, nparallel=1, **kwargs):

        self.fct = fct
        self.currentIteration = 0
        self.nparas = len(xbounds)
        self.ntrgts = len(ybounds)
        self.ncstrs = len(cbounds)

        self.xlb, self.xub = np.asarray([x[0] for x in xbounds]), np.asarray([x[1] for x in xbounds])
        self.ylb, self.yub = np.asarray([x[0] for x in ybounds]), np.asarray([x[1] for x in ybounds])
        self.clb, self.cub = np.asarray([x[0] for x in cbounds]), np.asarray([x[1] for x in cbounds])

        self.Xbest, self.Ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))

        ### Niching parameters ###
        self.delta_y = nichingDistanceY
        self.delta_x = nichingDistanceX
        self.epsDominanceBins = epsDominanceBins

        ### Sanity checks ###
        assert np.any(self.xlb<self.xub), "X: Lower bound must be smaller than upper bound"
        assert np.any(self.ylb<self.yub), "Y: Lower bound must be smaller than upper bound"
        if self.clb.shape[0]>0:
            assert np.any(self.clb<self.cub), "C: Lower bound must be smaller than upper bound"

        ### Database ###
        self.paralabels=["para_{}".format(p) for p in range(self.nparas)]
        self.trgtlabels=["trgt_{}".format(p) for p in range(self.ntrgts)]
        self.cstrlabels=["cstr_{}".format(p) for p in range(self.ncstrs)]

        self.nparallel=nparallel
 
        ### Keywordarguments ###
        self.kwargs = kwargs

    ### Evaluate function ###
    def evaluate(self, X):

        ### Evaluate toolchain ###
        if self.nparallel == 1:
            output = [self.fct(X[i,:], **self.kwargs) for i in range(X.shape[0])]
            Y = np.asarray([x[0] for x in output]).reshape(X.shape[0], self.ntrgts)
            if self.ncstrs>0:
                C = np.asarray([x[1] for x in output]).reshape(X.shape[0], self.ncstrs)
            else:
                C = np.zeros((X.shape[0], self.ncstrs))
        else:
            ### Initialize parallel processes ###
            processes, pipe_list = [], []
            Y,C = X.shape[0]*[0], X.shape[0]*[0]

            ### loop over different designs ###
            for i in range(X.shape[0]):
                
                x = X[i,:]
                ### Run WFF ###
                recv_end, send_end = Pipe(False)

                p = Process(target=my_decorator(self.fct), args=(x, i, send_end, self.kwargs))
                processes.append(p)
                pipe_list.append(recv_end)
                p.start()

                loop = (self.nparallel-1) if i<(X.shape[0]-1) else 0

                while len(processes) > loop:
                    for proc, pipe in zip(processes, pipe_list):
                        if not proc.is_alive():
                            [idesign, output] = pipe.recv()
                            Y[idesign], C[idesign] = output[0], output[1]
                            processes.remove(proc)
                            pipe_list.remove(pipe)


            ### Reshape ###
            Y = np.asarray(Y).reshape(X.shape[0], self.ntrgts)
            if self.ncstrs == 0:
                C = np.zeros((X.shape[0], self.ncstrs))
            else:
                C = np.asarray(C).reshape(X.shape[0], self.ncstrs)

        ### Build penalty function ###
        Py = Optimizer.boundaryCheck(Y, self.ylb, self.yub)
        Px = Optimizer.boundaryCheck(X, self.xlb, self.xub)
        Pc = Optimizer.boundaryCheck(C, self.clb, self.cub)
        P  = (Py+Px+Pc)

        ### Return to optimizer ###
        return Y, C, P

    ### Store it ###
    def store(self, X, Y, C, P):
        it = self.currentIteration*np.ones((X.shape[0], 1))
        Database.insertMany(Optimizer.TABLENAME, rows=np.hstack((it, X, Y, C, P)).tolist(), 
                            columnNames=["iter"]+self.paralabels+self.trgtlabels+self.cstrlabels+["penalty"])

        it = self.currentIteration*np.ones((self.Xbest.shape[0], 1))
        Database.insertMany(Optimizer.TABLEBEST, rows=np.hstack((it, self.Xbest, self.Ybest)).tolist(), 
                            columnNames=["iter"]+self.paralabels+self.trgtlabels)

    ### Initialize ###
    def initialize(self):
        ### Initialize table ###
        self.createTable()
        self.currentIteration = 0

    ### Check boundary violation and penalizis it ###
    @staticmethod
    def boundaryCheck(Y, ylb, yub):
        Y = Optimizer._nondimensionalize(Y, ylb, yub)
        index = (Y<0) | (Y>1.0)
        Y[~index] = 0.0
        return np.sum(Y**2,axis=1).reshape(-1,1) 

    ### Update leader particle without external archive ###
    def findParetoMembers(self, X, Y, P):
        ### Only designs without penality violation ###
        validIndex = P[:,0] == 0.0
        
        ### Some designs are valid ###
        if P[validIndex].shape[0] > 0:

            # Calculate pareto ranks
            X = np.vstack((self.Xbest, X[validIndex,:]))
            Y = np.vstack((self.Ybest, Y[validIndex,:]))

            paretoIndex, dominatedIndex = Pareto.computeParetoOptimalMember(Y)
            Xpareto, Ypareto = X[paretoIndex,:], Y[paretoIndex,:]
            Ppareto = np.zeros((Xpareto.shape[0],1))
            ### Update best particle archive ###                                                
            self.Xbest, self.Ybest = Xpareto, Ypareto

            ### Probality of being a leader ###
            py = Optimizer.neighbors(Ypareto, self.delta_y)
            px = Optimizer.neighbors(Xpareto, self.delta_x)
            pvals = py*px

        ### All designs have a penalty ==> Move to point with lowest violation ###
        else:
            index = np.argmin(P, axis=0)
            Ypareto, Xpareto = Y[index, :].reshape(1,self.ntrgts), X[index, :].reshape(1,self.nparas)
            Ppareto = np.zeros((Xpareto.shape[0],1))
            pvals = np.ones(1)

        return  Xpareto, Ypareto, Ppareto, pvals/pvals.sum()


    @staticmethod
    def neighbors(X, delta):
        D,N = np.zeros((X.shape[0],X.shape[0])), np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i,j] = np.linalg.norm(X[i,:]-X[j,:])

        N[D<delta] = 1.0
        del D
        return 1.0/np.exp(N.sum(axis=0))


    ### Epsilon Dominance ###
    def epsDominance(self):
        bins = np.linspace(0,1,self.epsDominanceBins)
        binDistance, index2delete = {}, []

        for n in range(self.Ybest.shape[0]):
            Ydim = Optimizer._nondimensionalize(self.Ybest[n,:], self.ylb, self.yub)
            
            inds = np.digitize(Ydim, bins)
            
            inds_key = '-'.join(map(str,inds))
            dist = sum([(Ydim[i]-bins[inds[i]-1])**2 for i in range(self.ntrgts)])

            if not inds_key in list(binDistance.keys()):
                binDistance[inds_key] = [dist, n]
            else:
                if binDistance[inds_key][0] < dist:
                    index2delete.append(n)
                else:
                    index2delete.append(binDistance[inds_key][1])
                    binDistance[inds_key][0] = dist
                    binDistance[inds_key][1] = n

        self.Ybest = np.delete(self.Ybest,index2delete,axis=0)
        self.Xbest = np.delete(self.Xbest,index2delete,axis=0)
        

    @staticmethod
    def _nondimensionalize(x, lb, ub):
        return (x - lb) / (ub - lb)

    @staticmethod
    def _dimensionalize(x, lb, ub):
        return lb + x * (ub - lb)

    ### Create table ###
    def createTable(self):
        columns = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "iter": "INT"} 
        for xlabel in self.paralabels:
            columns[xlabel] = "FLOAT"
        for ylabel in self.trgtlabels:
            columns[ylabel] = "FLOAT"
        for clabel in self.cstrlabels:
            columns[clabel] = "FLOAT"
        columns["penalty"] = "FLOAT"

        Database.delete_table(Optimizer.TABLENAME)
        Database.create_table(Optimizer.TABLENAME, columns)

        ### Table for Pareto members ###
        columns = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "iter": "INT"} 
        for xlabel in self.paralabels:
            columns[xlabel] = "FLOAT"
        for ylabel in self.trgtlabels:
            columns[ylabel] = "FLOAT"

        Database.delete_table(Optimizer.TABLEBEST)
        Database.create_table(Optimizer.TABLEBEST, columns)
        print("Database created!")


    ### Restart ###
    def restart(self):
        self.Xbest, self.Ybest = self.postprocessReturnBest()


    ### Get names ###
    @staticmethod
    def splitColumnNames():
        columnNames = [x[1] for x in Database.getMetaData(Optimizer.TABLENAME)][2:]
        xlabels, ylabels, clabels = [],[],[]
        for name in columnNames:
            if name[:4] == "para":
                xlabels.append(name)
            elif name[:4] == "trgt":
                ylabels.append(name)
            elif name[:4] == "cstr":
                clabels.append(name)
        return xlabels, ylabels, clabels

    ### Return ###
    @staticmethod
    def postprocessReturnAll():
        iters = [x["iter"] for x in Database.find(Optimizer.TABLENAME, variables=["iter"])]
        xlabels, ylabels, clabels = Optimizer.splitColumnNames()

        X = np.asarray([[d[k] for k in xlabels] for d in Database.find(Optimizer.TABLENAME, variables=xlabels)])
        Y = np.asarray([[d[k] for k in ylabels] for d in Database.find(Optimizer.TABLENAME, variables=ylabels)])
        C = np.asarray([[d[k] for k in clabels] for d in Database.find(Optimizer.TABLENAME, variables=clabels)])
        P = np.asarray([[d[k] for k in ["penalty"]] for d in Database.find(Optimizer.TABLENAME, variables=["penalty"])])
        return iters, X, Y, C, P


    ### Return averaged values ###
    @staticmethod
    def postprocessReturnStatistics():
        iters_distinct = [x["iter"] for x in Database.find(Optimizer.TABLENAME, variables=["iter"], distinct=True)]
        xlabels, ylabels, clabels = Optimizer.splitColumnNames()
        dataAll = []

        for labels in [xlabels,ylabels,clabels]:

            data_mean= np.zeros((len(iters_distinct), len(labels)))
            data_max = np.zeros((len(iters_distinct), len(labels)))
            data_min = np.zeros((len(iters_distinct), len(labels)))
            data_std = np.zeros((len(iters_distinct), len(labels)))

            for n, it in enumerate(iters_distinct):
                data = np.asarray([[d[k] for k in labels] for d in Database.find(Optimizer.TABLENAME, variables=labels, 
                                                                                    query={"iter":["=",it]})])
                data_mean[n,:] = data.mean(axis=0)
                data_max[n,:] = data.max(axis=0)
                data_min[n,:] = data.min(axis=0)
                data_std[n,:] = data.std(axis=0)

            dataAll.append([data_mean,data_max,data_min,data_std])
        return iters_distinct, dataAll

    ### add to database ###
    def postprocess(self, resdir='./', store=False, xlabel=None, ylabel=None, clabel=None):

        iters = [x["iter"] for x in Database.find(Optimizer.TABLENAME, variables=["iter"], 
                                                  distinct=True)]

        columnNames = [x[1] for x in Database.getMetaData(Optimizer.TABLENAME)][2:]

        data_mean= np.zeros((len(iters), len(columnNames)))
        data_max = np.zeros((len(iters), len(columnNames)))
        data_min = np.zeros((len(iters), len(columnNames)))
        data_std = np.zeros((len(iters), len(columnNames)))

        for n, it in enumerate(iters):
            data = np.asarray([[d[k] for k in columnNames] for d in Database.find(Optimizer.TABLENAME, 
                                                                                  query={"iter":["=",it]})])
            data_mean[n,:] = data.mean(axis=0)
            data_max[n,:] = data.max(axis=0)
            data_min[n,:] = data.min(axis=0)
            data_std[n,:] = data.std(axis=0)

        ### Bounds ###
        lb = self.xlb.tolist()+self.ylb.tolist()+self.clb.tolist()+[0]
        ub = self.xub.tolist()+self.yub.tolist()+self.cub.tolist()+[0]

        clabel = self.cstrlabels if clabel is None else clabel
        xlabel = self.paralabels if xlabel is None else xlabel
        ylabel = self.trgtlabels if ylabel is None else ylabel

        assert len(self.cstrlabels) == len(clabel), "C Label must match dimensions!"
        assert len(self.paralabels) == len(xlabel), "C Label must match dimensions!"
        assert len(self.trgtlabels) == len(ylabel), "C Label must match dimensions!"

        ### Plot it ###
        for n, var in enumerate(xlabel+ylabel+clabel+["penalty"]):
            fig, ax1 = plt.subplots()

            for k, (data, style) in enumerate(zip([data_mean, data_max, data_min, data_std], ['k-','b-','r-','--'])):

                ax1.axhline(y=lb[n], color='r', linestyle='--')
                ax1.axhline(y=ub[n], color='r', linestyle='--')
                if k == 3:
                    ax1.plot(iters,data_mean[:,n]+data[:,n],style, color="gray")
                    ax1.plot(iters,data_mean[:,n]-data[:,n],style, color="gray")
                else:
                    ax1.plot(iters,data[:,n],style, lw=3)

                ax2 = ax1.twinx()
                #ax2.plot(data[:,0],data[:,5],'m-',lw=3)
                #ax2.set_ylabel('Mean Pareto Change')

            ax1.set_xlabel("Iterations")
            ax1.set_ylabel(var)
            ax1.grid(True)
            if store:
                plt.savefig(os.path.join(resdir, "opti_{}.png".format(var)))
                plt.close()
            else:
                plt.show()

    ### Return best values ###
    def postprocessReturnBest(self):
        iters = [x["iter"] for x in Database.find(Optimizer.TABLEBEST, variables=["iter"], 
                                                  distinct=True)]

        Xbest = np.asarray([[d[k] for k in self.paralabels] for d in Database.find(Optimizer.TABLEBEST,
                                                                                  variables=self.paralabels,
                                                                                  query={"iter":["=",iters[-1]]})])

        Ybest = np.asarray([[d[k] for k in self.trgtlabels] for d in Database.find(Optimizer.TABLEBEST,
                                                                                  variables=self.trgtlabels,
                                                                                  query={"iter":["=",iters[-1]]})])

        return Xbest, Ybest, iters[-1]




### Particle Swarm Optimizer ###
class Swarm(Optimizer):

    def __init__(self, fct, xbounds, ybounds, cbounds=[], nparticles=10, nparallel=1,
                 nichingDistanceY=0.1, nichingDistanceX=0.1, epsDominanceBins=6, minimumSwarmSize=10, **kwargs):
        super().__init__(fct, xbounds, ybounds, cbounds=cbounds, nichingDistanceY=nichingDistanceY, 
                         nichingDistanceX=nichingDistanceX, epsDominanceBins=epsDominanceBins, nparallel=nparallel, **kwargs)

        ### Initialize swarm ###
        self.particles = [Particle(self.nparas, pid, vinitScale=0.5) for pid in range(nparticles)]
        self.Xbest, self.Ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))
        self.minimumSwarmSize = minimumSwarmSize
        self.particleReductionRate = 0.99

    ### swarm size ###
    @property
    def swarmSize(self):
        return len(self.particles)
    

    ### Iterate ###
    def iterate(self, itermax):

        ### Start iterating ###
        for n in range(itermax):
            self.currentIteration += 1
            print(" Episode : {}".format(self.currentIteration))

            ### Remove particles ###
            SwarmSizeTarget = int(self.particleReductionRate*self.swarmSize) if self.swarmSize > self.minimumSwarmSize else self.minimumSwarmSize
            while self.swarmSize > SwarmSizeTarget:
                self.particles.pop()

            ### Evaluate the new particle's position ###
            X = np.asarray([Optimizer._dimensionalize(particle.x, self.xlb, self.xub) for particle in self.particles])

            ### Evaluate it ###
            Y, C, P = self.evaluate(X)

            ### Update particle targets ###
            for i, particle in enumerate(self.particles):
                particle.y = Optimizer._nondimensionalize(Y[i, :], self.ylb, self.yub)
                particle.p = P[i, :]

            ### Determine new pbest ###
            [particle.updatePersonalBest() for particle in self.particles]

            ### Determine new gbest ###
            Xpareto, Ypareto, Ppareto, pvals = self.findParetoMembers(X, Y, P)

            for i, particle in enumerate(self.particles):
                index = np.random.choice(np.arange(Xpareto.shape[0]), p=pvals)
                particle.ygbest = Optimizer._nondimensionalize(Ypareto[index, :], self.ylb, self.yub)
                particle.xgbest = Optimizer._nondimensionalize(Xpareto[index, :], self.xlb, self.xub)
                particle.pgbest = Ppareto[index, :]

            ### Apply eps dominance ###
            self.epsDominance()

            ### Update particles ###
            [particle.update() for particle in self.particles]

            ### Store ###
            self.store(X, Y, C, P)
            self._storeSwarm()

    ### Restart ###
    def restart(self, resetParticles=False ,filename="swarm.pkl"):
        self.Xbest, self.Ybest, self.currentIteration = self.postprocessReturnBest()
        
        with open(filename,'rb') as f1:
            self.particles = pickle.load(f1)

        if resetParticles:
            [particle.reset() for particle in self.particles]

    ### Store the swarm ###
    def _storeSwarm(self, filename="swarm.pkl"):
        with open(filename,'wb') as f1:
            pickle.dump(self.particles,f1)


    ### Array for animation ###
    def postprocessAnimate(self):
        iters = [x["iter"] for x in Database.find(Optimizer.TABLENAME, variables=["iter"], 
                                                  distinct=True)]

        Xcine, Ycine = [], []

        for n, it in enumerate(iters):

            X = np.asarray([[d[k] for k in self.paralabels] for d in Database.find(Optimizer.TABLENAME, 
                                                                               variables=self.paralabels, 
                                                                               query={"iter":["=",it]})])
            Y = np.asarray([[d[k] for k in self.trgtlabels] for d in Database.find(Optimizer.TABLENAME, 
                                                                               variables=self.trgtlabels,
                                                                               query={"iter":["=",it]})])
            Xcine.append(X)
            Ycine.append(Y)

        return np.asarray(Xcine), np.asarray(Ycine)