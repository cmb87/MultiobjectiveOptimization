import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import functools

from src.optimizer.pareto import Pareto


### Generic Optimizer class ###
class Optimizer(object):

    ### Constructor ###
    def __init__(self, fct, xbounds, ybounds, cbounds=[], epsDominanceBins=None, **kwargs):

        self.fct = fct
        self.currentIteration = 0
        self.nparas = len(xbounds)
        self.ntrgts = len(ybounds)
        self.ncstrs = len(cbounds)

        self.xlb, self.xub = np.asarray([x[0] for x in xbounds]), np.asarray([x[1] for x in xbounds])
        self.ylb, self.yub = np.asarray([x[0] for x in ybounds]), np.asarray([x[1] for x in ybounds])
        self.clb, self.cub = np.asarray([x[0] for x in cbounds]), np.asarray([x[1] for x in cbounds])

        self.Xbest, self.Ybest = np.zeros((0, self.xlb.shape[0])), np.zeros((0, self.ylb.shape[0]))

        ### Sanity checks ###
        assert np.any(self.xlb<self.xub), "X: Lower bound must be smaller than upper bound"
        assert np.any(self.ylb<self.yub), "Y: Lower bound must be smaller than upper bound"
        if self.clb.shape[0]>0:
            assert np.any(self.clb<self.cub), "C: Lower bound must be smaller than upper bound"

        ### Database ###
        self.paralabels=["para_{}".format(p) for p in range(self.nparas)]
        self.trgtlabels=["trgt_{}".format(p) for p in range(self.ntrgts)]
        self.cstrlabels=["cstr_{}".format(p) for p in range(self.ncstrs)]

        ### Eps dominace ###
        self.epsDominanceBins = epsDominanceBins

        ### Keywordarguments ###
        self.kwargs = kwargs


    ### Evaluate function ###
    def evaluate(self, X):
        ### Evaluate toolchain ###
        output = self.fct(X, **self.kwargs)
        Y = output[0].reshape(X.shape[0], self.ntrgts)
        C = output[1].reshape(X.shape[0], self.ncstrs) if self.ncstrs>0 else np.zeros((X.shape[0], self.ncstrs))


        ### Build penalty function ###
        Py = Optimizer.boundaryCheck(Y, self.ylb, self.yub)
        Px = Optimizer.boundaryCheck(X, self.xlb, self.xub)
        Pc = Optimizer.boundaryCheck(C, self.clb, self.cub)
        P  = (Py+Px+Pc)

        ### Return to optimizer ###
        return Y, C, P

    ### Initialize ###
    def initialize(self):
        self.currentIteration = 0

    ### Check boundary violation and penalizis it ###
    @staticmethod
    def boundaryCheck(Y, ylb, yub):
        Y = Optimizer._nondimensionalize(Y, ylb, yub)
        index = (Y<0) | (Y>1.0)
        Y[~index] = 0.0
        return np.sum(Y**2,axis=1).reshape(-1,1) 


    ### Epsilon Dominance ###
    def epsDominance(self):
        bins = np.linspace(0,1, self.epsDominanceBins)
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

    ### Restart ###
    @staticmethod
    def load(filename=".optimizerBackup.pkl"):
        with open(filename,'rb') as f1:
            datalist = pickle.load(f1)
        return datalist

    ### Backup ###
    @staticmethod
    def store(datalist, filename=".optimizerBackup.pkl"):
        with open(filename,'wb') as f1:
            pickle.dump(datalist,f1)

