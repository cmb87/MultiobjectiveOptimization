import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import functools
from datetime import datetime

from src.optimizer.pareto import Pareto


### Generic Optimizer class ###
class Optimizer(object):

    ### Constructor ###
    def __init__(self, fct, xbounds, ybounds, cbounds=[], epsDominanceBins=None, optidir='./.myOpti' , args=(), parallel=False):

        self.fct = fct
        self.currentIteration = 0
        self.parallel = parallel

        self.xdim = len(xbounds)
        self.ydim = len(ybounds)
        self.cdim = len(cbounds)

        self.xlb, self.xub = np.asarray([x[0] for x in xbounds]), np.asarray([x[1] for x in xbounds])
        self.ylb, self.yub = np.asarray([x[0] for x in ybounds]), np.asarray([x[1] for x in ybounds])
        self.clb, self.cub = np.asarray([x[0] for x in cbounds]), np.asarray([x[1] for x in cbounds])

        self.xbest, self.ybest, self.cbest  = np.zeros((0, self.xdim )), np.zeros((0, self.ydim )), np.zeros((0, self.cdim ))

        ### Sanity checks ###
        assert np.any(self.xlb<self.xub), "X: Lower bound must be smaller than upper bound"
        assert np.any(self.ylb<self.yub), "Y: Lower bound must be smaller than upper bound"
        if self.clb.shape[0]>0:
            assert np.any(self.clb<self.cub), "C: Lower bound must be smaller than upper bound"

        ### Database ###
        self.paralabels=["para_{}".format(p) for p in range(self.xdim)]
        self.trgtlabels=["trgt_{}".format(p) for p in range(self.ydim)]
        self.cstrlabels=["cstr_{}".format(p) for p in range(self.cdim)]

        ### Eps dominace ###
        self.epsDominanceBins = epsDominanceBins

        ### Store opti dir and name ###
        self.optidir = optidir

        ### Keywordarguments ###
        self.args = args


    ### Evaluate function ###
    def evaluate(self, X):
        ### Evaluate toolchain ###
        output = self.fct(X, *self.args)

        Y = output[0].reshape(X.shape[0], self.ydim)
        C = output[1].reshape(X.shape[0], self.cdim) if self.cdim>0 else np.zeros((X.shape[0], self.cdim))

        ### Build penalty function ###
        Py = Optimizer.boundaryCheck(Y, self.ylb, self.yub)
        Px = Optimizer.boundaryCheck(X, self.xlb, self.xub)
        Pc = Optimizer.boundaryCheck(C, self.clb, self.cub)
        P  = (Py+Px+Pc)

        ### Return to optimizer ###
        return Y, C, P

    ### Initialize ###
    def initialize(self):
        os.makedirs(self.optidir, exist_ok=True)
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

        for n in range(self.ybest.shape[0]):
            Ydim = Optimizer._nondimensionalize(self.ybest[n,:], self.ylb, self.yub)
            
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

        self.ybest = np.delete(self.ybest,index2delete,axis=0)
        self.xbest = np.delete(self.xbest,index2delete,axis=0)
        

    @staticmethod
    def _nondimensionalize(x, lb, ub):
        return (x - lb) / (ub - lb)

    @staticmethod
    def _dimensionalize(x, lb, ub):
        return lb + x * (ub - lb)

    ### Restart ###
    def load(self):
        if not os.path.isdir(self.optidir):
            return
        elif len(os.listdir(self.optidir)) == 0:
            return 
        with open(self.returnLatest(),'rb') as f1:
            datalist = pickle.load(f1)
        return datalist

    ### Backup ###
    def store(self, datalist):
        with open(os.path.join(self.optidir, datetime.now().strftime("%Y%m%d-%H%M%S")+'.pkl'), 'wb') as f1:
            pickle.dump(datalist,f1)

    ### List backupfiles and return latest file ###
    def returnLatest(self):
        return os.path.join(self.optidir, sorted(os.listdir(self.optidir))[-1])