import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import time
from multiprocessing import Process
from multiprocessing import Pipe 

from src.database import Database

class my_decorator(object):
    def __init__(self, target):
        self.target = target

    def __call__(self, x, i, send_end, kwargs):
        output = self.target(x, **kwargs)
        return send_end.send([i,output])


### Generic toolchain object ###
class Toolchain(object):

    TABLENAME = "toolchain"
    RESPONSEDUMMYVAL = 9000

    ### Constructor ###
    def __init__(self, npara, nrspns, trgtIndex=[0], cstrsIndex=[], xlabels=None, rlabels=None, nparallel=2):
        self.designID, self.iteration = 0,0
        self.nparallel = nparallel

        self.trgtIndex = trgtIndex
        self.cstrsIndex = cstrsIndex

        self.xlabels = ["para_{}".format(p) for p in range(npara)]  if xlabels is None else xlabels
        self.rlabels = ["rspn_{}".format(p) for p in range(nrspns)] if rlabels is None else rlabels

        self.npara = npara
        self.nrspns = nrspns

        assert len(self.xlabels) == self.npara,  "xlabels length must match specified parameter count!"
        assert len(self.rlabels) == self.nrspns, "rlabels length must match specified response count!"
        assert len(self.trgtIndex) >0 , "trgtIndex must have at least one entry!"

        ### Initialize table ###
        self.createTable()

    ### Start simulation/run ###
    def execute(self, X):
        ### Check parameters ###
        assert X.shape[1] == self.npara, "Number of parameter of toolchain not matched!"

        ### Start toolchain ###
        rspns_chain = np.asarray(self.chain(X)).reshape(X.shape[0], -1)

        ### Evaluate response ###
        if rspns_chain.shape[1] == self.nrspns:
            pass
        elif rspns_chain.shape[1] < self.nrspns-1:
            rspns_chain = np.vstack((rspns_chain, Toolchain.RESPONSEDUMMYVAL*np.ones((X.shape(0),self.nrspns-rspns_chain.shape[1]))))
        elif rspns_chain.shape[1] >= self.nrspns:
            raise

        ### Store to DB ###
        self.store(X, rspns_chain)

        ### Increment iteration ctr ###
        self.iteration+=1
        return rspns_chain[:,self.trgtIndex], rspns_chain[:,self.cstrsIndex]


    #############################################
    ### HERE THE CUSTOM TOOLCHAIN IS ADDED ###
    ### Dummmy toolchain ###
    def chain(self, X):
        return  Toolchain.RESPONSEDUMMYVAL*np.ones((X.shape(0),self.nrspns))



    #### DB STUFF ####
    ### Create a DB ###
    def createTable(self):
        columns = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "iter": "INT"} 
        for xlabel in self.xlabels:
            columns[xlabel] = "FLOAT"
        for rlabel in self.rlabels:
            columns[rlabel] = "FLOAT"

        Database.delete_table(Toolchain.TABLENAME)
        Database.create_table(Toolchain.TABLENAME, columns)
        print("Database created!")

    ### Store it ###
    def store(self, X, R):
        it = self.iteration*np.ones((X.shape[0], 1))
        Database.insertMany(Toolchain.TABLENAME, rows=np.hstack((it, X, R)).tolist(), 
                            columnNames=["iter"]+self.xlabels+self.rlabels )


    ### Return ###
    def postprocessReturnAll(self):
        iters = [x["iter"] for x in Database.find(Toolchain.TABLENAME, variables=["iter"])]
        X = np.asarray([[d[k] for k in xlabels] for d in Database.find(Toolchain.TABLENAME, variables=self.xlabels)])
        R = np.asarray([[d[k] for k in ylabels] for d in Database.find(Toolchain.TABLENAME, variables=self.rlabels)])
        return iters, X, R


    ### add to database ###
    def postprocess(self, resdir='./', store=False):
        iters = [x["iter"] for x in Database.find(Toolchain.TABLENAME, variables=["iter"],  distinct=True)]

        columnNames = [x[1] for x in Database.getMetaData(Toolchain.TABLENAME)][2:]
        data_mean= np.zeros((len(iters), len(columnNames)))
        data_max = np.zeros((len(iters), len(columnNames)))
        data_min = np.zeros((len(iters), len(columnNames)))
        data_std = np.zeros((len(iters), len(columnNames)))

        for n, it in enumerate(iters):
            data = np.asarray([[d[k] for k in columnNames] for d in Database.find(Toolchain.TABLENAME, query={"iter":["=",it]})])
            data_mean[n,:] = data.mean(axis=0)
            data_max[n,:] = data.max(axis=0)
            data_min[n,:] = data.min(axis=0)
            data_std[n,:] = data.std(axis=0)


        for n, var in enumerate(columnNames):
            fig, ax1 = plt.subplots()

            for k, (data, style) in enumerate(zip([data_mean, data_max, data_min, data_std], ['k-','b-','r-','--'])):
                if k == 3:
                    ax1.plot(iters,data_mean[:,n]+data[:,n],style, color="gray")
                    ax1.plot(iters,data_mean[:,n]-data[:,n],style, color="gray")
                else:
                    ax1.plot(iters,data[:,n],style, lw=3)

                ax2 = ax1.twinx()

            ax1.set_xlabel("Iterations")
            ax1.set_ylabel(var)
            ax1.grid(True)
           # ax1.set_ylim(lb[n], ub[n])
            if store:
                plt.savefig(os.path.join(resdir, "opti_{}.png".format(var)))
                plt.close()
            else:
                plt.show()


    ### Visualize ###
    def postprocessAnimate(self):
        iters = [x["iter"] for x in Database.find(Toolchain.TABLENAME, variables=["iter"], distinct=True)]
        Xcine, Ycine = [], []

        for n, it in enumerate(iters):
            X = np.asarray([[d[k] for k in self.xlabels] for d in Database.find(Toolchain.TABLENAME, 
                                                                               variables=self.xlabels, 
                                                                               query={"iter":["=",it]})])
            Y = np.asarray([[d[k] for k in self.rlabels] for d in Database.find(Toolchain.TABLENAME, 
                                                                               variables=self.rlabels,
                                                                               query={"iter":["=",it]})])
            Xcine.append(X)
            Ycine.append(Y)

        return np.asarray(Xcine), np.asarray(Ycine)








    ### Start simulation parallel ###
    def executeParallel(self, X):
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