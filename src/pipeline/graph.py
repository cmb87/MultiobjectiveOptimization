import numpy as np
import os
import sys
import importlib
import inspect 
import time
import json
import re

from multiprocessing import Process
from multiprocessing import Pipe 

from src.common.database import Database
from src.pipeline.operations import Operation, FunctionOperation
from src.pipeline.placeholders import Placeholder
from src.pipeline.variables import Variable

### This is the Graph class ###
class Graph:
    ### Constructor ###
    def __init__(self, graph={}):
        self.graph = graph

    ### makes sure computations are done in correct order ###
    @staticmethod
    def traverse_postorder(operation):
        nodes_postorder = []
        def recurse(node):
            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)

        recurse(operation)
        return nodes_postorder


    ### Run the graph ###
    def run(self, feed_dict, outputOperations):
        outputs = []
        for outputNode in outputOperations:
            ### Get postorder from graph ###
            nodes_postorder = Graph.traverse_postorder(self.graph[outputNode])
            
            ### Iterate over output nodes ###
            for node in nodes_postorder:
                if not node.evaluated:
                    ### Sanity check ###
                    if not node.sanity_check():
                        nodes_postorder[-1].output = None
                        break
  
                    ### Check if node is a Placeholder ###
                    if isinstance(node, Placeholder):
                        node.output = feed_dict[node.nid]

                    ### Check if node is a Variable ###
                    elif isinstance(node, Variable):
                        node.output = node.value
                    ### It is a operation ###
                    else:  
                        node.inputs = [input_node.output[con] for input_node, con in zip(node.input_nodes, node.input_cons)]
                        node.output = node.compute(*node.inputs)    # args asterixs
                    ### Set evalutation flag to true after successfull execution ###
                    node.evaluated = True

            ### Add computed result to outputs ###
            outputs.append(nodes_postorder[-1].output)

        ### Arm graph for next run ###
        self.arm()

        return outputs

    ### Arm graph again ###
    def arm(self):
        for nid, node in self.graph.items():
            node.evaluated = False

    ### Properties ###
    @property
    def placeholders(self):
        phs = {}
        for nid, node in self.graph.items():
            if isinstance(node, Placeholder):
                phs[node.nid] = node.name
        return phs
    
    @property
    def variables(self):
        vrs = {}
        for nid, node in self.graph.items():
            if isinstance(node, Variable):
                vrs[node.nid] = node.name
        return vrs

    @property
    def operations(self):
        phs = {}
        for nid, node in self.graph.items():
            if isinstance(node, Operation):
                phs[node.nid] = node.name
        return phs

    ### Add node ###
    def addNodeToGraph(self, nodeobjct):
        if not nodeobjct.nid in list(self.graph.keys()):
            self.graph[nodeobjct.nid] = nodeobjct

    ### Add connection ###
    def addConnectionToGraph(self, snid, rnid, snid_con=0, rnid_con=0):
        if snid in list(self.graph.keys()) and rnid in list(self.graph.keys()):
            self.graph[rnid].addFromNode(self.graph[snid], connector=snid_con)
            self.graph[snid].addToNode(  self.graph[rnid], connector=rnid_con)


    ### Generate graph from flowchart ###
    def generateGraphFromFlowchart(self, flowchart):
        ### Convert pipeline to graph ###
        self.graph, self.modules, self.dependencies = {}, [], []

        for nid, node in flowchart['operators'].items():
            ### Figure out dependencies ###         
            if 'dependencies' in node['process']:
                self.dependencies.extend(node['process']['dependencies']) 

            ### Determine type of node ###
            if 'module_path' in node['process'] and 'name' in node['process']:
                ### Add module to list ###
                module_path = node['process']['module_path']
                ninputs = len(node['properties']['inputs'])
                noutputs = len(node['properties']['outputs'])
                class_name  = node['process']['name']
                process_name = node['properties']['title']
                try:
                    value = node['process']['value']
                except:
                    value = 0

                self.modules.append(module_path) 

                ### Import class from module ###            
                ImportedClass = getattr(importlib.import_module(module_path), class_name)
                node = ImportedClass(nid=nid, name=process_name, noutputs=noutputs, ninputs=ninputs, value=value)
                self.addNodeToGraph(node)

        ### Set in and output ###
        for cid, con in flowchart['links'].items():
            snid, snid_con = str(con['fromOperator']), int(re.findall('\d+', con['fromConnector'] )[0])
            rnid, rnid_con = str(con['toOperator']),   int(re.findall('\d+', con['toConnector'] )[0])
            self.addConnectionToGraph(snid, rnid, snid_con, rnid_con)


    ### Generate flowchart from graph ###
    def generateFlowchartFromGraph(self):
        flowchart = {'operators':{}, 'links':{}}
        linkctr = 0
        for nid, node in self.graph.items():

            ninputs = node.ninputs
            noutputs = node.noutputs

            process = {'ninputs': ninputs, 'noutputs': noutputs,
                       'module_path': inspect.getmodule(node).__name__,
                       'description': 'Nope', 'name': node.__class__.__name__, 
                       'output_dtype': ['float' for i in range(noutputs)],
                       'output_labels': ['z{}'.format(i) for i in range(noutputs)],
                       'input_labels': ['x{}'.format(i) for i in range(ninputs)],
                       'value': node.value if 'value' in vars(node) else None
                       }

            properties = {'title':  node.__class__.__name__,
                          'inputs' : { 'input_{}'.format(i): {'label': 'x{}'.format(i)} for i in range(ninputs)},
                          'outputs': {'output_{}'.format(i): {'label': 'z{}'.format(i)} for i in range(noutputs)},
                          }

            flowchart['operators'][nid] = {}
            flowchart['operators'][nid]["top"] = 60
            flowchart['operators'][nid]["left"] = 500+10*int(nid)
            flowchart['operators'][nid]["selected"] = ""
            flowchart['operators'][nid]["properties"] = properties
            flowchart['operators'][nid]["process"] = process

            ### Add links ###
            for i, nodeout in enumerate(node.input_nodes):
                flowchart['links'][str(linkctr)] = {}
                flowchart['links'][str(linkctr)]['fromOperator']      =  nodeout.nid
                flowchart['links'][str(linkctr)]['fromConnector']     =  'Output{}'.format(node.input_cons[i])
                flowchart['links'][str(linkctr)]['fromSubConnector']  =  '0'                      
                flowchart['links'][str(linkctr)]['toOperator']        =  nid
                flowchart['links'][str(linkctr)]['toConnector']       =  'Input{}'.format(nodeout.output_cons[i])
                flowchart['links'][str(linkctr)]['toSubConnector']    =  '0'
                linkctr+=1
        
        return flowchart






### Special Graph for optimizations (only numeric in and output) ###
class OptimizationGraph(Graph):

    ### Constructor ###
    def __init__(self, name="MyOptimization", xdim=1, rdim=1, tindex=[0], cindex=[], xlabels=None, 
                 rlabels=None, output_nid=None, input_nid=None, iteration=0):

        super().__init__()
        self.rdim = rdim
        self.xdim = xdim
        self.tindex = tindex
        self.cindex = cindex
        self.output_nid = output_nid
        self.input_nid = input_nid
        self.iteration = iteration
        self.name = name
        self.xlabels = ["x{}".format(i) for i in range(self.xdim)] if xlabels is None else xlabels
        self.rlabels = ["r{}".format(i) for i in range(self.rdim)] if rlabels is None else rlabels

        assert len(self.xlabels) == self.xdim, "xlabels must match specified parameter dimension!"
        assert len(self.rlabels) == self.rdim, "rlabels must match specified response  dimension!"

        self.createTable()

    ### if we are working with a single process ###
    def singleProcessChain(self, function):
        placeholder = Placeholder(nid="0", name="Placeholder", noutputs=1)
        operation = FunctionOperation(nid="1", name="OptimizationFct", noutputs=1)
        operation.function = function

        self.addNodeToGraph(placeholder)
        self.addNodeToGraph(operation)
        self.addConnectionToGraph("0","1", snid_con=0, rnid_con=0)

        self.input_nid, self.output_nid = "0", "1"
        if self.sanityCheck():
            print("Single function graph created!")
        else:
            print("Something weired is going on for the single graph")
        

    ### Sanity check ###
    def sanityCheck(self):
        if not len(self.placeholders) == 1:
            print("There can only be one placeholder for an OptimizationGraph!")
            return False

        if not self.output_nid in [key for key in list(self.operations.keys())]:
            print("Output node ID not found in graph!")
            return False

        if not self.input_nid in [key for key in list(self.placeholders.keys())]:
            print("Input node ID not found in graph!")
            return False

        if not isinstance(self.graph[self.input_nid], Placeholder):
            print("Input node must be a Placeholder!")
            return False

        if not isinstance(self.graph[self.output_nid], Operation):
            print("Output node must be a Operation!")
            return False

        return True

    ### run chain, this is what the optimization class gets ###
    def run(self, X):

        ### Get postorder from graph ###
        nodes_postorder = Graph.traverse_postorder(self.graph[self.output_nid])
        feed_dict = {self.input_nid: [X]}

        ### Iterate over output nodes ###
        for node in nodes_postorder:
            if not node.evaluated:

                ### Check if node is a Placeholder ###
                if isinstance(node, Placeholder):
                    node.output = feed_dict[node.nid]

                ### Check if node is a Variable ###
                elif isinstance(node, Variable):
                    node.output = node.value
                ### It is a operation ###
                else:  
                    node.inputs = [input_node.output[con] for input_node, con in zip(node.input_nodes, node.input_cons)]
                    node.output = node.compute(*node.inputs)    # args asterixs
                ### Set evalutation flag to true after successfull execution ###
                node.evaluated = True

        ### Reshape to numpy array ###
        R = np.asarray(nodes_postorder[-1].output[0]).reshape(X.shape[0], -1)

        ### Store current results in DB ###
        self.store(X, R)
        ### Arm graph for next run ###
        self.arm()
        ### Increase iteration counter ###
        self.iteration +=1
        ### Add computed result to outputs ###
        return R[:,self.tindex], R[:,self.cindex]


    ### Store toolchain in Mastertable ###
    ### Create specific DB for toolchain ###
    def createTable(self):
        columns = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "iter": "INT"} 
        for xlabel in self.xlabels:
            columns[xlabel] = "FLOAT"
        for rlabel in self.rlabels:
            columns[rlabel] = "FLOAT"

        Database.delete_table(self.name)
        Database.create_table(self.name, columns)
        print("Database created!")

    ### Store it ###
    def store(self, X, R):
        it = self.iteration*np.ones((X.shape[0], 1))
        Database.insertMany(self.name, rows=np.hstack((it, X, R)).tolist(), 
                            columnNames=["iter"]+self.xlabels+self.rlabels)




    ### Return ###
    def postprocessReturnAll(self):
        iters = [x["iter"] for x in Database.find(self.name, variables=["iter"])]
        X = np.asarray([[d[k] for k in xlabels] for d in Database.find(self.name, variables=self.xlabels)])
        R = np.asarray([[d[k] for k in ylabels] for d in Database.find(self.name, variables=self.rlabels)])
        return iters, X, R


    ### add to database ###
    def postprocess(self, resdir='./', store=False):
        iters = [x["iter"] for x in Database.find(self.name, variables=["iter"],  distinct=True)]

        columnNames = [x[1] for x in Database.getMetaData(self.name)][2:]
        data_mean= np.zeros((len(iters), len(columnNames)))
        data_max = np.zeros((len(iters), len(columnNames)))
        data_min = np.zeros((len(iters), len(columnNames)))
        data_std = np.zeros((len(iters), len(columnNames)))

        for n, it in enumerate(iters):
            data = np.asarray([[d[k] for k in columnNames] for d in Database.find(self.name, query={"iter":["=",it]})])
            data_mean[n,:] = data.mean(axis=0)
            data_max[n,:] = data.max(axis=0)
            data_min[n,:] = data.min(axis=0)
            data_std[n,:] = data.std(axis=0)

        import matplotlib.pyplot as plt

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
        iters = [x["iter"] for x in Database.find(self.name, variables=["iter"], distinct=True)]
        Xcine, Ycine = [], []
        for n, it in enumerate(iters):
            X = np.asarray([[d[k] for k in self.xlabels] for d in Database.find(self.name, 
                                                                               variables=self.xlabels, 
                                                                               query={"iter":["=",it]})])
            Y = np.asarray([[d[k] for k in self.rlabels] for d in Database.find(self.name, 
                                                                               variables=self.rlabels,
                                                                               query={"iter":["=",it]})])
            Xcine.append(X)
            Ycine.append(Y)

        return np.asarray(Xcine), np.asarray(Ycine)



    # ### Start simulation parallel ###
    # def executeParallel(self, X):
    #     ### Initialize parallel processes ###
    #     processes, pipe_list = [], []
    #     Y,C = X.shape[0]*[0], X.shape[0]*[0]

    #     ### loop over different designs ###
    #     for i in range(X.shape[0]):
            
    #         x = X[i,:]
    #         ### Run WFF ###
    #         recv_end, send_end = Pipe(False)

    #         p = Process(target=my_decorator(self.fct), args=(x, i, send_end, self.kwargs))
    #         processes.append(p)
    #         pipe_list.append(recv_end)
    #         p.start()

    #         loop = (self.nparallel-1) if i<(X.shape[0]-1) else 0

    #         while len(processes) > loop:
    #             for proc, pipe in zip(processes, pipe_list):
    #                 if not proc.is_alive():
    #                     [idesign, output] = pipe.recv()
    #                     Y[idesign], C[idesign] = output[0], output[1]
    #                     processes.remove(proc)
    #                     pipe_list.remove(pipe)


    #     ### Reshape ###
    #     Y = np.asarray(Y).reshape(X.shape[0], self.ntrgts)
    #     if self.ncstrs == 0:
    #         C = np.zeros((X.shape[0], self.ncstrs))
    #     else:
    #         C = np.asarray(C).reshape(X.shape[0], self.ncstrs)


