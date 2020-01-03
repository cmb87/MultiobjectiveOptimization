"""Graph module
"""
import numpy as np
import os

import importlib
import inspect
import re
import logging
import matplotlib.pyplot as plt
from typing import Callable, Union, Tuple, NewType

from .database import Database
from .operations import Operation, FunctionOperation
from .placeholders import Placeholder
from .variables import Variable


OperationType = NewType('OperationType', Operation)
PlaceholderType = NewType('PlaceholderType', Placeholder)
VariableType = NewType('VariableType', Variable)

class Graph:

    def __init__(self, graph: dict = {}) -> None:
        """Constructor of the the Graph class

        Parameters
        ----------
        graph : dict, optional
            Dictionary of the graph
        """
        self.graph = graph

    @staticmethod
    def traverse_postorder(operation: OperationType) -> list:
        """Makes sure computations are done in correct order

        Parameters
        ----------
        operation : OperationType
            Description

        Returns
        -------
        list
            Description
        """
        nodes_postorder = []

        def recurse(node):
            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)

        recurse(operation)
        return nodes_postorder


    def run(self, feed_dict: dict, outputOperations: list) -> list:
        """Run the graph

        Parameters
        ----------
        feed_dict : dict
            Description
        outputOperations : list
            Operations to be executed

        Returns
        -------
        list
            Description
        """
        outputs = []
        for outputNode in outputOperations:
            # Get postorder from graph
            nodes_postorder = Graph.traverse_postorder(self.graph[outputNode])

            # Iterate over output nodes
            for node in nodes_postorder:
                if not node.evaluated:
                    # Sanity check
                    if not node.sanity_check():
                        nodes_postorder[-1].output = None
                        break

                    # Check if node is a Placeholder
                    if isinstance(node, Placeholder):
                        node.output = feed_dict[node.nid]

                    # Check if node is a Variable
                    elif isinstance(node, Variable):
                        node.output = node.value
                    # It is a operation
                    else:
                        node.inputs = [input_node.output[con] for input_node,
                                       con in zip(node.input_nodes,
                                       node.input_cons)]

                        # args asterixs to unpack content of node.inputs
                        node.output = node.compute(*node.inputs)
                    # Set evalutation flag to true after successfull execution
                    node.evaluated = True

            # Add computed result to outputs
            outputs.append(nodes_postorder[-1].output)

        # Arm graph for next run
        self.arm()

        return outputs

    def arm(self) -> None:
        """Arm graph again
        """
        for nid, node in self.graph.items():
            node.evaluated = False

    @property
    def placeholders(self) -> dict:
        """Returns all placeholders of graph

        Returns
        -------
        dict
            Dict of placeholders
        """
        phs = {}
        for nid, node in self.graph.items():
            if isinstance(node, Placeholder):
                phs[node.nid] = node.name
        return phs

    @property
    def variables(self) -> dict:
        """Returns all variables of graph

        Returns
        -------
        dict
            Description
        """
        vrs = {}
        for nid, node in self.graph.items():
            if isinstance(node, Variable):
                vrs[node.nid] = node.name
        return vrs

    @property
    def operations(self) -> dict:
        """Returns all operations of graph

        Returns
        -------
        dict
            Description
        """
        phs = {}
        for nid, node in self.graph.items():
            if isinstance(node, Operation):
                phs[node.nid] = node.name
        return phs

    def addNodeToGraph(
        self,
        nodeobjct: Union[VariableType, OperationType, PlaceholderType]
    ) -> None:
        """Add node to the graph

        Parameters
        ----------
        nodeobjct : Union[VariableType, OperationType, PlaceholderType]
            Description
        """
        if nodeobjct.nid in list(self.graph.keys()):
            return
        else:
            self.graph[nodeobjct.nid] = nodeobjct

    def addConnectionToGraph(
        self,
        snid: int,
        rnid: int,
        snid_con: int = 0,
        rnid_con: int = 0
    ) -> None:
        """Add connection to the graph

        Parameters
        ----------
        snid : int
            Sending node ID
        rnid : int
            Receiving node ID
        snid_con : int, optional
            Description
        rnid_con : int, optional
            Description
        """
        if snid in list(self.graph.keys()) and rnid in list(self.graph.keys()):
            self.graph[rnid].addFromNode(self.graph[snid], connector=snid_con)
            self.graph[snid].addToNode(self.graph[rnid], connector=rnid_con)


    def generateGraphFromFlowchart(self, flowchart: dict) -> None:
        """Generate graph from flowchart
        (https://github.com/sdrdis/jquery.flowchart)

        Parameters
        ----------
        flowchart : dict
            Flowchart dictionary (comes from Browser/Client)
        """
        # Convert pipeline to graph
        self.graph, self.modules, self.dependencies = {}, [], []

        for nid, node in flowchart['operators'].items():
            # Figure out dependencies
            if 'dependencies' in node['process']:
                self.dependencies.extend(node['process']['dependencies'])

            # Determine type of node
            if 'module_path' in node['process'] and 'name' in node['process']:
                # Add module to list
                module_path = node['process']['module_path']
                ninputs = len(node['properties']['inputs'])
                noutputs = len(node['properties']['outputs'])
                class_name = node['process']['name']
                process_name = node['properties']['title']
                try:
                    value = node['process']['value']
                except:
                    value = 0

                self.modules.append(module_path)

                # Import class from module
                ImportedClass = getattr(
                    importlib.import_module(module_path),
                    class_name
                )
                node = ImportedClass(
                    nid=nid,
                    name=process_name,
                    noutputs=noutputs,
                    ninputs=ninputs,
                    value=value
                )
                self.addNodeToGraph(node)

        # Set in and output
        for cid, con in flowchart['links'].items():
            snid = str(con['fromOperator'])
            snid_con = int(re.findall('\d+', con['fromConnector'])[0])

            rnid = str(con['toOperator'])
            rnid_con = int(re.findall('\d+', con['toConnector'])[0])
            self.addConnectionToGraph(snid, rnid, snid_con, rnid_con)


    def generateFlowchartFromGraph(self) -> dict:
        """Generate flowchart from graph

        Returns
        -------
        dict
            Flowchart dictionary to be sent to jquery
        """
        flowchart = {'operators': {}, 'links': {}}
        linkctr = 0
        for nid, node in self.graph.items():

            ninputs = node.ninputs
            noutputs = node.noutputs

            process = {'ninputs': ninputs, 'noutputs': noutputs,
                       'module_path': inspect.getmodule(node).__name__,
                       'description': 'Nope', 'name': node.__class__.__name__,
                       'output_dtype': ['float' for i in range(noutputs)],
                       'output_labels': ['z{}'.format(i)
                                         for i in range(noutputs)],
                       'input_labels': ['x{}'.format(i)
                                        for i in range(ninputs)],
                       'value': node.value if 'value' in vars(node) else None
                       }

            properties = {'title': node.__class__.__name__,
                          'inputs': {'input_{}'.format(i): {'label': f"x{i}"}\
                          for i in range(ninputs)},
                          'outputs': {'output_{}'.format(i): {'label': f"z{i}"}\
                          for i in range(noutputs)},
                          }

            flowchart['operators'][nid] = {}
            flowchart['operators'][nid]["top"] = 60
            flowchart['operators'][nid]["left"] = 500 + 10 * int(nid)
            flowchart['operators'][nid]["selected"] = ""
            flowchart['operators'][nid]["properties"] = properties
            flowchart['operators'][nid]["process"] = process

            # Add links
            for i, nodeout in enumerate(node.input_nodes):
                flowchart['links'][str(linkctr)] = {}
                flowchart['links'][str(linkctr)]['fromOperator'] = nodeout.nid
                flowchart['links'][str(linkctr)]['fromConnector'] = f"Output\
                {node.input_cons[i]}"

                flowchart['links'][str(linkctr)]['fromSubConnector'] = '0'
                flowchart['links'][str(linkctr)]['toOperator'] = nid
                flowchart['links'][str(linkctr)]['toConnector'] = f"Input\
                {nodeout.output_cons[i]}"
                flowchart['links'][str(linkctr)]['toSubConnector'] = '0'
                linkctr += 1

        return flowchart



class OptimizationGraph(Graph):

    # Constructor
    def __init__(
        self,
        name: str = "MyOptimization",
        xdim: int = 1,
        rdim: int = 1,
        tindex: list = [0],
        cindex: list = [],
        xlabels: Union[list, None]=None,
        rlabels: Union[list, None]=None,
        output_nid: Union[int, None]=None,
        input_nid: Union[int, None]=None,
        iteration: int = 0
    ) -> None:
        """Special Graph for optimizations (only numeric in and output)

        Parameters
        ----------
        name : str, optional
            Name of the graph
        xdim : int, optional
            Input node dimensions of the graph
        rdim : int, optional
            Output node response dimension of the graph
        tindex : list, optional
            target index for optimization of output node
        cindex : list, optional
            constraint index for optimization of output node
        xlabels : Union[list, None], optional
            Labels of the inputs
        rlabels : Union[list, None], optional
            Labels of the responses
        output_nid : Union[int, None], optional
            Output node ID
        input_nid : Union[int, None], optional
            Input node ID
        iteration : int, optional
            Description
        """
        super().__init__()
        self.rdim = rdim
        self.xdim = xdim
        self.tindex = tindex
        self.cindex = cindex
        self.output_nid = output_nid
        self.input_nid = input_nid
        self.iteration = iteration
        self.name = name
        self.xlabels = ["x{}".format(i) for i in range(self.xdim)] \
        if xlabels is None else xlabels
        self.rlabels = ["r{}".format(i) for i in range(self.rdim)] \
        if rlabels is None else rlabels

        assert len(self.xlabels) == self.xdim, \
        "xlabels must match specified parameter dimension!"
        assert len(self.rlabels) == self.rdim, \
        "rlabels must match specified response  dimension!"

        self.createTable()


    def singleProcessChain(self, function: Callable) -> None:
        """If we are working with a single process

        Parameters
        ----------
        function : Callable
            Description
        """
        placeholder = Placeholder(
            nid="0",
            name="Placeholder",
            noutputs=1
        )

        operation = FunctionOperation(
            nid="1",
            name="OptimizationFct",
            noutputs=1
        )

        operation.function = function

        # Construct graph
        self.addNodeToGraph(placeholder)
        self.addNodeToGraph(operation)
        self.addConnectionToGraph("0", "1", snid_con=0, rnid_con=0)

        self.input_nid, self.output_nid = "0", "1"
        if self.sanityCheck():
            logging.info("Single function graph created!")
        else:
            logging.info("Something weired is going on for the single graph!")


    def sanityCheck(self) -> bool:
        """Sanity check of the graph

        Returns
        -------
        None
            Description
        """
        if not len(self.placeholders) == 1:
            logging.info("There can only be one \
                placeholder for an OptimizationGraph!")
            return False

        if not self.output_nid in \
            [key for key in list(self.operations.keys())]:
            logging.info("Output node ID not found in graph!")
            return False

        if not self.input_nid in \
            [key for key in list(self.placeholders.keys())]:
            logging.info("Input node ID not found in graph!")
            return False

        if not isinstance(self.graph[self.input_nid], Placeholder):
            logging.info("Input node must be a Placeholder!")
            return False

        if not isinstance(self.graph[self.output_nid], Operation):
            logging.info("Output node must be a Operation!")
            return False

        return True


    def run(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run chain, this is what the optimization class gets

        Parameters
        ----------
        X : np.ndarray
            Input design vector

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tupel of targets and constraints
        """
        # Get postorder from graph
        nodes_postorder = Graph.traverse_postorder(self.graph[self.output_nid])
        feed_dict = {self.input_nid: [X]}

        # Iterate over output nodes
        for node in nodes_postorder:
            if not node.evaluated:

                # Check if node is a Placeholder
                if isinstance(node, Placeholder):
                    node.output = feed_dict[node.nid]

                # Check if node is a Variable
                elif isinstance(node, Variable):
                    node.output = node.value
                # It is a operation
                else:
                    node.inputs = [input_node.output[con]
                                   for input_node, con
                                   in zip(node.input_nodes, node.input_cons)
                                   ]
                    node.output = node.compute(*node.inputs)    # args asterixs
                # Set evalutation flag to true after successfull execution
                node.evaluated = True

        # Reshape to numpy array
        R = np.asarray(nodes_postorder[-1].output).reshape(X.shape[0], -1)

        # Store current results in DB
        self.store(X, R)
        # Arm graph for next run
        self.arm()
        # Increase iteration counter
        self.iteration += 1
        # Add computed result to outputs
        return R[:, self.tindex], R[:, self.cindex]


    def createTable(self) -> None:
        """Store toolchain in Mastertable
        """
        columns = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "iter": "INT"}
        for xlabel in self.xlabels:
            columns[xlabel] = "FLOAT"
        for rlabel in self.rlabels:
            columns[rlabel] = "FLOAT"

        Database.delete_table(self.name)
        Database.create_table(self.name, columns)
        logging.info("Database created!")


    def store(self, X: np.ndarray, R: np.ndarray) -> None:
        """Store in database

        Parameters
        ----------
        X : np.ndarray
            Design vector (Graph input)
        R : np.ndarray
            Response vector (Graph output)
        """
        it = self.iteration * np.ones((X.shape[0], 1))
        Database.insertMany(self.name, rows=np.hstack((it, X, R)).tolist(),
                            columnNames=["iter"] + self.xlabels + self.rlabels)

    def postprocessReturnAll(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns array with iterations, design vectors and responses

        Returns
        -------
        param
            Tupel of three array (iters, X and R)
        """
        iters = [x["iter"] for x in Database.find(self.name, variables=["iter"])]
        X = np.asarray([[d[k] for k in xlabels] for d in Database.find(
            self.name,
            variables=self.xlabels
        )])
        R = np.asarray([[d[k] for k in ylabels] for d in Database.find(
            self.name,
            variables=self.rlabels
        )])
        return iters, X, R


    def postprocess(self, resdir: str = './', store:bool = False) -> None:
        """Plotss resutls

        Parameters
        ----------
        resdir : str, optional
            Where to store the plots
        store : bool, optional
            If true plots will be store, else shown.
        """
        iters = [x["iter"] for x in Database.find(
            self.name,
            variables=["iter"],
            distinct=True
        )]

        columnNames = [x[1] for x in Database.getMetaData(self.name)][2:]
        data_mean = np.zeros((len(iters), len(columnNames)))
        data_max = np.zeros((len(iters), len(columnNames)))
        data_min = np.zeros((len(iters), len(columnNames)))
        data_std = np.zeros((len(iters), len(columnNames)))

        for n, it in enumerate(iters):
            data = np.asarray([[d[k] for k in columnNames]
                for d in Database.find(self.name, query={"iter": ["=", it]})])
            data_mean[n, :] = data.mean(axis=0)
            data_max[n, :] = data.max(axis=0)
            data_min[n, :] = data.min(axis=0)
            data_std[n, :] = data.std(axis=0)

        for n, var in enumerate(columnNames):
            fig, ax1 = plt.subplots()

            for k, (data, style) in enumerate(
                zip([data_mean, data_max, data_min, data_std],
                    ['k-', 'b-', 'r-', '--'])
            ):
                if k == 3:
                    ax1.plot(
                        iters,
                        data_mean[:, n] + data[:, n],
                        style,
                        color="gray"
                    )
                    ax1.plot(
                        iters,
                        data_mean[:, n] - data[:, n],
                        style,
                        color="gray"
                    )
                else:
                    ax1.plot(iters, data[:, n], style, lw=3)

                ax2 = ax1.twinx()

            ax1.set_xlabel("Iterations")
            ax1.set_ylabel(var)
            ax1.grid(True)

            if store:
                plt.savefig(os.path.join(resdir, "opti_{}.png".format(var)))
                plt.close()
            else:
                plt.show()


    def postprocessAnimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocessing class needed to create animations

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tupel of design and target vectors
        """
        iters = [x["iter"] for x in Database.find(
            self.name,
            variables=["iter"],
            distinct=True
        )]

        Xcine, Ycine = [], []
        for n, it in enumerate(iters):
            X = np.asarray([[d[k] for k in self.xlabels]
                for d in Database.find(
                    self.name,
                    variables=self.xlabels,
                    query={"iter": ["=", it]}
            )])

            Y = np.asarray([[d[k] for k in self.rlabels]
                for d in Database.find(
                    self.name,
                    variables=self.rlabels,
                    query={"iter": ["=", it]}
            )])

            Xcine.append(X)
            Ycine.append(Y)

        return np.asarray(Xcine), np.asarray(Ycine)
