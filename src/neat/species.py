import numpy as np
import json
import re
import copy
import networkx as nx
import matplotlib.pyplot as plt
import random

from activations import ACTIVATIONS
from innomanager import InnovationManager



class Specie:

    ### Constructor ###
    def __init__(self, nids, nids_input, nids_output, structure, maxtimelevel=2,
                 _id=None, generation=0, parents=[None,None], crossover=True, iter_survived=0):

        ### Structural parameters ###
        self.nids = nids
        self.structure = structure
        self.maxtimelevel = maxtimelevel

        assert len(self.structure)  == len(self.nids), "Length of structure must match length of Nids!"
        ### Previous nodestate ###
        self.last_states = []

        ### For prediction ###
        self.nids_input  = nids_input
        self.nids_output = nids_output
        self.nids_input_index  = [self.nids.index(nid) for nid in nids_input]
        self.nids_output_index = [self.nids.index(nid) for nid in nids_output]

        ### For evolution ###
        self._id = _id
        self.generation = generation
        self.parents = parents
        self.crossover = crossover
        self.iter_survived = iter_survived


    ### to Json ###
    def json(self):
        ### Note that the innvation manager is explicitly not added here! ###
        return {"_id": self._id, "nids": self.nids, "structure": self.structure, "maxtimelevel":self.maxtimelevel,
                "nids_input": self.nids_input, "nids_output": self.nids_output, "generation": self.generation,
                "parents": self.parents, "crossover": self.crossover, "iter_survived": self.iter_survived,
                }

    ### print detailed structure ###
    def show(self):
        return "<{}>".format(json.dumps(self.json(), indent=4, sort_keys=True))

    ### When printing out ###
    def __repr__(self):
        return "<Specie ID {}, created generation {}, survived {}>".format(self._id, self.generation, self.iter_survived)

    ### Check for innovations ###
    def checkInnovations(self):
        if True:
            ### Innovation manager is specified ###  
            for nid in self.nids:
                self.structure[nid]["connections"]["innovations"] = []
                snids, levels = self.structure[nid]["connections"]["snids"], self.structure[nid]["connections"]["level"]
                for snid, level in zip(snids, levels):
                    self.structure[nid]["connections"]["innovations"].append(innovationManager.add(snid, nid, level))
        else:
            print("Global innovationManger is not defined!")


    ### Make a forward prediction ###
    def run(self, X):

        ### Check input ###
        assert X.shape[1] == len(self.nids_input_index), "Input nodes must match dimesion of X!"

        ### Initialize node states ###
        node_states = np.zeros((X.shape[0], len(self.nids), self.maxtimelevel+1))

        ### Initialiaze previous node state for recurrent connections ###
        if len(self.last_states)>0:
            for t, last_state in enumerate(self.last_states):
                node_states[:,:,t+1] = last_state

        ### Assign values to input nodes ###
        node_states[:,self.nids_input_index, 0] = X

        ### Forward propagation ###
        for nid_index, nid in enumerate(self.nids):

            snids  = self.structure[nid]["connections"]["snids"]
            weights  = self.structure[nid]["connections"]["weights"]
            level = self.structure[nid]["connections"]["level"] # The time level of the connection
            bias = self.structure[nid]["bias"]
            fct = ACTIVATIONS[self.structure[nid]["activation"]]

            snids_index = [self.nids.index(snid) for snid in snids]

            ### Calculate node state value ###
            if len(snids) > 0:
                assert nid_index>max([snid for l, snid in zip(level,snids_index) if l == 0]), "Network is not feedforward!"
                node_states[:,nid_index, 0] += fct(np.sum(np.asarray(weights)* node_states[:, snids_index, level], axis=1) + bias)

            ### The node seems not to have any input ###
            else:
                if not nid_index in self.nids_input_index:
                    print("Warning: Node {} seems not to be used".format(nid))
                continue

        ### Store node state (for recurrent connections) ###
        self.last_states.insert(0, node_states[:,:,0].copy())
        self.last_states = self.last_states[:self.maxtimelevel]

        ### Return output nodes values ###
        return node_states[:,self.nids_output_index, 0].reshape(X.shape[0],len(self.nids_output_index))

    ### Remove node ###
    @classmethod
    def mutate_remove_node(cls, specie1, generation=None):
        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)
        nid_remove = nids[np.random.randint(len(specie1.nids_input), len(nids)-len(specie1.nids_output))]
    
        ### Remove node ###
        nids.remove(nid_remove)
        del structure[nid_remove]

        ### Remove connections ###
        for nid in nids:
            if nid_remove in structure[nid]["connections"]["snids"]:
                index = structure[nid]["connections"]["snids"].index(nid_remove)
                structure[nid]["connections"]["innovations"].pop(index)
                structure[nid]["connections"]["weights"].pop(index)
                structure[nid]["connections"]["snids"].pop(index)
                structure[nid]["connections"]["level"].pop(index)

        print("Node {} removed".format(nid_remove))
        return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                   maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id])

    ### Add node ###
    @classmethod
    def mutate_add_node(cls, specie1, value_in=0.1, value_out=0.1, generation=None):

        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)
        nid_new = innovationManager.getNewNodeID()

        inos = {ino:nid for nid in specie1.nids for ino in specie1.structure[nid]["connections"]["innovations"]}
        inomut = list(inos.keys())[np.random.randint(0, len(inos))]

        innovation = innovationManager.getNodes(inomut)
        [snid, rnid], level = innovation["nodes"], innovation["level"]

        ### Inserting new point ###
        for nid_index, nid in enumerate(nids):
            if nid_index <= nids.index(snid):
                continue
            if nid_index > nids.index(rnid):
                continue
            index_insert = nid_index
            break

        nids.insert(index_insert, nid_new)

        ### Delete old connection ###
        index = structure[inos[inomut]]["connections"]["innovations"].index(inomut)
        structure[inos[inomut]]["connections"]["innovations"].pop(index)
        structure[inos[inomut]]["connections"]["weights"].pop(index)
        structure[inos[inomut]]["connections"]["snids"].pop(index)
        structure[inos[inomut]]["connections"]["level"].pop(index)

        ### Add connection from new node ###
        structure[inos[inomut]]["connections"]["innovations"].append(innovationManager.add(nid_new, rnid, level))
        structure[inos[inomut]]["connections"]["weights"].append(value_out)
        structure[inos[inomut]]["connections"]["snids"].append(nid_new)
        structure[inos[inomut]]["connections"]["level"].append(level)

        ### Add connection to new node ###
        structure[nid_new] = {"connections": {}}
        structure[nid_new]["connections"]["innovations"] = [innovationManager.add(snid, nid_new, level)]
        structure[nid_new]["connections"]["weights"] = [value_in]
        structure[nid_new]["connections"]["snids"] = [snid]
        structure[nid_new]["connections"]["level"] = [level]
        structure[nid_new]["bias"] = 0.0
        structure[nid_new]["activation"] = np.random.randint(0,len(ACTIVATIONS))



        return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                   maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id])


    ### Crossover specie 1 is assumed to be the fitter one ###
    @classmethod
    def crossover(cls, specie1, specie2, generation=None):
        
        inos1 = {ino:nid for nid in specie1.nids for ino in specie1.structure[nid]["connections"]["innovations"]}
        inos2 = {ino:nid for nid in specie2.nids for ino in specie2.structure[nid]["connections"]["innovations"]}

        cinos  = list(set(inos1.keys()) & set(inos2.keys()))
        uinos1 = list(set(inos1.keys()).difference(set(cinos)))

        nids = specie1.nids.copy()

        structure = {nid: {'connections':{"snids":[], "weights":[], "level":[]}, 
                           'activation': None, 'bias': None} for nid in nids}

        ### inputs ###
        for nid in specie1.nids_input:
            structure[nid] = specie1.structure[nid]

        ### Common innovations ###
        for cino in cinos:
            parent = [specie1.structure[inos1[cino]], specie2.structure[inos2[cino]]][np.random.randint(0,2)]
            
            innovation = innovationManager.getNodes(cino)
            snid, level = innovation["nodes"][0], innovation["level"]

            weight = parent["connections"]['weights'][parent["connections"]['snids'].index(snid)]

            structure[inos1[cino]]['connections']['snids'].append(snid)
            structure[inos1[cino]]['connections']['level'].append(level)
            structure[inos1[cino]]['connections']['weights'].append(weight)
            structure[inos1[cino]]['bias'] = parent["bias"]
            structure[inos1[cino]]['activation'] = parent["activation"]

        ### Uncommon (only taken from fitter species 1) ###
        for uino in uinos1:
            structure[inos1[uino]] = specie1.structure[inos1[uino]]


        return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                   maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id, specie2._id])



    ### Use networkx to create a graph ###
    def showGraph(self):
        G=nx.DiGraph(specie=self._id)
        G.add_nodes_from(self.nids)
        for nid_index, nid in enumerate(self.nids):
            snids = self.structure[nid]["connections"]["snids"]
            weights = self.structure[nid]["connections"]["weights"]
            for i, snid in enumerate(snids):
                G.add_edge(snid, nid)
                G[snid][nid]["weight"] = weights[i]

        nx.draw(G, with_labels=True)
        plt.show()


#### TEST #######
if __name__ == "__main__":


    innovationManager = InnovationManager()

    structure = {1: {"connections": {"snids": [], "weights":[],"innovations":[], "level":[]},  "activation": 0, "bias":0},
                 2: {"connections": {"snids": [], "weights":[],"innovations":[], "level":[]},  "activation": 0, "bias":0},
                 3: {"connections": {"snids": [1,2], "weights":[1.0, 1.0], "innovations":[], "level":[0,0]},  "activation": 1, "bias":0},
                 5: {"connections": {"snids": [1,2], "weights":[2.0, 1.0], "innovations":[], "level":[0,0]},  "activation": 1, "bias":0},
                 6: {"connections": {"snids": [3,5], "weights":[1.0, 1.0], "innovations":[], "level":[0,0]},  "activation": 0, "bias":0},
                }

    specie1 = Specie(nids=[1,2,3,5,6], structure=structure, nids_input=[1, 2], nids_output=[6])
    specie1.checkInnovations()


    structure = {1: {"connections": {"snids": [], "weights":[],"innovations":[], "level":[]},  "activation": 0, "bias":0},
                 2: {"connections": {"snids": [], "weights":[],"innovations":[], "level":[]},  "activation": 0, "bias":0},
                 5: {"connections": {"snids": [1,2], "weights":[2.0, 1.0], "innovations":[], "level":[0,0]},  "activation": 1, "bias":0},
                 7: {"connections": {"snids": [1,5], "weights":[2.0, 1.0], "innovations":[], "level":[0,0]},  "activation": 1, "bias":0},
                 6: {"connections": {"snids": [5,7], "weights":[1.0, 1.0], "innovations":[], "level":[0,0]},  "activation": 0, "bias":0},
                }

    specie2 = Specie(nids=[1,2,5,7,6], structure=structure, nids_input=[1, 2], nids_output=[6])
    specie2.checkInnovations()


    specie3 = Specie.crossover(specie1, specie2)
    specie4 = Specie.mutate_add_node(specie1)
    specie5 = Specie.mutate_remove_node(specie4)
    specie6 = Specie.crossover(specie4, specie5)
    
    specie3.showGraph()
    specie5.showGraph()
    # 

    for _ in range(1):
        print("####", _ ,"####")
        X = np.random.rand(10,2)
        y = specie3.run(X)
        print(y)