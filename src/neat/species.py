import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt

from activations import ACTIVATIONS
from innomanager import InnovationManager



class Specie:

    ### Constructor ###
    def __init__(self, nids, nfcts, connections, nids_input, nids_output, recurrent_connections=None, bias=None, _id=None,
                 generation=0, parents=[None,None], crossover=True, iter_survived=0, innovations=[], innovationManager=None):

        ### Structural parameters ###
        self.nids = nids
        self.nfcts = nfcts
        self.bias = len(nids)*[0] if bias is None else bias
        self.connections = connections
        self.recurrent_connections = len(nids)*[[[],[]]] if recurrent_connections is None else recurrent_connections
        
        assert len(self.bias)  == len(self.nids ), "Length of bias must match length of Nids!"
        assert len(self.nfcts) == len(self.nids ), "Length of nodal functions must match length of Nids!"
        assert len(self.connections) == len(self.nids ), "Length of connections must match length of Nids!"
        assert len(self.recurrent_connections) == len(self.nids ), "Length of recurrent connections must match length of Nids!"
        ### Previous nodestate ###
        self.node_states0 = None

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
        self.innovations = innovations
        ### I'll be specified outside ###
        self.innovationManager = innovationManager

    ### to Json ###
    def json(self):
        ### Note that the innvation manager is explicitly not added here! ###
        return {"_id": self._id, "nids": self.nids, "nfcts": self.nfcts, "connections": self.connections, 
                "bias": self.bias, "recurrent_connections": self.recurrent_connections,
                "nids_input": self.nids_input, "nids_output": self.nids_output, "generation": self.generation,
                "parents": self.parents, "crossover": self.crossover, "iter_survived": self.iter_survived,
                "innovations": self.innovations}

    ### print detailed structure ###
    def show(self):
        return "<{}>".format(json.dumps(self.json(), indent=4, sort_keys=True))

    ### When printing out ###
    def __repr__(self):
        return "<Specie ID {}, created generation {}, survived {}>".format(self._id, self.generation, self.iter_survived)

    ### Check for innovations ###
    def checkInnovations(self):
        self.innovations = []
        if self.innovationManager is None:
            print("No InnovationManager specified!")
            return

        ### Innovation manager is specified ###  
        for nid_index, nid in enumerate(self.nids):
            snids,  weights = self.connections[nid_index]
            for i, snid in enumerate(snids):
                self.innovations.append(self.innovationManager.add(snid, nid))


    ### Use networkx to create a graph ###
    def showGraph(self):
        G=nx.DiGraph(specie=self._id)
        G.add_nodes_from(self.nids, fcts=self.nfcts, bias=self.bias)
        for nid_index, nid in enumerate(self.nids):
            snids,  weights = self.connections[nid_index]
            for i, snid in enumerate(snids):
                G.add_edge(snid, nid)
                G[snid][nid]["weight"] = weights[i]

        nx.draw(G, with_labels=True)
        plt.show()



    ### Make a forward prediction ###
    def run(self, X):

        ### Check input ###
        assert X.shape[1] == len(self.nids_input_index), "Input nodes must match dimesion of X!"

        ### Initialize node states ###
        node_states = np.zeros((X.shape[0], len(self.nids)))

        ### Initialiaze previous node state for recurrent connections ###
        self.node_states0 = node_states.copy() if self.node_states0 is None else self.node_states0

        ### Assign values to input nodes ###
        node_states[:,self.nids_input_index] = X

        ### Forward propagation ###
        for nid_index, nid in enumerate(self.nids):

            snids,  weights = self.connections[nid_index]
            snids0, weights0 = self.recurrent_connections[nid_index]
            snids_index  = [self.nids.index(snid) for snid in snids]
            snids_index0 = [self.nids.index(snid) for snid in snids0]

            afct = ACTIVATIONS[self.nfcts[nid_index]]

            ### Calculate node state value ###
            if len(snids) > 0:
                assert nid_index>max(snids_index), "Network is not feedforward!"
                node_states[:,nid_index] += afct(  np.sum(np.asarray(weights )*      node_states[:, snids_index ], axis=1) + \
                                                 + np.sum(np.asarray(weights0)*self.node_states0[:, snids_index0], axis=1) + \
                                                 + self.bias[nid_index])
            else:
                if not nid_index in self.nids_input_index:
                    print("Warning: Node {} seems not to be used".format(nid))
                continue

        ### Store node state (for recurrent connections) ###
        self.node_states0 = node_states.copy()

        return node_states[:,self.nids_output_index].reshape(X.shape[0],-1)


    ### Crossover specie 1 is assumed to be the fitter one ###
    @classmethod
    def crossover(cls, specie1, specie2):

        complete = list(set(specie1.innovations+specie2.innovations))
        common = list(set(specie1.innovations) & set(specie2.innovations))
        uncom1 = list(set(specie1.innovations).difference(set(common)))
        uncom2 = list(set(specie2.innovations).difference(set(common)))

        print("Crossover")

        inos = len(complete)*[None]
        ### Common numbers ###
        common_index = [max([specie1.innovations.index(ino), specie2.innovations.index(ino)]) for ino in common]
        for index, ino in zip(common_index, common):
            inos[index] = ino
        
        disjoint, excess = [], []
        ### Uncommon numbers ###
        for n, ino in enumerate(complete):
            if ino in uncom1:
                s1_index = specie1.innovations.index(ino)
                inos[s1_index] = ino
                if s1_index > max(common_index):
                    excess.append(ino)
                else:
                    disjoint.append(ino)

            elif ino in uncom2:
                s2_index = specie2.innovations.index(ino)
                inos[s2_index] = ino
                if s2_index > max(common_index):
                    excess.append(ino)
                else:
                    disjoint.append(ino)


        print(inos)
        print(disjoint)
        print(excess)

#### TEST #######
if __name__ == "__main__":


    innomanager = InnovationManager()

    connections = [ [[],[]],
                    [[],[]],
                    [[1,2],[1.0, 1.0]],
                    [[1,2],[1.0, 1.0]],
                    [[3,5],[1.0, 1.0]],
                    ]

    specie1 = Specie(nids=[1,2,3,5,6], nfcts=[0,0,1,1,0], connections=connections, recurrent_connections=None,
                    nids_input=[1,2], nids_output=[6], innovationManager=innomanager)


    connections = [ [[],[]],
                    [[],[]],
                    [[1,2],[1.0, 1.0]],
                    [[3,2],[1.0, 1.0]],
                    [[1,2],[1.0, 1.0]],
                    [[3,5,7],[1.0, 1.0, 1.0]],
                    ]

    specie2 = Specie(nids=[1,2,3,7,5,6], nfcts=[0,0,1,1,1,0], connections=connections, recurrent_connections=None,
                    nids_input=[1,2], nids_output=[6], innovationManager=innomanager)


    specie1.checkInnovations()
    specie2.checkInnovations()

    print(specie1.innovations)
    print(specie2.innovations)
    Specie.crossover(specie1, specie2)

    #specie2.showGraph()

    X = np.random.rand(10,2)
    y = specie1.run(X)

