import numpy as np
import json
import copy
import matplotlib.pyplot as plt

from activations import ACTIVATIONS

INNOVATIONSCTR = 0
INNOVATIONS = {}
GENOMCTR = 0

### Genom class for neat ###
class Genom:

    ### Constructor ###
    def __init__(self, nids, nids_input, nids_output, structure, maxtimelevel=2,
                 _id=None, generation=0, parents=[None,None], crossover=True, iter_survived=0):

        ### Structural parameters ###
        self.nids = nids
        self.structure = structure
        self.maxtimelevel = maxtimelevel

        assert len(self.structure) == len(self.nids), "Length of structure must match length of Nids!"
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
        return "<Genom ID {}, parents {} ,gen. {}, survived {}>".format(self._id, self.parents, self.generation, self.iter_survived)

    ###
    @property
    def innovationNumbers(self):
        return sorted([ino for nid in self.nids for ino in self.structure[nid]["connections"]["innovations"]])

    @property
    def numberOfConnections(self):
        return sum([len(self.structure[nid]["connections"]["snids"]) for nid in self.nids])

    @property
    def sumOfWeightsAndBiases(self):
        return sum([sum(self.structure[nid]["connections"]["weights"]) + self.structure[nid]["bias"] for nid in self.nids])

    ### ======================================
    # Innovation manager stuff
    ### ======================================
    ### Add innovation ###
    @staticmethod
    def _addInnovation(n1, n2, level):
        global INNOVATIONSCTR
        ### Name of the invention ###
        name = "{}-{}-{}".format(n1, n2, level)

        if name in list(INNOVATIONS.keys()):
            return INNOVATIONS[name]["id"]
        else:
            INNOVATIONSCTR += 1
            INNOVATIONS[name] = {"id":INNOVATIONSCTR, "feats": [n1, n2, level]}
            print("New innovationID {} ({}) added!".format(INNOVATIONSCTR,name))
            return INNOVATIONSCTR

    ### Return innovation features ###
    @staticmethod
    def _getFeatures(ino):
        return [vals for key, vals in INNOVATIONS.items() if vals["id"] == ino][0]["feats"]

    ### ======================================
    # Feedforward run
    ### ======================================
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
        for nid_index in range(len(self.nids_input), len(self.nids)):
            nid = self.nids[nid_index]
            snids  = self.structure[nid]["connections"]["snids"]
            weights  = self.structure[nid]["connections"]["weights"]
            level = self.structure[nid]["connections"]["level"] # The time level of the connection
            bias = self.structure[nid]["bias"]
            fct = ACTIVATIONS[self.structure[nid]["activation"]]

            assert all(snid in self.nids for snid in snids), "Snid not in nid: {}, {}, {}".format(self.nids, self.structure, self.parents)
            snids_index = [self.nids.index(snid) for snid in snids]

            ### Calculate node state value ###
            if len(snids) > 0:
                assert nid_index>max([snid for l, snid in zip(level, snids_index) if l == 0], default=0), "Network is not feedforward! nids: {}, structure:{}, parents {}".format(self.nids, self.structure, self.parents)
                node_states[:,nid_index, 0] += fct(np.sum(np.asarray(weights)* node_states[:, snids_index, level], axis=1) + bias)

            ### The node seems not to have any input ###
            else:
                continue

        #print(node_states[:,:,0])
        ### Store node state (for recurrent connections) ###
        self.last_states.insert(0, node_states[:,:,0].copy())
        self.last_states = self.last_states[:self.maxtimelevel]

        ### Return output nodes values ###
        return node_states[:,self.nids_output_index, 0].reshape(X.shape[0],len(self.nids_output_index))


    ### ======================================
    # Initialization
    ### ======================================
    ### Create random structure ###
    @classmethod
    def initializeRandomly(cls, ninputs, noutputs, maxtimelevel=2, paddcon=0.8, paddnode=0.0, paddbias=0.5, pmutact=0.0, nrerun=None, output_activation=None):
        global GENOMCTR
        nids_input, nids_output = [x for x in range(0,ninputs)], [x for x in range(ninputs ,ninputs+noutputs)]
        nids = nids_input+nids_output
        nrerun = max([ninputs, noutputs]) if nrerun is None else nrerun

        output_activation = noutputs*[0] if output_activation is None else output_activation
        input_activation = ninputs*[0]

        assert len(output_activation) == noutputs, "Output activation must match number of outputs"

        structure= {}
        for nid,act in zip(nids, input_activation+output_activation):
            structure[nid] = {'connections':{"snids":[], "weights":[], "level":[], "innovations": []},'activation': act, 'bias': 0.0}

        genom = cls(nids, nids_input, nids_output, structure, maxtimelevel=maxtimelevel,
                     _id=GENOMCTR, generation=0, parents=['init'])

        for _ in range(nrerun):
            if np.random.rand() < paddcon:
                genom = Genom.mutate_add_connection(genom)
            if np.random.rand() < paddbias:
                genom = Genom.mutate_bias(genom)
            if np.random.rand() < paddnode:
                genom = Genom.mutate_add_node(genom)
            if np.random.rand() < pmutact:
                pass #genom = Genom.mutate_activation(genom)

        GENOMCTR +=1
        return genom

    ### Mutate bias ###
    @classmethod
    def mutate_activation(cls, genom1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)
        nid_mut = nids[np.random.randint(len(genom1.nids_input), len(nids))]
        structure[nid_mut]["activation"] = np.random.randint(0,len(ACTIVATIONS))

        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_activation'], _id=genom1._id,
                   iter_survived=genom1.iter_survived, crossover=genom1.crossover)


    ### Mutate bias ###
    @classmethod
    def mutate_bias(cls, genom1, valueabs=1.0, pbigChange=0.3, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)
        nid_mut = nids[np.random.randint(len(genom1.nids_input), len(nids))]
        if np.random.rand() < pbigChange:
            structure[nid_mut]["bias"] = -valueabs+2*valueabs*np.random.rand()
        else:
            structure[nid_mut]["bias"] += 0.1*valueabs*np.random.normal()

        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_bias'], _id=genom1._id,
                   iter_survived=genom1.iter_survived, crossover=genom1.crossover)

    ### Mutate weight ###
    @classmethod
    def mutate_weight(cls, genom1, valueabs=1.0, pbigChange=0.3, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)

        nid_mut = nids[np.random.randint(len(genom1.nids_input), len(nids))]

        if len(structure[nid_mut]["connections"]["snids"]) > 0:
            index = np.random.randint(0, len(structure[nid_mut]["connections"]["snids"]))

            if np.random.rand() < pbigChange:
                structure[nid_mut]["connections"]["weights"][index] = -valueabs+2*valueabs*np.random.rand()
            else:
                structure[nid_mut]["connections"]["weights"][index] += 0.1*valueabs*np.random.normal()

            return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                       maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_weight'], _id=genom1._id,
                       iter_survived=genom1.iter_survived, crossover=genom1.crossover)
        else:
            return genom1

    ### Add connection ###
    @classmethod
    def mutate_add_connection(cls, genom1, valueabs=1, maxretries=1, generation=None, timelevel=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)
        timelevel = genom1.maxtimelevel if timelevel is None else timelevel
        level = np.random.randint(0, timelevel)
        weight = valueabs*np.random.normal()

        #nid_add = nids[np.random.randint(len(genom1.nids_input), len(nids))]
        nid_add = nids[np.random.randint(len(genom1.nids_input), len(nids))]

        ### Find a valid SNID ###
        for _ in range(maxretries):
            snid = nids[np.random.randint(0, nids.index(nid_add))] if level == 0 else nids[np.random.randint(0, nids.index(nid_add))]
            if not snid in structure[nid_add]["connections"]["snids"] and not snid in genom1.nids_output:
                break
            elif snid == nid_add and level >0:
                break
            if _ == maxretries -1:
                return genom1

        ### Add everything ###
        structure[nid_add]["connections"]["snids"].append(snid)
        structure[nid_add]["connections"]["weights"].append(weight)
        structure[nid_add]["connections"]["level"].append(level)
        structure[nid_add]["connections"]["innovations"].append(Genom._addInnovation(snid, nid_add, level))

        print("Connection added")
        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_add_connection'], _id=genom1._id,
                   iter_survived=genom1.iter_survived, crossover=genom1.crossover)

    ### Remove node ###
    @classmethod
    def mutate_remove_connection(cls, genom1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)

        if len(genom1.nids_input) == len(nids)-len(genom1.nids_output):
            nid_remove = nids[len(genom1.nids_input)]
        else:
            nid_remove = nids[np.random.randint(len(genom1.nids_input), len(nids)-len(genom1.nids_output))]

        if len(structure[nid_remove]["connections"]["snids"]) > 0:
            index = np.random.randint(0, len(structure[nid_remove]["connections"]["snids"]))
            structure[nid_remove]["connections"]["innovations"].pop(index)
            structure[nid_remove]["connections"]["weights"].pop(index)
            structure[nid_remove]["connections"]["snids"].pop(index)
            structure[nid_remove]["connections"]["level"].pop(index)

            print("Connection removed ")
            return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                       maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_remove_connection'], _id=genom1._id,
                       iter_survived=genom1.iter_survived, crossover=genom1.crossover)
        else:
            return genom1

    ### Add node ###
    @classmethod
    def mutate_add_node(cls, genom1, activation=1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)

        ### get rnid index ### 
        while True:
            if len(genom1.nids_input+genom1.nids_output) == len(genom1.nids):
                idx = len(genom1.nids_input)
                rnid = genom1.nids[idx]
                nid_new = len(genom1.nids)
            else:
                idx = np.random.randint(len(genom1.nids_input),len(genom1.nids)-len(genom1.nids_output))
                rnid = genom1.nids[idx]
                nid_new = rnid+1
            
            if not nid_new in genom1.nids:
                break

        ### Inserting new nid ###
        nids.insert(idx, nid_new)
        structure[nid_new] = {"connections": {"snids":[], "innovations":[], "level":[], "weights":[]}, "bias":0.0, "activation": 1}

        ### Link it ###
        if len(genom1.structure[rnid]["connections"]["snids"])>0:
            snid_idx = np.random.randint(0, len(genom1.structure[rnid]["connections"]["snids"]))
            snid = genom1.structure[rnid]["connections"]["snids"][snid_idx]
            weight = genom1.structure[rnid]["connections"]["weights"][snid_idx]
            level = genom1.structure[rnid]["connections"]["level"][snid_idx]
            ino  = genom1.structure[rnid]["connections"]["innovations"][snid_idx]

            ### Remove original connection ###
            structure[rnid]["connections"]["snids"].pop(snid_idx)
            structure[rnid]["connections"]["level"].pop(snid_idx)
            structure[rnid]["connections"]["weights"].pop(snid_idx)
            structure[rnid]["connections"]["innovations"].pop(snid_idx)

            ### Add link to new node ###
            structure[nid_new]["connections"]["snids"].append(snid)
            structure[nid_new]["connections"]["level"].append(level)
            structure[nid_new]["connections"]["weights"].append(1.0)
            structure[nid_new]["connections"]["innovations"].append(Genom._addInnovation(snid, nid_new, level))

            ### Add link to rnid ###
            structure[rnid]["connections"]["snids"].append(nid_new)
            structure[rnid]["connections"]["level"].append(level)
            structure[rnid]["connections"]["weights"].append(weight)
            structure[rnid]["connections"]["innovations"].append(Genom._addInnovation(nid_new, rnid, level))

            print("New node added")
            return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                       maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_add_node'], _id=genom1._id,
                       iter_survived=genom1.iter_survived, crossover=genom1.crossover)
        else:
            return genom1

    ### Remove node ###
    @classmethod
    def mutate_remove_node(cls, genom1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)

        ### Check if additional nodes exists ###
        if not len(nids) == len(genom1.nids_input) + len(genom1.nids_output):
            nid_remove = nids[np.random.randint(len(genom1.nids_input), len(nids)-len(genom1.nids_output))]
            snids = structure[nid_remove]["connections"]["snids"]
            level = structure[nid_remove]["connections"]["level"]
            weights = structure[nid_remove]["connections"]["weights"]
            ### Remove node ###
            nids.remove(nid_remove)
            del structure[nid_remove]

            ### Remove connections ###
            for nid in nids:
                if nid_remove in structure[nid]["connections"]["snids"]:
                    ### Removing connection to deleted node ###
                    index = structure[nid]["connections"]["snids"].index(nid_remove)
                    structure[nid]["connections"]["innovations"].pop(index)
                    structure[nid]["connections"]["weights"].pop(index)
                    structure[nid]["connections"]["snids"].pop(index)
                    structure[nid]["connections"]["level"].pop(index)

                    ### Adding connections ###
                    for l,snid,weight in zip(level,snids,weights):
                        if not snid in structure[nid]["connections"]["snids"]:
                            structure[nid]["connections"]["innovations"].append(Genom._addInnovation(snid, nid, l))
                            structure[nid]["connections"]["weights"].append(weight)
                            structure[nid]["connections"]["snids"].append(snid)
                            structure[nid]["connections"]["level"].append(l)

            print("Node {} removed".format(nid_remove))
            return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                       maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_remove_node'], _id=genom1._id,
                       iter_survived=genom1.iter_survived, crossover=genom1.crossover)

        else:
            return genom1

    ### Crossover genom 1 is assumed to be the fitter one ###
    @classmethod
    def crossover(cls, genom1, genom2, generation=None):
        global GENOMCTR

        n1, n2 = genom1.innovationNumbers, genom2.innovationNumbers
        cn = list(set(n1) & set(n2))
        n3 = list(set(n1 + [ino for ino in n2 if ino<max(n1)]))

        ### Get NIDS ###
        nids_remaining = []
        for ino in n3:
            snid, rnid, level = Genom._getFeatures(ino)
            if not snid in genom1.nids_input+genom1.nids_output and not snid in nids_remaining:
                nids_remaining.append(snid)
            if not rnid in genom1.nids_input+genom1.nids_output and not rnid in nids_remaining:
                nids_remaining.append(rnid)

        nids = genom1.nids_input.copy() + sorted(nids_remaining)[::-1] + genom1.nids_output.copy()

        ### add nids from innovation number
        structure = {nid: genom1.structure[nid] for nid in genom1.nids_input}
        for nid in nids_remaining+genom1.nids_output:
            structure[nid] = {'connections':{"snids":[], "weights":[], "level":[], "innovations": []},'activation': 0, 'bias': 0.0}

        ### Add connections ###
        for ino in n3:
            snid, rnid, level = Genom._getFeatures(ino)

            ### Get weight, bias and activation ###
            if ino in cn:
                idx1 = genom1.structure[rnid]['connections']['snids'].index(snid)
                idx2 = genom2.structure[rnid]['connections']['snids'].index(snid)
                weight = [genom1.structure[rnid]['connections']['weights'][idx1], genom2.structure[rnid]['connections']['weights'][idx2]][np.random.randint(0,2)]
                bias = [genom1.structure[rnid]['bias'], genom2.structure[rnid]['bias']][np.random.randint(0,2)]
                activation = [genom1.structure[rnid]['activation'], genom2.structure[rnid]['activation']][np.random.randint(0,2)]

            elif ino in n1:
                idx1 = genom1.structure[rnid]['connections']['snids'].index(snid)
                weight = genom1.structure[rnid]['connections']['weights'][idx1]
                bias = genom1.structure[rnid]['bias']
                activation = genom1.structure[rnid]['activation']

            elif ino in n2:
                idx2 = genom2.structure[rnid]['connections']['snids'].index(snid)
                weight = genom2.structure[rnid]['connections']['weights'][idx2]
                bias = genom2.structure[rnid]['bias']
                activation = genom2.structure[rnid]['activation']

            structure[rnid]['connections']['snids'].append(snid)
            structure[rnid]['connections']['weights'].append(weight)
            structure[rnid]['connections']['level'].append(level)
            structure[rnid]['connections']['innovations'].append(ino)
            structure[rnid]['bias'] = bias
            structure[rnid]['activation'] = activation

        ### Add output nids (in case they are not added yet) ###
        for nid in genom1.nids_output:
            if not nid in list(structure.keys()):
                bias, activation = genom1.structure[nid]["bias"], genom1.structure[nid]["activation"]
                structure[nid] = {'connections':{"snids":[], "weights":[], "level":[], "innovations": []},
                                  'activation': activation, 'bias': bias}

        if not len(nids) == len(structure):
            print(list(structure.keys()))
            print(nids)

        ### NIDS ###
        GENOMCTR+=1
        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, genom2._id], _id=GENOMCTR,
                   crossover=True)

    ### =====================================
    # Calculate compability
    ### =====================================
    @staticmethod
    def compabilityMeasure(genom1, genom2, c1=1.0, c2=1.0, c3=0.3):
        inos1 = genom1.innovationNumbers
        inos2 = genom2.innovationNumbers
        ncons1 = genom1.numberOfConnections
        ncons2 = genom2.numberOfConnections
        nsumw1 = genom1.sumOfWeightsAndBiases
        nsumw2 = genom2.sumOfWeightsAndBiases

        if ncons1 < 1 or ncons1 < 1:
            return 1

        cinos  = list(set(inos1) & set(inos2))
        uinos1 = list(set(inos1).difference(set(cinos)))
        uinos2 = list(set(inos2).difference(set(cinos)))
        ainos =  list(set(inos1+inos2))

        cinos_index=[max([inos1.index(ino), inos2.index(ino)]) for ino in cinos]

        excess, disjoint = [], []
        for ino in uinos1:
            if inos1.index(ino) > max(cinos_index, default=0):
                excess.append(ino)
            else:
                disjoint.append(ino)

        for ino in uinos2:
            if inos2.index(ino) > max(cinos_index, default=0):
                excess.append(ino)
            else:
                disjoint.append(ino)


        N = len(inos1) if len(inos1) > len(inos2) else len(inos2)

        return c1*len(excess)/(N+1e-5) + c2*len(disjoint)/(N+1e-5) + c3*abs(nsumw1/(ncons1+1e-5)-nsumw2/(ncons2+1e-5))



    ### Use to create a graph ###
    def showGraph(self, showLabels=True, store=False, picname="network.png"):
        executionLevel = []
        for nid in self.nids:
            if nid in self.nids_input:
                executionLevel.append(0)
            else:
                snids = self.structure[nid]["connections"]["snids"]
                executionLevel.append(max([executionLevel[self.nids.index(snid)] for snid in snids], default=0)+1)
        for nid in self.nids_output:
            executionLevel[self.nids.index(nid)] = max([executionLevel[self.nids.index(nido)] for nido in self.nids_output], default=0)

        y = []
        for i, (l, nid) in enumerate(zip(executionLevel, self.nids)):
            y.append(-0.5*executionLevel.count(l) + executionLevel[i:].count(l)+0.1-0.2*np.random.rand())

        for nid_index, nid in enumerate(self.nids):
            for snid, weight, level in zip(self.structure[nid]["connections"]["snids"],
                self.structure[nid]["connections"]["weights"], self.structure[nid]["connections"]["level"]):
                snid_index = self.nids.index(snid)
                color = "r" if weight>0 else "b"
                style = "-" if level == 0 else "--"

                plt.arrow(executionLevel[snid_index], y[snid_index],
                          (executionLevel[nid_index]-executionLevel[snid_index]),(y[nid_index]-y[snid_index]), ls=style,
                          color=color, head_width=0.05, head_length=0.1, fc=color, ec=color, length_includes_head=True)

        for nid_index, nid in enumerate(self.nids):
            plt.plot(executionLevel[nid_index], y[nid_index], 'o', color="k", markersize=12)
            if showLabels:
                plt.text(executionLevel[nid_index], y[nid_index], nid, ha='center', va='center', color="w")
        plt.title("{}".format(self))
        plt.axis('off')
        if store:
            plt.savefig(picname)
            plt.close()
        else:
            plt.show()


#### TEST #######
if __name__ == "__main__":

    npop =10
    genoms = [Genom.initializeRandomly(ninputs=2, noutputs=2, maxtimelevel=1) for pop in range(npop)]

    for _ in range(100):
        X = np.random.rand(10,2)

        genoms_run = (genom.run(X) for genom in genoms)

        for run in genoms_run:
            run

        for genom1 in genoms:
            for genom2 in genoms:
                sigma = Genom.compabilityMeasure(genom1, genom2)
                print(sigma)
        sys.exit(9)


        for n in range(npop):

            genoms[n] = Genom.crossover(genoms[np.random.randint(0,npop)], genoms[np.random.randint(0,npop)])

            if np.random.rand() < 0.02:
                genoms[n] = Genom.mutate_add_node(genoms[n])
            if np.random.rand() < 0.05:
                genoms[n] = Genom.mutate_remove_node(genoms[n])
            if np.random.rand() < 0.5:
                genoms[n] = Genom.mutate_add_connection(genoms[n])
            if np.random.rand() < 0.1:
                genoms[n] = Genom.mutate_remove_connection(genoms[n])
            if np.random.rand() < 0.5:
                genoms[n] = Genom.mutate_weight(genoms[n])
            if np.random.rand() < 0.5:
                genoms[n] = Genom.mutate_bias(genoms[n])
            if np.random.rand() < 0.5:
                genoms[n] = Genom.mutate_activation(genoms[n])

    for genom in genoms:
        genom.showGraph()
        print(genom.cost())
    # genom6 = initializeRandomly()

    # genom3 = Genom.crossover(genom1, genom2)
    # genom4 = Genom.mutate_add_node(genom1)
    # genom5 = Genom.mutate_remove_node(genom4)
    # genom6 = Genom.crossover(genom4, genom5)
    # genom5 = Genom.mutate_add_connection(genom5)
    # genom3 = Genom.mutate_remove_connection(genom3)

    # genom4.showGraph()
    # genom5.showGraph()
    # #

    # for _ in range(1):
    #     print("####", _ ,"####")
    #     X = np.random.rand(10,2)
    #     y = genom5.run(X)
    #     print(y)