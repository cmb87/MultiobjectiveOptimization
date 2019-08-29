import numpy as np
import json
import copy
import matplotlib.pyplot as plt

from activations import ACTIVATIONS

INNOVATIONSCTR = 0
INNOVATIONS = {}
NIDDICT = {}
NIDMAX = 0
SPECIECTR = 0

### Specie class for neat ###
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
        return "<Specie ID {}, parents {} ,gen. {}, survived {}>".format(self._id, self.parents, self.generation, self.iter_survived)

    ####
    @property
    def innovationNumbers(self):
        return {ino:nid for nid in self.nids for ino in self.structure[nid]["connections"]["innovations"]}
    
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
        global INNOVATIONSCTR, NIDMAX
        ### Update NIDMAX ###
        NIDMAX = max([n1, n2, NIDMAX])
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

    ### get new node id ###
    @staticmethod
    def _getNewNID(nid_previous=None, nid_following=None):
        global NIDMAX, NIDDICT
        name = "{}-{}".format(nid_previous, nid_following)

        if not nid_previous is None and not nid_following is None:
            if not name in list(NIDDICT.keys()):
                NIDMAX += 1
                NIDDICT[name] = NIDMAX
                return NIDMAX
            else:
                return NIDDICT[name]
        else:
            NIDMAX += 1
            return NIDMAX

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
        for nid_index, nid in enumerate(self.nids):
            snids  = self.structure[nid]["connections"]["snids"]
            weights  = self.structure[nid]["connections"]["weights"]
            level = self.structure[nid]["connections"]["level"] # The time level of the connection
            bias = self.structure[nid]["bias"]
            fct = ACTIVATIONS[self.structure[nid]["activation"]]

            snids_index = [self.nids.index(snid) for snid in snids]

            ### Calculate node state value ###
            if len(snids) > 0:
                assert nid_index>max([snid for l, snid in zip(level, snids_index) if l == 0], default=0), "Network is not feedforward!"
                node_states[:,nid_index, 0] += fct(np.sum(np.asarray(weights)* node_states[:, snids_index, level], axis=1) + bias)

            ### The node seems not to have any input ###
            else:
                continue

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
    def initializeRandomly(cls, ninputs, noutputs, maxtimelevel=2, paddcon=0.8, paddnode=0.0, paddbias=0.5, pmutact=0.5, nrerun=None):

        global NIDMAX, SPECIECTR
        nids_input, nids_output = [x for x in range(0,ninputs)], [x for x in range(ninputs ,ninputs+noutputs)]
        nids = nids_input+nids_output
        nrerun = max([ninputs, noutputs]) if nrerun is None else nrerun
        NIDMAX = max(nids+[NIDMAX])

        structure= {}
        for nid in nids:
            structure[nid] = {'connections':{"snids":[], "weights":[], "level":[], "innovations": []},'activation': 0, 'bias': 0.0} 

        specie = cls(nids, nids_input, nids_output, structure, maxtimelevel=maxtimelevel, 
                     _id=SPECIECTR, generation=0, parents=['init'])

        for _ in range(nrerun):
            if np.random.rand() < paddcon:
                specie = Specie.mutate_add_connection(specie)
            if np.random.rand() < paddbias:
                specie = Specie.mutate_bias(specie)
            if np.random.rand() < paddnode:
                specie = Specie.mutate_add_node(specie)
            if np.random.rand() < pmutact:
                specie = Specie.mutate_activation(specie)

        SPECIECTR +=1
        return specie

    ### get structure cost ###
    def cost(self, c1=1.0, c2=1.0, c3=1.0):
        nnids = len(self.nids)
        ncons = sum([len(self.structure[nid]["connections"]["snids"]) for i, nid in enumerate(self.nids)])
        if ncons == 0:
            awgts = 0
        else:
            awgts = sum([sum(self.structure[nid]["connections"]["weights"])/ncons for i, nid in enumerate(self.nids)])/nnids
        return nnids*c1 + ncons*c2 + awgts*c3

    ### Mutate bias ###
    @classmethod
    def mutate_activation(cls, specie1, generation=None):
        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)
        nid_mut = nids[np.random.randint(len(specie1.nids_input), len(nids))]
        structure[nid_mut]["activation"] = np.random.randint(0,len(ACTIVATIONS))

        return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                   maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id], _id=specie1._id, 
                   iter_survived=specie1.iter_survived)


    ### Mutate bias ###
    @classmethod
    def mutate_bias(cls, specie1, valueabs=1.0, pbigChange=0.5, generation=None):
        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)
        nid_mut = nids[np.random.randint(len(specie1.nids_input), len(nids))]
        if np.random.rand() > pbigChange:
            structure[nid_mut]["bias"] = valueabs*np.random.normal()
        else:
            structure[nid_mut]["bias"] += valueabs*np.random.normal()

        return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                   maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id], _id=specie1._id, 
                   iter_survived=specie1.iter_survived)

    ### Mutate weight ###
    @classmethod
    def mutate_weight(cls, specie1, valueabs=1.0, pbigChange=0.5, generation=None):
        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)
        nid_mut = nids[np.random.randint(len(specie1.nids_input), len(nids))]

        if len(structure[nid_mut]["connections"]["snids"]) > 0:
            index = np.random.randint(0, len(structure[nid_mut]["connections"]["snids"]))

            if np.random.rand() > pbigChange:
                structure[nid_mut]["connections"]["weights"][index] = valueabs*np.random.normal()
            else:
                structure[nid_mut]["connections"]["weights"][index] += valueabs*np.random.normal()

            return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                       maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id], _id=specie1._id, 
                       iter_survived=specie1.iter_survived)
        else:
            return specie1

    ### Add connection ###
    @classmethod
    def mutate_add_connection(cls, specie1, valueabs=0.5, maxretries=1, generation=None, timelevel=None):
        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)
        timelevel = specie1.maxtimelevel if timelevel is None else timelevel
        level = np.random.randint(0, timelevel)
        weight = valueabs*np.random.normal()

        #nid_add = nids[np.random.randint(len(specie1.nids_input), len(nids))]
        nid_add = nids[np.random.randint(len(specie1.nids_input), len(nids))]

        ### Find a valid SNID ###
        for _ in range(maxretries):
            snid = nids[np.random.randint(0, nids.index(nid_add))] if level == 0 else nids[np.random.randint(0, nids.index(nid_add))]
            if not snid in structure[nid_add]["connections"]["snids"] and not snid in specie1.nids_output:
                break
            elif snid == nid_add and level >0:
                break
            if _ == maxretries -1:
                return specie1

        ### Add everything ###
        structure[nid_add]["connections"]["snids"].append(snid)
        structure[nid_add]["connections"]["weights"].append(weight)
        structure[nid_add]["connections"]["level"].append(level)
        structure[nid_add]["connections"]["innovations"].append(Specie._addInnovation(snid, nid_add, level))

        print("Connection added")
        return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                   maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id], _id=specie1._id, 
                   iter_survived=specie1.iter_survived)

    ### Remove node ###
    @classmethod
    def mutate_remove_connection(cls, specie1, generation=None):
        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)

        if len(specie1.nids_input) == len(nids)-len(specie1.nids_output):
            nid_remove = nids[len(specie1.nids_input)]
        else:
            nid_remove = nids[np.random.randint(len(specie1.nids_input), len(nids)-len(specie1.nids_output))]

        if len(structure[nid_remove]["connections"]["snids"]) > 0:
            index = np.random.randint(0, len(structure[nid_remove]["connections"]["snids"]))
            structure[nid_remove]["connections"]["innovations"].pop(index)
            structure[nid_remove]["connections"]["weights"].pop(index)
            structure[nid_remove]["connections"]["snids"].pop(index)
            structure[nid_remove]["connections"]["level"].pop(index)

            print("Connection removed ")
            return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                       maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id], _id=specie1._id, 
                       iter_survived=specie1.iter_survived)
        else:
            return specie1

    ### Add node ###
    @classmethod
    def mutate_add_node(cls, specie1, activation=1, generation=None):
        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)
        
        if len(specie1.nids_input) == len(nids)-len(specie1.nids_output):
            rnid_index = len(specie1.nids_input)
            rnid = nids[rnid_index]
        else:
            rnid_index = np.random.randint(len(specie1.nids_input), len(nids)-len(specie1.nids_output))
            rnid = nids[rnid_index]

        if len(structure[rnid]["connections"]["innovations"]) > 0:
            index = np.random.randint(0, len(structure[rnid]["connections"]["snids"]))
            snid, _, level = Specie._getFeatures(structure[rnid]["connections"]["innovations"][index]) 
            weight = structure[rnid]["connections"]["weights"][index]

            ### Insert new node ###
            if max([len(specie1.nids_input), nids.index(snid)+1]) == nids.index(rnid):
                index_insert = max([len(specie1.nids_input), nids.index(snid)+1])
            else:
                index_insert = np.random.randint(max([len(specie1.nids_input), nids.index(snid)+1]),nids.index(rnid))

            nid_new = Specie._getNewNID(nid_previous=nids[index_insert], nid_following=nids[index_insert-1])
            nids.insert(index_insert, nid_new)

            ### Remove existing connection ###
            structure[rnid]["connections"]["innovations"].pop(index)
            structure[rnid]["connections"]["weights"].pop(index)
            structure[rnid]["connections"]["snids"].pop(index)
            structure[rnid]["connections"]["level"].pop(index)

            ### Add connection from new node ###
            structure[rnid]["connections"]["innovations"].append(Specie._addInnovation(nid_new, rnid, level))
            structure[rnid]["connections"]["weights"].append(weight)
            structure[rnid]["connections"]["snids"].append(nid_new)
            structure[rnid]["connections"]["level"].append(level)

            ### Add connection to new node ###
            structure[nid_new] = {"connections": {}}
            structure[nid_new]["connections"]["innovations"] = [Specie._addInnovation(snid, nid_new, level)]
            structure[nid_new]["connections"]["weights"] = [np.random.normal()]
            structure[nid_new]["connections"]["snids"] = [snid]
            structure[nid_new]["connections"]["level"] = [level]
            structure[nid_new]["bias"] = 0.0
            structure[nid_new]["activation"] = np.random.randint(0,len(ACTIVATIONS)) if activation is None else activation

            print("Node added")
            return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                       maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id], _id=specie1._id, 
                       iter_survived=specie1.iter_survived)
        else:
            return specie1

    ### Remove node ###
    @classmethod
    def mutate_remove_node(cls, specie1, generation=None):
        nids = specie1.nids.copy()
        structure = copy.deepcopy(specie1.structure)

        ### Check if additional nodes exists ###
        if not len(nids) == len(specie1.nids_input) + len(specie1.nids_output):
            nid_remove = nids[np.random.randint(len(specie1.nids_input), len(nids)-len(specie1.nids_output))]
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
                            structure[nid]["connections"]["innovations"].append(Specie._addInnovation(snid, nid, l))
                            structure[nid]["connections"]["weights"].append(weight)
                            structure[nid]["connections"]["snids"].append(snid)
                            structure[nid]["connections"]["level"].append(l)

            print("Node {} removed".format(nid_remove))

            return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                       maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id], _id=specie1._id, 
                       iter_survived=specie1.iter_survived)

        else:
            return specie1

    ### Crossover specie 1 is assumed to be the fitter one ###
    @classmethod
    def crossover(cls, specie1, specie2, generation=None):
        global SPECIECTR
        inos1 = specie1.innovationNumbers
        inos2 = specie2.innovationNumbers

        cinos  = list(set(inos1.keys()) & set(inos2.keys()))
        uinos1 = list(set(inos1.keys()).difference(set(cinos)))

        nids = specie1.nids.copy()
        structure = {nid: {'connections':{"snids":[], "weights":[], "level":[], "innovations": []}, 
                           'activation': 0, 'bias': 0.0} for nid in nids}

        ### inputs ###
        for nid in specie1.nids_input:
            structure[nid] = specie1.structure[nid]

        ### Common innovations ###
        for cino in cinos:
            parent = [specie1.structure[inos1[cino]], specie2.structure[inos2[cino]]][np.random.randint(0,2)]
            
            snid, rnid, level = Specie._getFeatures(cino)

            weight = parent["connections"]['weights'][parent["connections"]['snids'].index(snid)]

            structure[inos1[cino]]['connections']['snids'].append(snid)
            structure[inos1[cino]]['connections']['level'].append(level)
            structure[inos1[cino]]['connections']['weights'].append(weight)
            structure[inos1[cino]]['connections']['innovations'].append(cino)
            structure[inos1[cino]]['bias'] = parent["bias"]
            structure[inos1[cino]]['activation'] = parent["activation"]

        ### Uncommon (only taken from fitter species 1) ###
        for uino in uinos1:
            structure[inos1[uino]] = specie1.structure[inos1[uino]]

        SPECIECTR+=1
        return cls(nids=nids, structure=structure, nids_output=specie1.nids_output, nids_input=specie1.nids_input,
                   maxtimelevel=specie1.maxtimelevel, generation=generation, parents=[specie1._id, specie2._id], _id=SPECIECTR)

    ### =====================================
    # Calculate compability
    ### =====================================
    @staticmethod
    def compabilityMeasure(specie1, specie2, c1=0.5, c2=0.5, c3=0.5):
        inos1 = [val for key, val in specie1.innovationNumbers.items()]
        inos2 = [val for key, val in specie1.innovationNumbers.items()]
        ncons1 = specie1.numberOfConnections
        ncons2 = specie2.numberOfConnections
        nsumw1 = specie1.sumOfWeightsAndBiases
        nsumw2 = specie2.sumOfWeightsAndBiases

        cinos  = list(set(inos1) & set(inos2))
        uinos1 = list(set(inos1).difference(set(cinos)))
        uinos2 = list(set(inos2).difference(set(cinos)))
        ainos =  list(set(inos1+inos2))

        cinos_index=[max([inos1.index(ino), inos2.index(ino)]) for ino in cinos] 

        excess, disjoint = [], []
        for ino in uinos1:
            if inos1.index(ino) > max(cinos_index):
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
    species = [Specie.initializeRandomly(ninputs=2, noutputs=2, maxtimelevel=1) for pop in range(npop)]

    for _ in range(100):
        X = np.random.rand(10,2)
        
        species_run = (specie.run(X) for specie in species)
        
        for run in species_run:
            run

        for specie1 in species:
            for specie2 in species:
                sigma = Specie.compabilityMeasure(specie1, specie2)
                print(sigma)
        sys.exit(9)


        for n in range(npop):

            species[n] = Specie.crossover(species[np.random.randint(0,npop)], species[np.random.randint(0,npop)])

            if np.random.rand() < 0.02:
                species[n] = Specie.mutate_add_node(species[n])
            if np.random.rand() < 0.05:
                species[n] = Specie.mutate_remove_node(species[n])
            if np.random.rand() < 0.5:
                species[n] = Specie.mutate_add_connection(species[n])
            if np.random.rand() < 0.1:
                species[n] = Specie.mutate_remove_connection(species[n])
            if np.random.rand() < 0.5:
                species[n] = Specie.mutate_weight(species[n])
            if np.random.rand() < 0.5:
                species[n] = Specie.mutate_bias(species[n])
            if np.random.rand() < 0.5:
                species[n] = Specie.mutate_activation(species[n])

    for specie in species:
        specie.showGraph()
        print(specie.cost())
    # specie6 = initializeRandomly()

    # specie3 = Specie.crossover(specie1, specie2)
    # specie4 = Specie.mutate_add_node(specie1)
    # specie5 = Specie.mutate_remove_node(specie4)
    # specie6 = Specie.crossover(specie4, specie5)
    # specie5 = Specie.mutate_add_connection(specie5)
    # specie3 = Specie.mutate_remove_connection(specie3)

    # specie4.showGraph()
    # specie5.showGraph()
    # # 

    # for _ in range(1):
    #     print("####", _ ,"####")
    #     X = np.random.rand(10,2)
    #     y = specie5.run(X)
    #     print(y)