### This is the operation class object ###
class Operation():

    def __init__(self, nid, name="node", noutputs=1, ninputs=1, value=None):
        self.input_nodes, self.input_cons = [], []
        self.output_nodes, self.output_cons = [], []
        self.name = name
        self.nid = nid
        self.evaluated = False
        self.noutputs = noutputs
        self.ninputs = ninputs

    ### Sanity check ###
    def sanity_check(self):
        if len(self.input_nodes) == 0:
            print("Warning for {}(id={}): No input specified. Is this node needed? Add a connection or remove this node!".format(self.name, self.nid))
            return False

        if not len(list(set(self.input_cons))) == self.ninputs:
            print("Warning for {}(id={}): Not all or too many input connectors are used".format(self.name, self.nid))

        if not len(list(set(self.output_cons))) == self.noutputs:
            print("Warning for {}(id={}): Not all or too many output connectors are used".format(self.name, self.nid))

        return True  

    ### nodeobjct ==> this op ###
    def addFromNode(self, nodeobj, connector=0):
        self.input_nodes.append(nodeobj)
        self.input_cons.append(connector)

    ### this op ==> nodeobjct ###
    def addToNode(self, nodeobj, connector=0):
        self.output_nodes.append(nodeobj)
        self.output_cons.append(connector)

    def __repr__(self):
        return "<{}>".format(self.name)
    def __str__(self):
        return "<{}>".format(self.name)
    def compute(self):
        pass



class FunctionOperation(Operation):
    def __init__(self, nid, name="node", noutputs=1, ninputs=1, value=None):
        super().__init__(nid, name, noutputs, ninputs, value=None)
        self.function = None

    def compute(self, X):
        return [self.function(X)]