### This is the placeholder object ###
class Variable():

    def __init__(self, nid, name="variable", value=[0], ninputs=0, noutputs=0):
        self.input_nodes, self.input_cons = [], []
        self.output_nodes, self.output_cons = [], []
        self.name = name
        self.evaluated = False
        self.nid = nid
        self.noutputs = len(value)
        self.ninputs = 0
        self.value = value

    def sanity_check(self):
        if not len(list(set(self.output_cons))) == len(self.value):
            print("Warning for {}(id={}): Not all or too many output connectors are used".format(self.name, self.nid))
        return True

    ### this op ==> nodeobjct ###
    def addToNode(self, nodeobj, connector=0):
        self.output_nodes.append(nodeobj)
        self.output_cons.append(connector)
        
    def __repr__(self):
        return "<Variable>"

    def __str__(self):
        return "variable"
