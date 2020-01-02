### This is the placeholder object ###
class Placeholder():

    def __init__(self, nid, name="placeholder", noutputs=1, ninputs=0, value=None):
        self.input_nodes, self.input_cons = [], []
        self.output_nodes, self.output_cons = [], []
        self.name = name
        self.evaluated = False
        self.nid = nid
        self.noutputs = noutputs
        self.ninputs = ninputs

    def sanity_check(self):
        if not len(list(set(self.output_cons))) == self.noutputs:
            print("Warning for {}(id={}): Not all or too many output connectors are used".format(self.name, self.nid))
        return True

    ### this op ==> nodeobjct ###
    def addToNode(self, nodeobj, connector=0):
        self.output_nodes.append(nodeobj)
        self.output_cons.append(connector)
        
    def __repr__(self):
        return "<Placeholder>"

    def __str__(self):
        return "placeholder"
