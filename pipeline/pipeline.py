import numpy as np
import matplotlib.pyplot as plt


class Operation():

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            # Assign all input nodes this node as output node
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def __repr__(self):
        return "Master operation class"

    def __str__(self):
        return "member of Test"

    def compute(self):
        pass


class add(Operation):   # Should be upper case but to follow with TFs naming convention ==> lower case

    def __init__(self, x, y):
        super().__init__([x, y]) # Initialize Operation class

    def __repr__(self):
        return "Add operation class"

    def __str__(self):
        return "Add, member of Operation"

    def compute(self, x_var, y_var): # Overwrites the compute function from Operation
        self.inputs = [x_var, y_var]
        return x_var + y_var




class multiply(Operation):

    def __init__(self, x, y):
        super().__init__([x, y]) # Initialize Operation

    def __repr__(self):
        return "Multiply operation class"

    def __str__(self):
        return "Multiply, member of Operation"

    def compute(self, x_var, y_var): # Overwrites the compute function from Operation
        self.inputs = [x_var, y_var]
        return x_var * y_var

class matmul(Operation):

    def __init__(self, x, y):
        super().__init__([x, y]) # Initialize Operation

    def __repr__(self):
        return "Matmul operation class"

    def __str__(self):
        return "Matmul, member of Operation"

    def compute(self, x_var, y_var): # Overwrites the compute function from Operation
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


class sigmoid(Operation):

    def __init__(self, z):
        super().__init__([z]) # Initialize Operation

    def __repr__(self):
        return "Sigmoid operation class"

    def __str__(self):
        return "Sigmoid, member of Operation"

    def compute(self, z_val):
        return 1 / (1+np.exp(-z_val))

    def plot(self):
        zs = np.linspace(-10,10,100)
        plt.plot(zs, self.compute(zs))
        plt.show()



class Placeholder():

    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)

    def __repr__(self):
        return "Placeholder class"

    def __str__(self):
        return "Placeholder"



class Variable():

    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)

    def __repr__(self):
        return "Variable class"

    def __str__(self):
        return "Variable"


class Graph():

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def __repr__(self):
        return "Graph class"

    def __str__(self):
        return "Graph"

    def set_as_default(self):
        global _default_graph
        _default_graph = self



class Session():

    # Now that graph has all the nodes we need to execute all the ops
    # within a Session ==> we will use PostOrder Tree Traversal to make
    # make sure we execute the nodes in the correct order.
    @staticmethod
    def traverse_postorder(operation):
        """
        PostOrder Traversal of Nodes. makes sure computations are done in correct order
        :param operation:
        :return:
        """
        nodes_postorder = []

        def recurse(node):
            if isinstance(node, Operation):
                #print(node, node.input_nodes)
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)

        recurse(operation)
        return nodes_postorder


    def run(self, operation, feed_dict={}):

        nodes_postorder = Session.traverse_postorder(operation)

        #print(nodes_postorder)
        for node in nodes_postorder:

            if type(node) == Placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value

            else:   # Its an operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)    # args asterixs

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output

