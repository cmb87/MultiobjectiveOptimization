import logging
import numpy as np

from test_functions import rosenbrock
from pipeline.graph import Graph
from pipeline.operations import Operation
from pipeline.placeholders import Placeholder
from pipeline.variables import Variable


# Set logging formats
logging.basicConfig(
    level=logging.INFO,
    format=("[%(filename)8s] [%(levelname)4s] :  %(funcName)s - %(message)s"),
)


if __name__ == "__main__":
    if True:
        np.random.seed(42)

        ### Special class for single processes ###
        class Rosenbrock(Operation):
            def compute(self, X):
                return [rosenbrock(X)]

        class Add(Operation):
            def compute(self, x, y):
                return [x+y]


        graph = Graph()
        graph.addNodeToGraph(Placeholder(nid="0", name="Placeholder", noutputs=2))
        graph.addNodeToGraph(Rosenbrock(nid="1", name="Rosenbrock",   noutputs=1))
        graph.addNodeToGraph(Rosenbrock(nid="2", name="Rosenbrock2",  noutputs=1))
        graph.addNodeToGraph(Rosenbrock(nid="3", name="Rosenbrock3",  noutputs=1))
        graph.addNodeToGraph(Add(nid="4", name="Add",  noutputs=1, ninputs=2))
        graph.addNodeToGraph(Rosenbrock(nid="5", name="Rosenbrock5",  noutputs=1))

        graph.addConnectionToGraph("0","1", snid_con=0, rnid_con=0)
        graph.addConnectionToGraph("0","2", snid_con=1, rnid_con=0)
        graph.addConnectionToGraph("2","3", snid_con=0, rnid_con=0)
        graph.addConnectionToGraph("0","4", snid_con=0, rnid_con=0)
        graph.addConnectionToGraph("0","4", snid_con=1, rnid_con=1)
        graph.addConnectionToGraph("4","5", snid_con=0, rnid_con=0)

        flowchart = graph.generateFlowchartFromGraph()
        graph.generateGraphFromFlowchart(flowchart)
        flowchart = graph.generateFlowchartFromGraph()
        #logging.info(json.dumps(flowchart, indent=4, sort_keys=True))

        feed_dict = {"0": [np.random.rand(10,2), np.ones((10,2))]}
        ops = ["4", "5", "3"]
        res = graph.run(feed_dict, ops)

        logging.info(res)


    else:
        class StringSplit(Operation):
            def compute(self, mystring):
                ihalf = int(0.5*len(mystring))
                return [mystring[:ihalf], mystring[ihalf:]]

        class Upper(Operation):
            def compute(self, mystring):
                return [mystring.upper()]

        class Lower(Operation):
            def compute(self, mystring):
                return [mystring.lower()]


        graph = Graph()
        graph.addNodeToGraph(Placeholder(nid="0", name="Placeholder", noutputs=1))
        graph.addNodeToGraph(StringSplit(nid="1", name="Splitter", noutputs=2, ninputs=1))
        graph.addNodeToGraph(Upper(nid="2", name="Upper", noutputs=1, ninputs=1))
        graph.addNodeToGraph(Lower(nid="3", name="Lower", noutputs=1, ninputs=1))
        graph.addNodeToGraph(Variable(nid="4", name="Variable", value=["You WANT me"]))
        graph.addNodeToGraph(Lower(nid="5", name="Lower", noutputs=1, ninputs=1))

        graph.addConnectionToGraph("0","1", snid_con=0, rnid_con=0)
        graph.addConnectionToGraph("1","2", snid_con=0, rnid_con=0)
        graph.addConnectionToGraph("1","3", snid_con=1, rnid_con=0)
        graph.addConnectionToGraph("4","5", snid_con=0, rnid_con=0)

        feed_dict = {"0": ["Hello There"]}

        flowchart = graph.generateFlowchartFromGraph()
        #logging.info(json.dumps(flowchart, indent=4, sort_keys=True))
        graph.generateGraphFromFlowchart(flowchart)

        res = graph.run(feed_dict, ["2","3", "5"])

        logging.info(graph.variables)

        logging.info(res)
