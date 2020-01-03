
import logging
from typing import Callable, Union, Tuple, Optional


class Variable():

    def __init__(
        self,
        nid: int,
        name: str = "variable",
        value: list = [0],
        ninputs: int = 0,
        noutputs: int = 0
    ) -> None:
        """This is the Variable object

        Parameters
        ----------
        nid : int
            Description
        name : str, optional
            Description
        value : list, optional
            Description
        ninputs : int, optional
            Description
        noutputs : int, optional
            Description
        """
        self.input_nodes, self.input_cons = [], []
        self.output_nodes, self.output_cons = [], []
        self.name = name
        self.evaluated = False
        self.nid = nid
        self.noutputs = len(value)
        self.ninputs = 0
        self.value = value


    def sanity_check(self) -> bool:
        """Sanity check for this node

        Returns
        -------
        bool
            Description
        """
        if not len(list(set(self.output_cons))) == len(self.value):
            logging.info(f"Warning for {self.name}(id={self.nid}): Not all or \
                too many output connectors are used")
        return True


    def addToNode(self, nodeobj: Callable, connector: int = 0) -> None:
        """Add a receiving node

        Parameters
        ----------
        nodeobj : Callable
            Description
        connector : int, optional
            Description
        """
        self.output_nodes.append(nodeobj)
        self.output_cons.append(connector)


    def __repr__(self):
        return "<Variable>"

    def __str__(self):
        return "variable"
