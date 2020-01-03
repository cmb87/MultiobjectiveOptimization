
import logging
from typing import Callable, Union, Tuple, Optional


class Placeholder():

    def __init__(
        self,
        nid: int,
        name: str = "placeholder",
        noutputs: int = 1,
        ninputs: int = 0,
        value: Optional = None
    ):
        """This is the placeholder object

        Parameters
        ----------
        nid : int
            Node ID
        name : str, optional
            Name of the node
        noutputs : int, optional
            Number of outputs
        ninputs : int, optional
            Number of inputs
        value : Optional, optional
            Value
        """
        self.input_nodes, self.input_cons = [], []
        self.output_nodes, self.output_cons = [], []
        self.name = name
        self.evaluated = False
        self.nid = nid
        self.noutputs = noutputs
        self.ninputs = ninputs

    def sanity_check(self) -> bool:
        """Sanity Check

        Returns
        -------
        bool
            Description
        """
        if not len(list(set(self.output_cons))) == self.noutputs:
            logging.info(f"Warning for {self.name}(id={self.nid}): \
                Not all or too many output connectors are used")
        return True

    def addToNode(self, nodeobj: Callable, connector: int = 0) -> None:
        """Add a receiving node

        Parameters
        ----------
        nodeobj : Callable
            Node
        connector : int, optional
            Description
        """
        self.output_nodes.append(nodeobj)
        self.output_cons.append(connector)

    def __repr__(self):
        return "<Placeholder>"

    def __str__(self):
        return "placeholder"
