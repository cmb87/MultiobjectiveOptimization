import logging
from typing import Union, NewType, Callable, Optional

import numpy as np
from .variables import Variable
from .placeholders import Placeholder

PlaceholderType = NewType("PlaceholderType", Placeholder)
VariableType = NewType("VariableType", Variable)


class Operation:
    def __init__(
        self,
        nid: Union[int, str],
        name: str = "node",
        noutputs: int = 1,
        ninputs: int = 1,
        value: Optional = None,
    ) -> None:
        """This is the operation class object

        Parameters
        ----------
        nid : int
            Description
        name : str, optional
            Description
        noutputs : int, optional
            Description
        ninputs : int, optional
            Description
        """
        self.input_nodes, self.input_cons = [], []
        self.output_nodes, self.output_cons = [], []
        self.name = name
        self.nid = nid
        self.evaluated = False
        self.noutputs = noutputs
        self.ninputs = ninputs
        self.value = value

    def sanity_check(self) -> bool:
        """Sanity check

        Returns
        -------
        bool
            Healty or not
        """
        if len(self.input_nodes) == 0:
            logging.info(
                f"Warning for {self.name}(id={self.nid}): No input \
                specified. Is this node  needed? Add a connection or remove \
                this node!"
            )
            return False

        if not len(list(set(self.input_cons))) == self.ninputs:
            logging.info(
                f"Warning for {self.name}(id={self.nid}): Not all \
                or too many input connectors are used"
            )

        if not len(list(set(self.output_cons))) == self.noutputs:
            logging.info(
                f"Warning for {self.name}(id={self.nid}): Not all \
                or too many output connectors are used"
            )

        return True

    def addFromNode(
        self,
        nodeobj: Union[VariableType, Callable, PlaceholderType],
        connector: int = 0,
    ) -> None:
        """Add a sending node

        Parameters
        ----------
        nodeobj : Union[VariableType, Callable, PlaceholderType]
            Description
        connector : int, optional
            Description
        """
        self.input_nodes.append(nodeobj)
        self.input_cons.append(connector)

    def addToNode(
        self,
        nodeobj: Union[VariableType, Callable, PlaceholderType],
        connector: int = 0,
    ):
        """Add a receiving node

        Parameters
        ----------
        nodeobj : Union[VariableType, Callable, PlaceholderType]
            Description
        connector : int, optional
            Description
        """
        self.output_nodes.append(nodeobj)
        self.output_cons.append(connector)

    def __repr__(self):
        return "<{}>".format(self.name)

    def __str__(self):
        return "<{}>".format(self.name)

    def compute(self) -> None:
        """Evaluate node
        """
        pass


class FunctionOperation(Operation):
    def __init__(
        self,
        nid: int,
        name="node",
        noutputs: int = 1,
        ninputs: int = 1,
        function: Union[Callable, None] = None,
        value: Optional = None,
    ) -> None:
        """FunctionOperation

        Parameters
        ----------
        nid : int
            Node ID
        name : str, optional
            Name of the node
        noutputs : int, optional
            Number of outputs
        ninputs : int, optional
            Description
        """
        super().__init__(nid, name, noutputs, ninputs, value=value)
        self.function = function

    def compute(self, X: np.ndarray) -> list:
        """Evaluate function

        Parameters
        ----------
        X : np.ndarray
            Design variables

        Returns
        -------
        list
            Function response wrapped in list
        """
        return [self.function(X)]
