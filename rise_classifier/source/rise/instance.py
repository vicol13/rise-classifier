
from dataclasses import dataclass
from numbers import Number
from typing import Tuple
import numpy as np


@dataclass(unsafe_hash=True, eq=True)
class GenericInstanceAttribute(object):
    """
        used as generic representation for attributes of instances
    """
    attribute_name: str

    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name


@dataclass(unsafe_hash=True, eq=True)
class NumericInstanceAttribute(GenericInstanceAttribute):
    """
        represent numerica attribute of instance
    """
    value: float

    def __init__(self, attribute_name: str, value: float):
        super().__init__(attribute_name)
        self.value = value


@dataclass(unsafe_hash=True, eq=True)
class CategoricalInstanceAttribute(GenericInstanceAttribute):
    value: int

    def __init__(self, attribute_name: str, value: int):
        super().__init__(attribute_name)
        self.value = value


@dataclass(unsafe_hash=True, eq=True)
class Instance(object):
    """
        represent the instance (df row)
    """
    label: str
    properties: Tuple[GenericInstanceAttribute]

    def __init__(self, label: str, properties: Tuple[GenericInstanceAttribute]):
        self.label = label
        self.properties = properties
        self.numpy = np.array([prop.value for prop in self.properties])


    @staticmethod
    def build(label: str, attributes_metadata: Tuple[str, Number, bool]) -> 'Instance':
        """
            builder method to keep boilerplate code inside the class 
        """
        llist = []
        for attr_metadata in attributes_metadata:
            # check if attribute is categorical
            if attr_metadata[2]:
                attr = CategoricalInstanceAttribute(attr_metadata[0], attr_metadata[1])
            else:
                attr = NumericInstanceAttribute(attr_metadata[0], attr_metadata[1])
            llist.append(attr)

        return Instance(label, llist)
