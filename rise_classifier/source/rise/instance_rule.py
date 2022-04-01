

from .instance import Instance, CategoricalInstanceAttribute, NumericInstanceAttribute
import numpy as np
from dataclasses import dataclass
from typing import List, Set, Tuple
import abc
from numbers import Number
import abc
import numpy as np
from fastdist import fastdist


@dataclass(unsafe_hash=True, eq=True)
class GenericAttributeRule:
    """
        represent generic rule for an attribute
    """
    attribute_name: str

    def __init__(self, name):
        self.attribute_name = name

    @abc.abstractclassmethod
    def is_covering(self, input: Number):
        """
            checks if attribute rule is covering the atribute of instance
        """
        raise NotImplementedError

    @abc.abstractclassmethod
    def fit(self, input: Number):
        """
            update the rule attribute to fit the instance attribute
        """
        raise NotImplementedError


@dataclass(unsafe_hash=True, eq=True)
class AtributeNumericalRule(GenericAttributeRule):
    """
        rule for numerical attribute of an instance
    """
    lower_bound: float
    upper_bound: float

    def __init__(self, name, lower_bound: float, upper_bound: float):
        super().__init__(name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def is_covering(self, input: float) -> bool:
        return self.lower_bound <= input and input <= self.upper_bound

    def fit(self, value_to_fit: float):
        if value_to_fit <= self.lower_bound:
            self.lower_bound = value_to_fit
        elif value_to_fit >= self.upper_bound:
            self.upper_bound = value_to_fit
        else:
            print('ahtung value is not fitted')

    def __str__(self):
        return f'({self.attribute_name} in [{round(self.lower_bound,3)},{round(self.upper_bound,3)}])'


@dataclass(unsafe_hash=True, eq=True)
class AtributeCategoricalRule(GenericAttributeRule):
    """
        represent categorical rule for an atribute of an instance
    """
    value: int

    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

    def is_covering(self, input: int) -> bool:
        return self.value == input

    def fit(self, value):
        self.value = value

    def __str__(self):
        return f'({self.attribute_name} == {self.value})'


@dataclass(unsafe_hash=True, eq=True)
class InstanceRule:
    """
        represent the collection  of rule for instane
    """

    conclusions: Tuple[GenericAttributeRule]
    label: str
    coverage: float
    precision: float

    def __init__(self, instance: Instance):
        """
            parse the attributes of the instance and turn them into rules
        """
        self.label = instance.label
        conclusions = []
        for attribute in instance.properties:
            if isinstance(attribute, CategoricalInstanceAttribute):
                rule = AtributeCategoricalRule(attribute.attribute_name, attribute.value)
                conclusions.append(rule)
            elif isinstance(attribute, NumericInstanceAttribute):
                rule = AtributeNumericalRule(attribute.attribute_name, attribute.value, attribute.value)
                conclusions.append(rule)
        self.conclusions = tuple(conclusions)
        
        # set it as None as we have "cache" this property
        # and we check when it is none see to_numpy() 
        self.numpy = None
        self.numpy = self.to_numpy()

        # set them as None for hash computing
        # properties will be update the rule infering is done
        self.coverage = None
        self.precision = None

    def __str__(self):
        return f'class[{self.label}]  - coverage[{self.coverage}%] -  precision[{self.precision}%] >> {[str(subr) for subr in self.conclusions]}'

    def get_rule(self, name) -> GenericAttributeRule:
        """
            return the attribute rule by name
        """
        return list(filter(lambda rule: rule.attribute_name == name, self.conclusions))[0]

    def is_covering(self, instance: Instance) -> bool:
        """
            runs through trough the attributes of the instance and check if they are covered by the current rules
            (Attribute rules)
        """
        for attr in instance.properties:
            name = attr.attribute_name
            rule = self.get_rule(name)
            if False == rule.is_covering(attr.value):
                return False
        return True

    def fit(self, instance: Instance):
        """
            AKA as most specific generalization 
            update the attribute rules to fit the instance aka
        """
        for attr in instance.properties:
            name = attr.attribute_name
            rule = self.get_rule(name)
            if not rule.fit(attr.value):
                rule.fit(attr.value)
        self.numpy = self.to_numpy(is_update_call=True)

    def to_numpy(self, is_update_call: bool = False) -> np.ndarray:
        """
            converts the rule to np array used 
            @param is_update_call used when we want to  
        """
        if is_update_call or self.numpy is None:
            ll = []
            for rule in self.conclusions:
                if isinstance(rule, AtributeNumericalRule):
                    # we get the middle between the lower bound and upper_bound
                    # when we have to calculate the distance
                    mid = (rule.lower_bound + rule.upper_bound)/2
                    ll.append(mid)
                elif isinstance(rule, AtributeCategoricalRule):
                    ll.append(rule.value)
            self.numpy = np.array(ll)
        return self.numpy

    def distance(self, instance: Instance) -> float:
        """
            compute the distance between rule and instance
        """
        return fastdist.euclidean(self.numpy, instance.numpy)

    def coverage_precision(self, instances: List[Instance]):
        """
            computes coverage and precision for current rule
            based on the input 
        """
        size = len(instances)
        coverage_counter, precision_counter = 0, 0
        for instance in instances:
            
            if self.is_covering(instance):
                coverage_counter +=1 
                if self.label == instance.label:
                    precision_counter +=1

        self.coverage = round((coverage_counter / size)*100,3)
        self.precision = round((precision_counter / coverage_counter)*100,3)
