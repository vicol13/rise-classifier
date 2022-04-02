from sklearn.metrics import accuracy_score
from typing import List
from .instance import Instance
from .instance_rule import InstanceRule
import numpy as np


class RiseUtils:
    
    """
        class which represent util for the rise classifer, here we calculacte the distance between 
        rules in the distance etc. All methods were moved to different class because it is easier to experiment
        with different distnaces and accuracy metrics
    """
    
    @staticmethod
    def rules_accuracy(instances: List[Instance], rules: List[InstanceRule],stats_dict:dict)->float:
        """
            calculate the accuracy of rules on the instances
        """
        predicted = 0
        for instance in instances:
            y_hat = RiseUtils.closest_rule(instance, rules,stats_dict)
            predicted += 1 if y_hat == instance.label else 0  
        return predicted/len(instances)

    @staticmethod
    def closest_rule(instance: Instance, rules: List[InstanceRule],stats_dict:dict)->str:
        """
            return the label of the closest_rule rule for instance 
        """
        closest_rule = None
        min = np.inf

        for rule in rules:
            distance = rule.distance(instance,stats_dict)
            if distance < min:
                min = distance
                closest_rule = rule
        return closest_rule.label

    @staticmethod
    def closest_instance(rule: InstanceRule, instances: List[Instance],stats_dict:dict) -> Instance:
        """
            return the closest_instance instance for rule
        """
        # filter out instances which are not covered and have different label
        new_list = list(filter(lambda instance: not rule.is_covering(instance) and rule.label == instance.label, instances))
        assert len(new_list) != 0, "List with instances is empty"
        closest_instance = None
        min = np.inf 
        for instance in new_list:
            distance = rule.distance(instance,stats_dict)
            if min>distance:
                min = distance
                closest_instance = instance
        return closest_instance
