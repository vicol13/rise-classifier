import pandas as pd
import copy

from .rise_utils import RiseUtils
from data_utils import DataUtils
from .instance import Instance
from .instance_rule import InstanceRule
from typing import List
from .progress_bar import progress_bar

SHOW_PROGRESS_BAR = False 

class RiseClassifier:
    """
        class which represent the actual Rise classifer were the algorithm is implemented

        todo:
            1. add some validation for columns on predict in case df contains some additional unprocessed columns 
    """

    def __init__(self,columns):
        self.columns = columns
        

    def fit(self,x_df:pd.DataFrame,y_df:pd.DataFrame):
        """
            converts the dataframe to instance and the infer and extend the rules
        """
        df = self.__process_df(x_df.copy())
        self.instances = self.__df_to_instances(df.to_numpy(),y_df.to_numpy().flatten())
        
        # induct the rules from instances  using leave-one-out
        self.rules = list(set([InstanceRule(inst) for inst in self.instances]))
        final_precision = RiseUtils.rules_accuracy(self.instances,self.rules) 
        rules_prime = copy.copy(self.rules)
       
        print(f'intial len of rule {len(rules_prime)}')
        while True:
            initial_precision = final_precision
            for index,rule in enumerate(progress_bar(self.rules)) if SHOW_PROGRESS_BAR else enumerate(self.rules):
                # in some cases we are popping rules from list and the loop
                # is executed len(initial_list) which smaller the index
                if index >= len(self.rules):
                    break
            
                # instance nearest to the rule not covered by the rule and being the same class
                closest_instance = RiseUtils.closest_instance(rule,self.instances)
                # most specific generalization
                rule_prime = copy.copy(rule)
                rule_prime.fit(closest_instance)
                # update set with new rule
                rules_prime[index] = rule_prime
                
                if RiseUtils.rules_accuracy(self.instances,rules_prime) >= RiseUtils.rules_accuracy(self.instances,self.rules):
                    if rule_prime in self.rules:
                        rules_prime.pop(index)
                    self.rules = rules_prime
                    rules_prime = copy.copy(self.rules)
                else:
                    rules_prime[index] = rule
        
            final_precision = RiseUtils.rules_accuracy(self.instances,self.rules)
            print(f'Iteration over rules is done initial precision: [{initial_precision}]  final precision:[{final_precision}]')
            if final_precision <= initial_precision:
                break
        [rule.coverage_precision(self.instances) for rule in self.rules]

            

    def predict(self,x_df:pd.DataFrame)->List[str]:
        """
            predicts the input datafram 
        """
        # mock label for prediction in order to build the instance
        y_mock = [ None for i in range(len(x_df))]
        df  = self.__proces_df_predict(x_df)
        instances = self.__df_to_instances(df.to_numpy(),y_mock)
        return [RiseUtils.closest_rule(instance,self.rules) for instance in instances]


    def __df_to_instances(self,x_df:pd.DataFrame,y_df:pd.DataFrame)->List[Instance]:
        """
            converts dataframe into list of instances 
        """
        instances = []
        categorical_colums = self.categorical_columns
        for x,y in zip(x_df,y_df):
            # here we preparte the atributes fro Instance
            # we keep in tuple format where (column_name,value,is_this_attribute_categorical)
            attributes = [(column,value, column in categorical_colums) for column,value in zip(self.columns,x)]
            instance = Instance.build(y,attributes)
            instances.append(instance)
        return instances 


    def __process_df(self,df:pd.DataFrame):
        """
            apply ordinal enconding and normalization for df attributes
            also save the attribute encoding in self.categorical_econding

            this method is used when we have to fit the dataframe for intial processing
        """
        categorical_columns = []
        self.column_transformers = {}
        for column in self.columns:
            if DataUtils.is_categorical(df[column]):
                encoded,encoder = DataUtils.encode(df[column])
                categorical_columns.append(column)
                self.column_transformers[column] = encoder
                df[column] = encoded
            elif DataUtils.is_numeric(df[column]):
                scalled,scaller = DataUtils.normalize(df[column])
                self.column_transformers[column] = scaller
                df[column] = scalled
        self.categorical_columns = set(categorical_columns)
        return df

    
    def __proces_df_predict(self,x_df:pd.DataFrame):
        """
            preprocess the input dataframe using the same transformers
            which were fitted on the train data set so the values are in the same ranges
        """
        df_copy = x_df.copy()
        for column in self.column_transformers.keys():
            transformer = self.column_transformers[column]
            arr = df_copy[column].to_numpy().reshape(-1,1)
           
            if column in self.categorical_columns:
                df_copy[column] = transformer.transform(arr).astype(int)
            else:
                df_copy[column] = transformer.transform(arr)
        return df_copy
