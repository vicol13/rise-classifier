import sys
import os
import glob
from turtle import title

import pandas as pd 
from data_utils import DataUtils
from rise.rise_classifier import RiseClassifier
import scipy
from sklearn.metrics import classification_report,balanced_accuracy_score,f1_score,precision_score,recall_score,accuracy_score
import time
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate

DATA_ROOT  = 'rise_classifier/data/'



def print_metrics(y_true,y_predicted,title=""):
    print(f"========================{title}===================================")
    balanced_acc = round(balanced_accuracy_score(y_true,y_predicted)*100,2)
    accuracy  = round(accuracy_score(y_true,y_predicted)*100,2)
    f1 = round(f1_score(y_true,y_predicted,average='weighted')*100,2)
    precision = round(precision_score(y_true,y_predicted, average='weighted')*100,2)
    recall =  round(recall_score(y_true,y_predicted,average='weighted')*100,2)
    tabulatet = tabulate([['balanced accuracy', balanced_acc], ['accuracy', accuracy],['f1', f1],['precision', precision], ['recall',recall]], headers=['Metric', 'Score'], tablefmt='orgtbl')
    print(tabulatet)
    print(f"\n\n\t\tClassification report [{title}] ::" )
    print(classification_report(y_true,y_predicted))
    print("================================================================")




if __name__ == "__main__":
    start = time.time()
    dataset = f'{sys.argv[1]}'
    path = f'{DATA_ROOT}{dataset}'

    if not os.path.isfile(path) :
        print(f"dataset doesn't exist only avail datasetes [{glob.glob(f'{DATA_ROOT}*.csv')}]")
        exit
    
    iterations = 2
    df = pd.read_csv(path)
    x_train, y_train, x_test, y_test = DataUtils.split_into_folds(df,iterations)
    # scipy.special.seterr('ingore')
    for x_train,y_train,x_test,y_test in zip(x_train,y_train,x_test,y_test):
      
        rise_classifier = RiseClassifier(x_train.columns)
    
        rise_classifier.fit(x_train,y_train)
  
        predicted_y_train = rise_classifier.predict(x_train)
        actual_y_train = y_train.to_numpy().flatten()
        print(f"Final Rules len {len(rise_classifier.rules)}")
        print_metrics(actual_y_train,predicted_y_train,title="train")


        predicted_y_test = rise_classifier.predict(x_test)
        actual_y_test = y_test.to_numpy().flatten()
        print_metrics(actual_y_test,predicted_y_test,title="test")
        
      
        print("Rules set ::")
        for rule in rise_classifier.rules:
            print(f'\t\t{str(rule)}')

       
        end = time.time()
        print(f'execution time [{round(end - start,3)}] seconds') 


