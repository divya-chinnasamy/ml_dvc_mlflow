# Logistic Regression Algorithm:
import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import mlflow
from mlflow.models.signature import infer_signature
from datetime import datetime
import dvc.api

# parameters:
'''penalty{‘l1’, ‘l2’, ‘elasticnet’}, default=’l2’
dual bool, default=False
tol float, default=1e-4
C float, default=1.0
fit_intercept bool, default=True
intercept_scaling float, default=1.0
class_weight dict or ‘balanced’, default=None
random_state int, RandomState instance, default=None
solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
max_iter int, default=100
multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
verbose int, default=0
warm_start bool, default=False
n_jobs int, default=None
l1_ratio float, default=None'''

def load():
    try:
        with open("conf_cls.json",'r')as file:
            con=json.load(file)
        return 1, con
    except:
        d1 = {'Status':'0', 'Error': 'No such file or Directory', 'Error_code': 'A345'}
        return 0,d1


def log_reg_data_process(con):
    try:
        
        # Get URL from dvc
        path = "new_iris.csv"  # wine-quality.csv.dvc file presence is enough if dvc push to remote storage is done
        repo = "https://github.com/divya-chinnasamy/ml_dvc_mlflow.git" 
        version = "dv1.0" # git tag -a 'v1' -m 'removed 1000 lines' command is required
        
        data_url = dvc.api.get_url(path, repo=repo) #,   rev=version()
        df = pd.read_csv(data_url, sep=";")
        
        #df = pd.read_csv(con["logistic_reg_classification"]["input_file"])
        x = df[con["logistic_reg_classification"]["IV"]]
        y = df[con["logistic_reg_classification"]["DV"]]
                
        x_support = x.apply(lambda row: all(isinstance(value, (int, float)) for value in row), axis=1).all()
        y_support = y.apply(lambda row: all(isinstance(value, (int, float)) for value in row), axis=1).all()
        if x_support == True and y_support == True:
            check_null1 = x.isna().any().any()
            check_null2 = y.isna().any().any()
            if check_null1 == False and check_null2 == False:
                X = np.asarray(x)
                Y = np.asarray(y)
                d1 = {'X': X, 'Y': Y}
                return d1
            else:
                d1 = {'Status': '0', 'Error': 'Null values found in data', 'Error_code': 'A234'}
                return  d1
        else:
            d1 = {'Status': '0', 'Error': 'Unsupported Data', 'Error_code': 'A233'}
            return  d1

    except Exception as e:
        logger = logging.getLogger()
        logger.critical(e)
        d1 = {'Status': '0', 'Error': str(e), 'Error code': 'A231'}
        return d1
    
def train_test_split(d1):  
    try:
        X = d1["X"]
        Y = d1["Y"]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        for train_index, test_index in skf.split(X, Y):
            x_train_fold, x_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = Y[train_index], Y[test_index]
        
        return x_train_fold, y_train_fold, y_test_fold, x_test_fold
    except Exception as e:
        print(str(e))
    
def log_reg_cls_train(x_train_fold, y_train_fold):
    try:
        h=con["logistic_reg_classification"]["class_weight"]
        if h == "balanced" or h == 'balanced_subsample':
            class_weights = h
        elif h == 'None':
            class_weights = None
        else:
            class_weights = eval(h)
        d=con["logistic_reg_classification"]["dual"]
        if d == 'True':
            d1=True
        elif d =='False':
            d1=False
        fit=con["logistic_reg_classification"]["fit_intercept"]
        if fit =='True':
            f1=True
        elif fit =='False':
            f1=False
        warm=con["logistic_reg_classification"]["warm_start"]
        if warm =='True':
            w1=True
        elif warm =='False':
            w1=False
        p1=con["logistic_reg_classification"]["penalty"]
        if p1=='l1' or p1=='l2' or p1=='elasticnet':
            p3=p1
        elif p1 == 'None':
            p3=None
        else:
            d1={'Status': '0',"Error":'penalty parameter should be None or (l1,l2,elasticnet)','Error_code': 'A619'}
            return d1
        r1=con["logistic_reg_classification"]["random_state"]
        if r1=='None':
            r2=None
        elif r1.isdigit():
            r2=int(r1)
        else:
            d1={'Status': '0',"Error":'random_state parameter should be None or integer','Error_code': 'A620'}
            return d1
        n=con["logistic_reg_classification"]["n_jobs"]
        if n == 'None':
            n1=None
        elif n.isdigit():
            n1=int(n)
        else:
            d1={'Status': '0',"Error":'n_jobs parameter should be None or integer','Error_code': 'A621'}
            return d1
        l1=con["logistic_reg_classification"]["l1_ratio"]
        if l1 =='None':
            l2=None
        elif l1.replace('.', '', 1).isdigit():
            l2=float(l2)
        else:
            d1={'Status': '0',"Error":'l1_ratio parameter should be None or float','Error_code': 'A622'}
            return d1
        logistic_classification = LogisticRegression(
            penalty=p3,
            dual=d1,
            tol=float(con["logistic_reg_classification"]["tol"]),
            C=float(con["logistic_reg_classification"]["C"]),
            fit_intercept=f1,
            intercept_scaling=float(con["logistic_reg_classification"]["intercept_scaling"]),
            class_weight=class_weights,
            random_state=r2,
            solver=str(con["logistic_reg_classification"]["solver"]),
            max_iter=int(con["logistic_reg_classification"]["max_iter"]),
            multi_class=str(con["logistic_reg_classification"]["multi_class"]),
            verbose=int(con["logistic_reg_classification"]["verbose"]),
            warm_start=w1,
            n_jobs=n1,
            l1_ratio=l2
        )
        pipe = make_pipeline(StandardScaler(), logistic_classification)     
        # model:
        pipeline1 = pipe.fit(x_train_fold, y_train_fold)
        return pipeline1
    except Exception as e:
        return str(e)
    
def log_reg_cls_evaluation(pipeline1, x_train_fold, y_train_fold, x_test_fold, y_test_fold):
    try:            
        #testing with X_train__fold:
        y_pred = pipeline1.predict(x_train_fold)
        confusion_matrix1 = confusion_matrix(y_train_fold, y_pred)
        classification_report1 = classification_report(y_train_fold, y_pred,output_dict=True)
        F1_score1 = f1_score(y_train_fold, y_pred, average='weighted') 
        
        # testing with X_test__fold:
        y_pred = pipeline1.predict(x_test_fold)
        confusion_matrix2 = confusion_matrix(y_test_fold, y_pred)
        classification_report2 = classification_report(y_test_fold, y_pred,output_dict=True)
        F1_score2 = f1_score(y_test_fold, y_pred, average='weighted') 
        
        d = {
            "Train_info":{
                "Train_f1_score":F1_score1,
                "Train_confusion_matrix":confusion_matrix1.tolist(),
                "Train_classification_report":[classification_report1]
                },
            
            "Test_info":{
                "Test_f1_score":F1_score2,
                "Test_confusion_matrix":confusion_matrix2.tolist(),
                "Test_classification_report":[classification_report2]
                }
        }
        
        if con["logistic_reg_classification"]["model_generation"] == "Yes":
            path = con["logistic_reg_classification"]["output_path"]
            name = path + con["logistic_reg_classification"]["model_name"] + '.sav'
            if os.path.exists(path):
                pickle.dump(pipeline1, open(name, 'wb'))
                pipeline1 = None
                b = "Model Generated Successfully"
                d1 = {"Status":'1', "Message":b, "Metrics":d, "download_path":name,"model_status":'1'}
                return d1
            else:
                os.mkdir(path)
                pickle.dump(pipeline1, open(name, 'wb'))
                pipeline1 = None
                b = "Model Generated Successfully"
                d1 = {"Status":'1', "Message":b, "Metrics":d, "download_path":name,"model_status":'1'}
                return d1
        else:
            b = "Please Ensure Model Generation option is selected"
            d1 = {"Status":'1', "Message":b, "Metrics":d,"model_status":'0'}
            return d1
    except Exception as e:
        return str(e)

def log_classification_report(report,prefix=""):
    metrics = {}
    # Recursively flatten the dictionary
    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            new_key = f'{parent_key}_{k}' if parent_key else k
            if isinstance(v, dict):
                flatten_dict(v, new_key)
            else:
                metrics[new_key] = v
    flatten_dict(report)
    mlflow.log_metrics(metrics) 

if __name__ == "__main__":
    #conf = sys.argv[1] if len(sys.argv) > 1 else "conf_cls.json"
    
    
    t,con=load()
    if t == 1:
        d1 = log_reg_data_process(con)
        x_train_fold, y_train_fold, y_test_fold, x_test_fold = train_test_split(d1)
        pipeline1 = log_reg_cls_train(x_train_fold, y_train_fold)
        d1=log_reg_cls_evaluation(pipeline1,x_train_fold, y_train_fold, x_test_fold, y_test_fold)
        print(d1)
        
        model = pipeline1
        print(model)
        
        #experiment name
        exp_timestamp = datetime.now().strftime("%Y%m%d")
        exp_name = "logistic_reg" + exp_timestamp
        print(exp_name)

        run_timestamp = datetime.now().strftime("%Y%m%d--%H%M%S")
        run_name = "log_reg"+ run_timestamp
        print(run_name)
        
        # Log input data using mlflow
        input_data = con["logistic_reg_classification"]["input_file"]
        data_source = pd.read_csv(input_data)
        dataset = mlflow.data.from_pandas( data_source, source=input_data) #, name="regression_test", targets="price")
        
        params = {
                "class_weight": con["logistic_reg_classification"]["class_weight"],
                 "dual": con["logistic_reg_classification"]["dual"],
                 "fit_intercept": con["logistic_reg_classification"]["fit_intercept"],
                 "warm_start": con["logistic_reg_classification"]["warm_start"],
                 "penalty": con["logistic_reg_classification"]["penalty"],
                 "random_state": con["logistic_reg_classification"]["random_state"],
                 "n_jobs": con["logistic_reg_classification"]["n_jobs"],
                 "l1_ratio": con["logistic_reg_classification"]["l1_ratio"],
                 "tol": con["logistic_reg_classification"]["tol"],
                 "C": con["logistic_reg_classification"]["C"],
                 "solver": con["logistic_reg_classification"]["solver"],
                 "max_iter":con["logistic_reg_classification"]["max_iter"],
                 "multi_class":con["logistic_reg_classification"]["multi_class"],
                 "verbose":con["logistic_reg_classification"]["verbose"]
                 }
        metrics_data = {
            "Train_r2_score":d1["Metrics"]["Train_info"]["Train_f1_score"],
            "Test_r2_score":d1["Metrics"]["Test_info"]["Test_f1_score"],
            # "Train_accuracy":d1["Metrics"]["Train_info"]["Train_accuracy"],
            # "Train_precision1":d1["Metrics"]["Train_info"]["Train_precision1"],
            # "Test_accuracy":d1["Metrics"]["Test_info"]["Test_accuracy"],
            # "Test_precision1":d1["Metrics"]["Test_info"]["Test_precision1"]
        }
        
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(exp_name)

        # ML Flow run starts
        with mlflow.start_run() as run_name:
            mlflow.log_input(dataset, context="log_training")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics_data)
            mlflow.sklearn.log_model(model, "model",registered_model_name='lr_cls')
            
            test_cls_report = d1["Metrics"]["Test_info"]["Test_classification_report"]
            
            if isinstance(test_cls_report, list) and len(test_cls_report) > 0:
                print("control here")
                report_dict = test_cls_report[0]
                log_classification_report(report_dict)
            else:
                print("The report is empty or not in the expected format.")
            
            print(mlflow.MlflowClient().get_run(run_name.info.run_id).data)      
            
            
          
    else:
        print(con)
