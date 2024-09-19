# one vs rest classification:
import numpy as np
import pandas as pd
import os
import json
import sys
import sklearn
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from statistics import mean, stdev
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *
import logging
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.server import get_app_client
from mlflow import MlflowClient
from mlflow.server.auth.client import AuthServiceClient

#Parameters:
'''parameters:
estimator:estimator object
n_jobs int, default=None
verbose int ,default=0'''

def load():
    try:
        with open('conf_ovr.json','r')as file:
            con=json.load(file)
        return 1, con
    except:
        d1 = {'Status':'0', 'Error': 'No such file or Directory', 'Error_code': 'A330'}
        return 0,d1
    
def ovr_data_process(con):
    try:
        df = pd.read_csv(con["ovr_classification"]["input_file"])
        x = df[con["ovr_classification"]["IV"]]
        y = df[con["ovr_classification"]["DV"]]
       
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
                d1 = {'Status': '0', 'Error': 'Null values found in data', 'Error_code': 'A334'}
                return  d1
        else:
            d1 = {'Status': '0', 'Error': 'Unsupported Data', 'Error_code': 'A333'}
            return  d1
    except Exception as e:
        logger = logging.getLogger()
        logger.critical(e)
        d1 = {'Status': '0', 'Error': str(e), 'Error_code': 'A331'}
        return d1
    
def test_train_split(d1):
    X = d1["X"]
    Y = d1["Y"]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(X, Y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = Y[train_index], Y[test_index]
                
    return x_train_fold, y_train_fold, x_test_fold, y_test_fold
    
def ovr_train(x_train_fold, y_train_fold):
    try:
        n=con["ovr_classification"]["n_jobs"]
        if n=='None':
            n1=None
        elif n.isdigit():
            n1=int(n)
        else:
            d1={'Status': '0',"Error":'n_jobs parameter should be None or integer','Error_code': 'A624'}
            return d1
        ovr_classification = OneVsRestClassifier(
                estimator=eval(con["ovr_classification"]["estimator"]),
                n_jobs=n1,
                verbose=int(con["ovr_classification"]["verbose"])
            )
        pipe = make_pipeline(StandardScaler(), ovr_classification)    
        # model:
        pipeline1 = pipe.fit(x_train_fold, y_train_fold)
        return pipeline1
    except Exception as e:
        return str(e)
            
def ovo_train(x_train_fold, y_train_fold):
    try:
        n=con["ovo_classification"]["n_jobs"]
        if n=='None':
            n1=None
        elif n.isdigit():
            n1=int(n)
        else:
            d1={'Status': '0',"Error":'n_jobs parameter should be None or integer','Error_code': 'A624'}
            return d1
        ovo_classification = OneVsOneClassifier(
                estimator=eval(con["ovo_classification"]["estimator"]),
                n_jobs=n1
            )
        pipe = make_pipeline(StandardScaler(), ovo_classification)    
        # model:
        pipeline1 = pipe.fit(x_train_fold, y_train_fold)
        return pipeline1
    except Exception as e:
        return str(e)            

def ovr_cls_evaluation(pipeline1,x_train_fold,y_train_fold,x_test_fold,y_test_fold):
    # #testing with X_train__fold:
    y_pred = pipeline1.predict(x_train_fold)
    confusion_matrix1 = confusion_matrix(y_train_fold, y_pred)
    classification_report1 = classification_report(y_train_fold, y_pred,output_dict=True)
    accuracy1 = accuracy_score(y_train_fold, y_pred) * 100
    precision1 = precision_score(y_train_fold, y_pred, average='weighted') * 100
    recall1 = recall_score(y_train_fold, y_pred, average='weighted') * 100
    F1_score1 = f1_score(y_train_fold, y_pred, average='weighted') 
    
    # testing with X_test__fold:
    y_pred = pipeline1.predict(x_test_fold)
    confusion_matrix2 = confusion_matrix(y_test_fold, y_pred)
    classification_report2 = classification_report(y_test_fold, y_pred,output_dict=True)
    accuracy2 = accuracy_score(y_test_fold, y_pred) * 100
    F1_score2 = f1_score(y_test_fold, y_pred, average='weighted') 
    precision2 = precision_score(y_test_fold, y_pred, average='weighted') * 100
    recall2 = recall_score(y_test_fold, y_pred, average='weighted') * 100

    d = {
        "Train_info":{
            "Train_f1_score":F1_score1,
            "Train_confusion_matrix":confusion_matrix1.tolist(),
            "Train_classification_report":[classification_report1],
            "Train_accuracy":accuracy1,
            "Train_precision":precision1
            },
        
        "Test_info":{
            "Test_f1_score":F1_score2,
            "Test_confusion_matrix":confusion_matrix2.tolist(),
            "Test_classification_report":[classification_report2],
            "Test_accuracy": accuracy2,
            "Test_precision": precision2
            }
    }

    if con["ovr_classification"]["model_generation"] == "Yes":
        path = con["ovr_classification"]["output_path"]
        name = path + con["ovr_classification"]["model_name"] + '.sav'
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
                      
def log_classification_report(report, prefix=""):
    """
    Flattens and logs a classification report dictionary to MLflow.

    Parameters:
    - report: The classification report dictionary.
    - prefix: A prefix for the keys in the flattened dictionary.
    """
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

    # Log the metrics to MLflow
    mlflow.log_metrics(metrics)                      
                      
    
if __name__ == "__main__":
    
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    
    t,con=load()
    if t == 1:
        d1 = ovr_data_process(con)
        x_train_fold, y_train_fold, x_test_fold, y_test_fold = test_train_split(d1)
        
        #for i in range(0,2):
        #    if i==0:
        run_name="ovr"
        pipeline1 = ovr_train(x_train_fold,y_train_fold)
        input_data = con["ovr_classification"]["input_file"]
        data_source = pd.read_csv(input_data)
        dataset = mlflow.data.from_pandas( data_source, source=input_data, name="iris_trans", targets="Species")
        param = {"estimator":con["ovr_classification"]["estimator"], "n_jobs":con["ovr_classification"]["n_jobs"], "verbose":con["ovr_classification"]["verbose"]}
#             else:
#                 run_name="ovo"
#                 pipeline1 = ovo_train(x_train_fold, y_train_fold)
#                 input_data = con["ovo_classification"]["input_file"]
#                 data_source = pd.read_csv(input_data)
#                 dataset = mlflow.data.from_pandas( data_source, source=input_data, name="iris_trans", targets="Species")
#                 param = {"estimator": con["ovo_classification"]["estimator"], "n_jobs":con["ovo_classification"]["n_jobs"]}
                
        d1=ovr_cls_evaluation(pipeline1,x_train_fold, y_train_fold, x_test_fold, y_test_fold)
        print(d1)
        model = pipeline1
        print(model)
                  
        experiment_name = "mclass_plugin"       
        #mlflow.set_tracking_uri("file-plugin://mlflow-tracking/mlruns")
        #path="file:\\\D:\Divya\Projects\MLOPS\MLFLOW\mlflow-tracking\plugintest"
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
       
        # Set the tracking URI to use the file-plugin
        #os.environ["MLFLOW_TRACKING_URI"] = path
                
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_input(dataset, context="training")
            mlflow.log_params(param)
            mlflow.log_metric("Train_f1_score", d1["Metrics"]["Train_info"]["Train_f1_score"])
            mlflow.log_metric("Train_accuracy", d1["Metrics"]["Train_info"]["Train_accuracy"])
            mlflow.log_metric("Train_precision1", d1["Metrics"]["Train_info"]["Train_precision"])
            mlflow.log_metric("Test_f1_score", d1["Metrics"]["Test_info"]["Test_f1_score"])
            mlflow.log_metric("Test_accuracy", d1["Metrics"]["Test_info"]["Test_accuracy"])
            mlflow.log_metric("Test_precision1", d1["Metrics"]["Test_info"]["Test_precision"])
            #mlflow.log_metric("Train_r2_score", d1["Metrics"]["Test_info"]["Test_confusion_matrix"])
            #mlflow.log_metric("Train_r2_score", d1["Metrics"]["Test_info"]["Test_classification_report"])
            
            test_cls_report = d1["Metrics"]["Test_info"]["Test_classification_report"]
            
            if isinstance(test_cls_report, list) and len(test_cls_report) > 0:
                print("control here")
                report_dict = test_cls_report[0]
                log_classification_report(report_dict)
            else:
                print("The report is empty or not in the expected format.")
            
            print(mlflow.MlflowClient().get_run(run.info.run_id).data)      
            mlflow.sklearn.log_model(model, "model",registered_model_name='ovrmcls')
            
        # '''    
        #     #Prediction using logged model        
        #     logged_model = 'runs:/aa5777e7fde24edaa53bfd96fbc6a1fa/model'
        #     # Load model as a PyFuncModel.
        #     loaded_model = mlflow.pyfunc.load_model(logged_model)
          
          
        #     #Register Model
        #     model_name = 'OneVsRest'
        #     run_id= 'a71f1bbfbc4d4d1dbf08f86bb06e559f'
        #     model_uri = f'runs:/{run_id}/model_name'
        #     mlflow.register_model(model_uri=model_uri, name=model_name)
            
        #     print(x_test_fold)
        #     y_pred = loaded_model.predict(pd.DataFrame(x_test_fold))
        #     print(y_pred)   
        # '''
    else:
        print(con)
