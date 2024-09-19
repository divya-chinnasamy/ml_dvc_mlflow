import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np

# Load and preprocess your dataset
data = pd.read_csv("telechurn.csv")
data.isnull().sum()
X = data.drop(columns=['Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)

# Start an MLflow experiment
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "logi_reg_exp"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

mlflow.sklearn.autolog()

with mlflow.start_run() as run:
      # Create and train the model
    model1 = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
                                fit_intercept=True, intercept_scaling=1, 
                                class_weight=None, random_state=None, solver='liblinear', 
                                max_iter=100, multi_class='ovr', verbose=0)
    model1.fit(X_train, y_train)

      # Make predictions on the test set
    y_pred = model1.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

      # Save the model
    #mlflow.sklearn.save_model(model1, "logi_regression1")
    
    # Log Model 
    mlflow.sklearn.log_model(model1, "model",registered_model_name='logi_reg')
    print("accuracy", accuracy)
    print("precision",precision)
    print("recall:",recall)
    print("f1 score:", f1)

# Close the MLflow run
mlflow.end_run()