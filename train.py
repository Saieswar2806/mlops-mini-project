import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn
print("Program started")
#load the data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target

#split the data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state = 42)


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris Classification")
#mlflow experiment tracking
with mlflow.start_run():

    estimators = 10
    depth = 3

    mlflow.log_param("n_estimators", estimators)
    mlflow.log_param("max_depth", depth)

    model=RandomForestClassifier(n_estimators=estimators,max_depth=depth,random_state=42)
    #train the model
    model.fit(X_train,Y_train)

    #test the model
    accuracy = model.score(X_test,Y_test)
    print(f"Model trained! Accuracy:{accuracy*100:.2f}%")

    #log accuracy
    mlflow.log_metric("accuracy",accuracy)

    #save the model
    joblib.dump(model,"iris_model.pkl")
    
    #store a version of the model in its registry
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print("Experiment logged successfully to MLflow!")