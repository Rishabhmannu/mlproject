import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
  try:
    dir_path=os.path.dirname(file_path)
    
    os.makedirs(dir_path,exist_ok=True)
    with open(file_path,'wb') as file:
      dill.dump(obj,file)
      
  except Exception as e:
    raise CustomException(e,sys)
  
  
def evaluate_model(X_train,y_train,X_test,y_test,models,param):
  try:
    report={}
    
    for i in range(len(list(models))):
      model=list(models.values())[i]
      para = param[list(models.keys())[i]]
      
      gs=GridSearchCV(model,para,cv=3,n_jobs=-1)
      gs.fit(X_train,y_train)
      
      model.set_params(**gs.best_params_)
      model.fit(X_train,y_train)
      
      #model.fit(X_train,y_train) # Train the model
      y_train_pred=model.predict(X_train)
      y_test_pred=model.predict(X_test)
      
      # Calculate the r2 score
      train_model_score=r2_score(y_train,y_train_pred)
      test_model_score=r2_score(y_test,y_test_pred)
      
      # Calculate the Mean Squared Error
      train_mse=mean_squared_error(y_train,y_train_pred)
      test_mse=mean_squared_error(y_test,y_test_pred)
      
      # Calculate the Mean Absolute Error
      train_mae=mean_absolute_error(y_train,y_train_pred)
      test_mae=mean_absolute_error(y_test,y_test_pred)
      
      
      
      report[list(models.keys())[i]]={
        "Train R2 Score":train_model_score,
        "Test R2 Score":test_model_score,
        "Train Mean Squared Error":train_mse,
        "Test Mean Squared Error":test_mse,
        "Train Mean Absolute Error":train_mae,
        "Test Mean Absolute Error":test_mae
      }
      
    return report
  
  except Exception as e:
    raise CustomException(e,sys)
    