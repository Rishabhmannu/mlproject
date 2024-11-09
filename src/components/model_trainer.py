import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
  
  def initiate_model_trainer(self,train_array,test_array):
    try:
      logging.info("Splitting training and testing data")
      X_train,y_train,X_test,y_test=(
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]
      )
      
      models={
        "Random Forest" : RandomForestRegressor(),
        "Decision Tree" : DecisionTreeRegressor(),
        "Gradient Boosting" : GradientBoostingRegressor(),
        "Linear Regression" : LinearRegression(),
        "Ada Boost Regressor" : AdaBoostRegressor(),
        "K-Neighbours Regressor" : KNeighborsRegressor(),
        "Cat Boost Regressor" : CatBoostRegressor(),
        "XGBoost Regressor" : XGBRegressor()
      }
      
      model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test,models=models)
      
      
      # Get the best model score and name
      best_model_name=max(model_report,key=lambda x:model_report[x]['Test R2 Score'])
      best_model_score=model_report[best_model_name]['Test R2 Score']
      
      if(best_model_score<0.6):
        raise CustomException("No best model found")
      
      logging.info(f"Best model found on both training and testing data")
      logging.info(f"Best model name : {best_model_name}")
      logging.info(f"Best model score : {best_model_score}")
      
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=models[best_model_name]
      )
      

      predicted = models[best_model_name].predict(X_test)
      best_model_r2_score = r2_score(y_test, predicted)

      
      return best_model_r2_score
      
    except Exception as e:
      raise CustomException(e,sys)