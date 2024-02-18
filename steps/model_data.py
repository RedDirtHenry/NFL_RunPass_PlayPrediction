import logging
import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client
from src.model_dev import LogisticRegressionModel
from sklearn.base import RegressorMixin
from sklearn.linear_model import LogisticRegression 
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train : pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
) -> LogisticRegression:
    """
    Training model.
    
    Arguments:
        df: pandas dataframe of dataset to be trained
    """
    model = None
    try:
        if config.model_name == "LogisticRegression":
            mlflow.sklearn.autolog()
            model = LogisticRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported.".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e