import logging
import pandas as pd
import numpy as np
import mlflow
from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin
from sklearn.linear_model import LogisticRegression 
from src.evaluation import Brier, ConfusionMat
from typing import Tuple
from typing_extensions import Annotated

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame 
)-> Annotated[float, "brier"]:
    """
    Evaluate model performance.

    Arguments:
        df: pandas dataframe with results from model training
    """
    try:
        prediction = pd.DataFrame(model.predict_proba(X_test), columns=['run','pass'])[['pass']]
        brier_class = Brier()
        brier = brier_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("brier", brier)
        return brier
    except Exception as e:
        logging.error("Error in model evaluation {}".format(e))
        raise e

