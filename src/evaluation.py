import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, brier_score_loss, confusion_matrix

class Evaluation(ABC):
    """
    Abstract class for model evaluation strategy.
    """
    @abstractmethod
    def calculate_scores(self,y_true: np.ndarray,y_pred: np.ndarray):
        """
        Calculate model scores
        Arguments:
            y_true: true labels ; numpy array
            y_pred: Predicted labels ; numpy array
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Mean Square Error evaluation strategy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE.")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in MSE calculation {}".format(e))
            raise e
        
class R2(Evaluation):
    """
    R2 Score evaluation strategy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score.")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in R2 Score calculation {}".format(e))
            raise e
        
class Brier(Evaluation):
    """
    Brier Score evaluation strategy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Brier Score.")
            brierscore = brier_score_loss(y_true, y_pred)
            logging.info("Brier Score: {}".format(brierscore))
            return brierscore
        except Exception as e:
            logging.error("Error in Brier Score calculation {}".format(e))
            raise e

class RMSE(Evaluation):
    """
    Root Mean Square Error evaluation strategy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE.")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in MSE calculation {}".format(e))
            raise e
        
class ConfusionMat(Evaluation):
    """
    Confusion Matrix evaluation strategy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Confusion Matrix.")
            cm = confusion_matrix(y_true, y_pred)
            logging.info("Confusion Matrix: {}".format(cm))
            return cm
        except Exception as e:
            logging.error("Error in Confusion Matrix calculation {}".format(e))
            raise e