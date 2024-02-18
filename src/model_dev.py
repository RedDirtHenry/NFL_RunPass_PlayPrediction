import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression 

class Model(ABC):
    """
    Abstract class for models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train model.
        Arguments:
            X_train: Training Data ; pandas dataframe
            y_train: Training labels ; pandas series
        Returns:
            None
        """
        pass

class LogisticRegressionModel(Model):
    """
    Logistic Regression model.
    """

    def train(self, X_train, y_train):
        """
        Training Log Reg model.
        Arguments:
            X_train: Training Data ; pandas dataframe
            y_train: Training labels ; pandas series
        Returns:
            Regressed model.
        """
        try:
            LogR = LogisticRegression()
            LogR.fit(X_train, y_train)
            logging.info("Model training finished.")
            return LogR
        except Exception as e:
            logging.error("Error in model training: {}".format(e))
            raise e