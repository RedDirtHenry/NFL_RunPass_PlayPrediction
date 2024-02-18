import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data = data[["play_id",	
                         "game_id",
                         "week",
                         "posteam",
                         "yardline_100",
                         "half_seconds_remaining",
                         "game_seconds_remaining",
                         "down",
                         "ydstogo",
                         "play_type",
                         "no_huddle",
                         "posteam_timeouts_remaining",
                         "score_differential",
                         "wp",
                         "pass",
                         "rush"]]
            data = data[((data['pass']==1) | (data['rush']==1)) & (data['play_type']!='no_play')]
            data = data.drop(["rush","play_type"], axis=1)
            data = data.dropna()
            data['down'] = data['down'].astype('category')
            data["posteam_timeouts_remaining"] = data["posteam_timeouts_remaining"].astype('category')
            return data
        except Exception as e:
            logging.error("Error in preprocessing data {}".format(e))
            raise e
        
class DataDivisionStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test sets.
    """
    try:
        def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
            data_no_ids = data.drop(columns = ['game_id', 'play_id','week','posteam'])
            data_no_ids = pd.get_dummies(data_no_ids, columns=['down'])
            X = data_no_ids.drop(["pass"],axis = 1)
            y = data_no_ids["pass"]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=82)
            return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in dividing data: {}".format(e))
        raise e

class DataCleaning:
    """
    Class for data preprocessing and division into training and test sets.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Final handle
        """

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e