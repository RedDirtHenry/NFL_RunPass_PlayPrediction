import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivisionStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Cleans data and divides into train and test sets.

    Arguments:
        df: Raw data input
    Returns:
        X_train: training data ; pd.DataFrame
        X_test: testing data ; pd.Dataframe
        y_train: training labels ; pd.Series
        y_test: testing labels ; pd.Series

    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning =  DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()    

        divide_strategy = DataDivisionStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed.")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e
