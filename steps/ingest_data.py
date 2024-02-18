import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Read data from the data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: directory path to the data.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Reading the data from the data path.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingest the data from the data_path

    Arguments:
        data_path: path to the data
    Returns:
        pd.DataFrame: ingested data in pandas dataframe

    """

    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error("Error while reading data: {e}")
        raise e
