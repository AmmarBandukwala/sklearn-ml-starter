import os
import pandas as pd
import pathlib

class StockData:
    """A POCO container for the company stock time series data set with type property attributes.
    """
    Title: str
    Data: pd.DataFrame
    def __init__(self, company_name: str, data: pd.DataFrame) -> None:
        self.Title = company_name
        self.Data = StockData.convert_currency_float(data)
    
    @staticmethod
    def convert_currency_float(df: pd.DataFrame) -> pd.DataFrame:
        """Take in a data frame and looks for the columns (Close, Open, High, Low) convert them from string currency to float number.

        Args:
            df (pd.DataFrame): Pandas data frame with required columns.

        Returns:
            DataFrame: Returns a Pandas dataframe with in-place converted columns.
        """
        df = df.dropna()
        cols = ['Close','Open','High','Low']
        df[cols] = df[cols].replace('[\$,)]', '', regex=True)
        df[cols] = df[cols].astype(float)
        return df
    
def load_data(data_folder_path: str, fileName: str) -> StockData:
    """This function loads the csv files from the data folder and encaptulates them in POCO object.

    Args:
        data_folder_path (str): Path to formatted csv or parquet data set.
        fileName (str): Name of csv or parquet file.

    Returns:
        StockData: Initialize object instance of StockData and its meta data.
    """
    company_name = fileName.split('_')[0]
    file_extension = pathlib.Path(fileName).suffix
    full_path = os.path.join(data_folder_path, fileName)
    if os.path.exists(full_path):
        if file_extension == ".csv":
            df = pd.read_csv(full_path)
            return StockData(company_name, df)
        if file_extension == ".parquet":
            df = pd.read_parquet(full_path)
            return StockData(company_name, df)
    else:
        print('The path provided does not exists on local file system.')
        return None