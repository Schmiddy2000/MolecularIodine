# Imports
import numpy as np
import pandas as pd

from typing import Tuple


def extract_data(file_name: str) -> Tuple[np.array, np.array]:
    dataframe = pd.read_csv(file_name, delimiter='\t', header=16, encoding='latin1')
    dataframe = dataframe.iloc[:-1, :]
    dataframe = dataframe.reset_index()
    dataframe.rename(columns={dataframe.columns[0]: 'channel', dataframe.columns[1]: 'counts'}, inplace=True)
    dataframe.channel = dataframe.channel.apply(lambda x: int(x.split(',')[0]))
    dataframe.counts = dataframe.counts.apply(lambda x: float(x.replace(',', '.')))

    return dataframe.channel.to_numpy(), dataframe.counts.to_numpy()
