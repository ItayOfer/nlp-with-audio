"""
This script contains utility functions that used in this repo.
"""
import pandas as pd


def file_key_generator(df: pd.DataFrame):
    # Creating file_key which is a unique identifier for each scene.
    df['file_key'] = 'dia' + df['Dialogue_ID'].astype(str) + '_' + 'utt' + df['Utterance_ID'].astype(str)
    return df


def generate_common_ids(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    """
    This function generates a list of common ids between two dataframes.
    """
    common_ids = list(set(df1.index).intersection(set(df2.index)))
    return common_ids