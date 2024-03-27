"""
This script contains utility functions that used in this repo.
"""
import pandas as pd


def match_index_to_file_name(file_name: str, df: pd.DataFrame) -> int:
    """
    This function will match the file name to the corresponding index in the data frame.
    :param file_name: The file name.
    :param df: The data frame.
    :return: The index of the file name.
    """
    # TODO or: add this method in the future.
