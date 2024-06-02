"""
This script contains utility functions that used in this repo.
"""
import pandas as pd
from text.TextModel import TextModel


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


def get_audio_data():
    audio_train = pd.read_csv('audio/train_fe.csv')
    audio_train = audio_train.set_index('file_key')
    audio_dev = pd.read_csv('audio/dev_fe.csv')
    audio_dev = audio_dev.set_index('file_key')
    audio_test = pd.read_csv('audio/test_fe.csv')
    audio_test = audio_test.set_index('file_key')
    return audio_train, audio_dev, audio_test


def get_text_data_and_labels():
    text_model = TextModel()
    text_train = pd.read_csv('MELD.Raw/train/train_sent_emo.csv')
    text_dev = pd.read_csv('MELD.Raw/dev_sent_emo.csv')
    text_test = pd.read_csv('MELD.Raw/test_sent_emo.csv')
    X_train_text, y_train = text_model.preprocessing(text_train)
    X_dev_text, y_dev = text_model.preprocessing(text_dev)
    X_test_text, y_test = text_model.preprocessing(text_test)
    return X_train_text, X_dev_text, X_test_text, y_train, y_dev, y_test


def concat_text_audio(audio: pd.DataFrame, text: pd.DataFrame, y: pd.Series):
    return pd.concat([audio, text, y], axis=1)

def get_data():
    audio_train, audio_dev, audio_test = get_audio_data()
    text_train, text_dev, text_test, y_train, y_dev, y_test = get_text_data_and_labels()
    train_data = concat_text_audio(audio_train, text_train, y_train)
    dev_data = concat_text_audio(audio_dev, text_dev, y_dev)
    test_data = concat_text_audio(audio_test, text_test, y_test)
    train_data = pd.concat([train_data, dev_data], axis=0)
    y_train = pd.concat([y_train, y_dev], axis=0)
    return train_data, test_data, y_train, y_test
