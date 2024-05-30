import pandas as pd

from text.TextModel import TextModel


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
    return text_train, text_dev, text_test
def concat_text_audio():
    ...