import pandas as pd

from text.TextModel import TextModel

audio_features_train = pd.read_csv('audio/train_fe.csv')
text_train = pd.read_csv('MELD.Raw/train/train_sent_emo.csv')
text_model_train = TextModel()
X_train_text, y_train = text_model_train.preprocessing(text_train)

a = 1