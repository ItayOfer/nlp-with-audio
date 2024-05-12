import pickle

from sklearn.metrics import accuracy_score, classification_report
import librosa
import numpy as np
import pandas as pd
from stop_words import get_stop_words
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
import gensim.downloader as api
import re
import xgboost as xgb


class TextCleaner:
    def __init__(self, language='en'):
        # Get a list of English stop words
        self.stop_words = get_stop_words(language)

    def _clean(self, text: str) -> str | None:
        # Convert text to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)

        # Optionally, remove stop words
        text_tokens = text.split()
        filtered_text = ' '.join([word for word in text_tokens if word not in self.stop_words and len(word) > 1])
        if len(filtered_text) == 0:
            return None
        return filtered_text

    def apply_cleaner(self, text_series: pd.Series) -> pd.Series:
        return text_series.apply(self._clean)

    def set_target(self, target: pd.Series) -> pd.Series:
        res = target.replace({'negative': 0, 'neutral': 1, 'positive': 2})
        return res


# Tokenization and Vocabulary Building
def build_vocab(data):
    counter = Counter()
    for text in data:
        tokens = text.lower().split()
        counter.update(tokens)
    vocab = {word: i + 1 for i, (word, _) in enumerate(counter.items())}
    vocab['<pad>'] = 0
    return vocab


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.long)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.long)
    return text_list, label_list


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [self.encode(text, vocab) for text in texts]
        self.labels = labels

    def encode(self, text, vocab):
        tokens = word_tokenize(text)
        return [vocab.get(token, 0) for token in tokens]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Multi-Class Feed-Forward Neural Network
class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 50, padding_idx=0)
        self.fc = nn.Linear(50, 3)

    def forward(self, x):
        x = self.embedding(x)
        mask = x.ne(0).float()  # Create a mask for non-zero entries

        # Sum embeddings across the sequence length and divide by the number of non-zero entries per sequence
        x = torch.sum(x, dim=1)  # Sum across sequence length
        mask_sum = mask.sum(dim=1)  # Sum mask across sequence length
        # Avoid division by zero by adding a small epsilon where mask_sum is zero
        mask_sum = mask_sum + (mask_sum == 0).float() * 1e-10
        x = x / mask_sum.unsqueeze(-1)  # Divide sum by non-zero counts, ensure right broadcasting

        x = self.fc(x)
        return x


def sentence_to_vec(sentence_list, embedding_model):
    vec_list = []
    for sentence in sentence_list:
        word_vectors = []
        for word in sentence.split():
            try:
                word_vectors.append(embedding_model[word])
            except KeyError:
                continue  # Skip words not in the vocabulary
        if word_vectors:
            vec_list.append(np.mean(word_vectors, axis=0))
        else:
            vec_list.append(np.zeros(100))  # Assuming 100 dimensional embeddings
    return np.array(vec_list)


def extract_features(waveform, sr):
    """
    Calculate various spectral features and return them in a dictionary.
    """
    # Basic spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=waveform)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
    rms_energy = librosa.feature.rms(y=waveform)[0]
    zcr = librosa.feature.zero_crossing_rate(waveform)[0]
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)

    # Initialize the feature dictionary
    features = {
        'centroid_mean': np.mean(spectral_centroid),
        'bandwidth_mean': np.mean(spectral_bandwidth),
        'flatness_mean': np.mean(spectral_flatness),
        'rolloff_mean': np.mean(spectral_rolloff),
        'rms_energy_mean': np.mean(rms_energy),
        'zcr_mean': np.mean(zcr)
    }

    # Adding MFCCs and Chroma features
    for i in range(mfccs.shape[0]):  # Assuming MFCCs are returned with shape (n_mfcc, t)
        features[f'mfccs_mean_{i}'] = np.mean(mfccs[i, :])

    for i in range(chroma.shape[0]):  # Assuming Chroma features are returned with shape (n_chroma, t)
        features[f'chroma_mean_{i}'] = np.mean(chroma[i, :])

    return features


def sentence_to_vec(df, embedding_model):
    vec_list = []
    for index, row in df.iterrows():
        sentence = row['tokens']  # Directly using the 'tokens' column
        word_vectors = []
        for word in sentence.split():
            try:
                word_vectors.append(embedding_model[word])
            except KeyError:
                continue  # Skip words not in the vocabulary
        if word_vectors:
            vec_list.append(np.mean(word_vectors, axis=0))
        else:
            vec_list.append(np.zeros(100))  # Assuming 100 dimensional embeddings

    # Create a DataFrame from the list of vectors and preserve the original index
    vec_df = pd.DataFrame(vec_list, index=df.index)
    return vec_df


if __name__ == '__main__':
    df_train = pd.read_csv('MELD.Raw/train/train_sent_emo.csv')
    df_dev = pd.read_csv('MELD.Raw/dev_sent_emo.csv')
    df_test = pd.read_csv('MELD.Raw/test_sent_emo.csv')
    text_cleaner = TextCleaner()
    # clean stop words and special characters
    df_train['tokens'] = text_cleaner.apply_cleaner(df_train['Utterance'])
    df_dev['tokens'] = text_cleaner.apply_cleaner(df_dev['Utterance'])
    df_test['tokens'] = text_cleaner.apply_cleaner(df_test['Utterance'])
    df_train = df_train.dropna(subset=['tokens'])
    df_dev = df_dev.dropna(subset=['tokens'])
    df_test = df_test.dropna(subset=['tokens'])
    # set key
    df_train['scene_id'] = 'dia' + df_train['Dialogue_ID'].astype(str) + '_' + 'utt' + df_train['Utterance_ID'].astype(
        str)
    df_test['scene_id'] = 'dia' + df_test['Dialogue_ID'].astype(str) + '_' + 'utt' + df_test['Utterance_ID'].astype(str)
    df_train = df_train.set_index('scene_id')
    df_test = df_test.set_index('scene_id')

    # set target
    df_train['labels'] = text_cleaner.set_target(df_train['Sentiment'])
    df_dev['labels'] = text_cleaner.set_target(df_dev['Sentiment'])
    df_test['labels'] = text_cleaner.set_target(df_test['Sentiment'])

    glove_model = api.load("glove-wiki-gigaword-100")
    sentence_vectors_test = sentence_to_vec(df_test, glove_model)
    sentence_vectors_train = sentence_to_vec(df_train, glove_model)

    ## ---- make audio fearues ----
    with open('AudioFeaturesExtraction/train_data.pkl', 'rb') as file:
        train_audio_dict = pickle.load(file)

    with open('AudioFeaturesExtraction/test_data.pkl', 'rb') as file:
        test_audio_dict = pickle.load(file)

    # sample_rate = 44100
    # audio_feature_train = process_audio_data(train_audio_dict, sample_rate)
    # audio_feature_test = process_audio_data(test_audio_dict, sample_rate)
    audio_feature_train = pd.read_csv('audio_features_train.csv')
    audio_feature_test = pd.read_csv('audio_features_test.csv')
    audio_feature_train = audio_feature_train.set_index('scene_id')
    audio_feature_test = audio_feature_test.set_index('scene_id')

    common_ids_train = list(set(audio_feature_train.index).intersection(set(df_train.index)))
    common_ids_test = list(set(audio_feature_test.index).intersection(set(df_test.index)))
    y_train = df_train.loc[common_ids_train, 'labels']
    y_test = df_test.loc[common_ids_test, 'labels']

    # set 3 dfs

    X_train_only_audio = audio_feature_train.loc[common_ids_train]
    X_test_only_audio = audio_feature_test.loc[common_ids_test]
    X_train_only_audio = X_train_only_audio.drop(columns='Unnamed: 0')
    X_test_only_audio = X_test_only_audio.drop(columns='Unnamed: 0')

    X_train_only_text = sentence_vectors_train.loc[common_ids_train]
    X_test_only_text = sentence_vectors_test.loc[common_ids_test]

    X_train_text_and_audio = pd.concat([X_train_only_audio, X_train_only_text], axis=1)
    X_test_text_and_audio = pd.concat([X_test_only_audio, X_test_only_text], axis=1)

    dtrain_audio = xgb.DMatrix(X_train_only_audio, label=y_train)
    dtest_audio = xgb.DMatrix(X_test_only_audio, label=y_test)

    dtrain_text = xgb.DMatrix(X_train_only_text, label=y_train)
    dtest_text = xgb.DMatrix(X_test_only_text, label=y_test)

    dtrain_text_audio = xgb.DMatrix(X_train_text_and_audio, label=y_train)
    dtest_text_audio = xgb.DMatrix(X_test_text_and_audio, label=y_test)

    # Set up parameters for xgboost
    params = {
        'max_depth': 6,  # you can tune this and other parameters
        'eta': 0.3,  # learning rate
        'objective': 'multi:softprob',  # multi-class classification
        'num_class': 3  # number of classes
    }

    num_boost_round = 100
    bst_audion = xgb.train(params, dtrain_audio, num_boost_round)
    preds = bst_audion.predict(dtest_audio)
    predictions_audio = np.asarray([np.argmax(line) for line in preds])

    bst_text = xgb.train(params, dtrain_text, num_boost_round)
    preds = bst_text.predict(dtest_text)
    predictions_text = np.asarray([np.argmax(line) for line in preds])

    bst_text_audio = xgb.train(params, dtrain_text_audio, num_boost_round)
    preds = bst_text_audio.predict(dtest_text_audio)
    predictions_text_audio = np.asarray([np.argmax(line) for line in preds])

    # Evaluate predictions
    accuracy_audio = accuracy_score(y_test, predictions_audio)
    print("Accuracy only audio features: %.2f%%" % (accuracy_audio * 100.0))

    accuracy_text = accuracy_score(y_test, predictions_text)
    print("Accuracy only text features: %.2f%%" % (accuracy_text * 100.0))

    accuracy_text_audio = accuracy_score(y_test, predictions_text_audio)
    print("Accuracy text and audio features: %.2f%%" % (accuracy_text_audio * 100.0))
