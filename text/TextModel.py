from typing import List

from sklearn.metrics import accuracy_score
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from stop_words import get_stop_words

import nltk

import utils

nltk.download('punkt')
import gensim.downloader as api
import re
import xgboost as xgb
import statsmodels.api as sm


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


def clean_stop_words_and_special_characters_and_set_target(df: pd.DataFrame):
    text_cleaner = TextCleaner()
    df['tokens'] = text_cleaner.apply_cleaner(df['Utterance'])
    df['labels'] = df['Sentiment'].replace({'negative': 0, 'neutral': 1, 'positive': 0})
    df = df.dropna(subset=['tokens'])
    return df


class TextModel:
    def __init__(self):
        self.glove_model = api.load("glove-wiki-gigaword-100")

    def preprocessing(self, df):
        df = clean_stop_words_and_special_characters_and_set_target(df)
        df = utils.file_key_generator(df)
        df = df.set_index('file_key')
        sentence_vectors = sentence_to_vec(df, self.glove_model)
        sentence_vectors.columns = [f'text_feautre_{i + 1}' for i in range(len(sentence_vectors.columns))]
        return sentence_vectors, df['labels']


if __name__ == '__main__':
    audio_train, audio_dev, audio_test = utils.get_audio_data()
    text_train, text_dev, text_test, y_train, y_dev, y_test = utils.get_text_data_and_labels()
    train_data = utils.concat_text_audio(audio_train, text_train, y_train)
    dev_data = utils.concat_text_audio(audio_dev, text_dev, y_dev)
    test_data = utils.concat_text_audio(audio_test, text_test, y_test)
    train_data = pd.concat([train_data, dev_data], axis=0)
    y_train = pd.concat([y_train, y_dev], axis=0)

    for feature in features:
        # Prepare the feature data with an intercept
        X = sm.add_constant(X_train_only_audio[feature])  # Adds a constant term to the feature
        y = y_train

        # Fit logistic regression model
        model = sm.Logit(y_train, X).fit(disp=0)  # disp=0 turns off the fitting summary output

        # Store the log-likelihood
        log_likelihoods.append(model.llf)  # llf is the log likelihood of the fitted model

    # Create a DataFrame to sort features by log-likelihood
    results = pd.DataFrame({
        'Feature': features,
        'Log-Likelihood': log_likelihoods
    }).sort_values(by='Log-Likelihood', ascending=True)

    # import matplotlib.pyplot as plt
    #
    # # Assuming 'results' DataFrame from your code is already prepared and sorted
    #
    # # Plotting
    # plt.figure(figsize=(10, 8))
    # # Scatter plot where we use the index as the y-value and log-likelihood as the x-value
    # plt.scatter(results['Log-Likelihood'], range(len(results['Feature'])), color='b')
    #
    # # Setting the y-ticks to show feature names
    # plt.yticks(range(len(results['Feature'])), results['Feature'])
    #
    # plt.title('Log-Likelihood of Logistic Regression Models by Feature')
    # plt.xlabel('Log-Likelihood')
    # plt.ylabel('Feature')
    # plt.show()

    # Select the best audio features
    best_audio_features = results.tail(15)['Feature'].values.tolist()
    X_train_only_audio = X_train_only_audio[best_audio_features]
    X_test_only_audio = X_test_only_audio[best_audio_features]

    # Combine text and audio features
    X_train_text_and_audio = pd.concat([X_train_only_audio, X_train_only_text], axis=1)
    X_test_text_and_audio = pd.concat([X_test_only_audio, X_test_only_text], axis=1)

    # Set up parameters for xgboost
    param_grid = {
        'max_depth': [3, 6, 9],
        'eta': [0.1, 0.3, 0.5],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'objective': ['binary:logistic']  # set the objective for binary classification
    }

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid_search_text = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
    grid_search_audio = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
    grid_search_audio_text = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3,
                                          verbose=1)

    grid_search_text.fit(X_train_only_text, y_train)
    grid_search_audio.fit(X_train_only_audio, y_train)
    grid_search_audio_text.fit(X_train_text_and_audio, y_train)

    best_xgb_model_text = grid_search_text.best_estimator_
    best_xgb_model_audio = grid_search_audio.best_estimator_
    best_xgb_model_audio_text = grid_search_audio_text.best_estimator_

    x_preds_audio_text = best_xgb_model_audio_text.predict(X_test_text_and_audio)
    x_preds_text = best_xgb_model_text.predict(X_test_only_text)
    x_preds_audio = best_xgb_model_audio.predict(X_test_only_audio)

    # Evaluate predictions
    accuracy_audio = accuracy_score(y_test, x_preds_audio)
    print("Accuracy only audio features: %.2f%%" % (accuracy_audio * 100.0))

    accuracy_text = accuracy_score(y_test, x_preds_text)
    print("Accuracy only text features: %.2f%%" % (accuracy_text * 100.0))

    accuracy_text_audio = accuracy_score(y_test, x_preds_audio_text)
    print("Accuracy text and audio features: %.2f%%" % (accuracy_text_audio * 100.0))

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    def plot_feature_distribution(train_df, test_df):
        features = train_df.columns

        for feature in features:
            plt.figure(figsize=(12, 6))
            sns.kdeplot(train_df[feature], label='Train', fill=True, color='blue', alpha=0.5)
            sns.kdeplot(test_df[feature], label='Test', fill=True, color='red', alpha=0.5)

            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
            plt.show()


    plot_feature_distribution(X_train_text_and_audio, X_test_text_and_audio)
