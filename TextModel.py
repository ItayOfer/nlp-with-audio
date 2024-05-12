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
    # set target
    df_train['labels'] = text_cleaner.set_target(df_train['Sentiment'])
    df_dev['labels'] = text_cleaner.set_target(df_dev['Sentiment'])
    df_test['labels'] = text_cleaner.set_target(df_test['Sentiment'])

    glove_model = api.load("glove-wiki-gigaword-100")
    text_test = df_test['tokens'].values.tolist()
    sentence_vectors_test = sentence_to_vec(text_test, glove_model)
    text_train = df_train['tokens'].values.tolist()
    sentence_vectors_train = sentence_to_vec(text_train, glove_model)
    X_train = pd.DataFrame(sentence_vectors_train)
    X_test = pd.DataFrame(sentence_vectors_test)

    y_train = df_train['labels']
    y_test = df_test['labels']

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set up parameters for xgboost
    params = {
        'max_depth': 6,  # you can tune this and other parameters
        'eta': 0.3,  # learning rate
        'objective': 'multi:softprob',  # multi-class classification
        'num_class': 3  # number of classes
    }

    num_boost_round = 100
    bst = xgb.train(params, dtrain, num_boost_round)
    preds = bst.predict(dtest)
    predictions = np.asarray([np.argmax(line) for line in preds])

    # Evaluate predictions
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy without audio features: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y_test, predictions))