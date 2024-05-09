import pandas as pd
from stop_words import get_stop_words


import re


class TextCleaner:
    def __init__(self, language='en'):
        # Get a list of English stop words
        self.stop_words = get_stop_words(language)

    def _clean(self, text: str) -> str | None:
        # Convert text to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[\W_]+', ' ', text)

        # Optionally, remove stop words
        text_tokens = text.split()
        filtered_text = ' '.join([word for word in text_tokens if word not in self.stop_words and len(word) > 1])
        if len(filtered_text) == 0:
            return None
        return filtered_text

    def apply_cleaner(self, text_series: pd.Series) -> pd.Series:
        return text_series.apply(self._clean)


class TextModel:
    ...


class TextModelRunner:
    ...


if __name__ == '__main__':
    df_train = pd.read_csv('MELD.Raw/train/train_sent_emo.csv')
    df_dev = pd.read_csv('MELD.Raw/dev_sent_emo.csv')
    df_test = pd.read_csv('MELD.Raw/test_sent_emo.csv')
    text_cleaner = TextCleaner()
    df_train['Utterance'] = text_cleaner.apply_cleaner(df_train['Utterance'])
    df_dev['Utterance'] = text_cleaner.apply_cleaner(df_dev['Utterance'])
    df_test['Utterance'] = text_cleaner.apply_cleaner(df_test['Utterance'])

