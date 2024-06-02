import pandas as pd

import utils


class FeatureEngineering:
    def __init__(self, path: str, percent_of_speakers: float = 0.8, speaker_combinations=None):
        self.path = path
        self.speaker_combinations = speaker_combinations if speaker_combinations is not None else {}
        self.top_speakers = ['other']
        self.percent_of_speakers = percent_of_speakers

    @staticmethod
    def calculate_utterance_length(data):
        """
        Calculate the length of each utterance in seconds
        """
        data['StartTime'] = pd.to_datetime(data['StartTime'], format='%H:%M:%S,%f')
        data['EndTime'] = pd.to_datetime(data['EndTime'], format='%H:%M:%S,%f')
        data['utterance_spoken_length(secs)'] = (data['EndTime'] - data['StartTime']).dt.total_seconds()
        return data

    @staticmethod
    def calculate_utterance_word_length(data):
        """
        Counts the number of words in each utterance
        """
        data['utterance_length(words)'] = data['Utterance'].apply(lambda x: len(x.split()))
        return data

    @staticmethod
    def dialogue_speakers(data):
        """
        Creates a feature with the unique combination of speakers for each dialogue.
        """
        unique_speakers = data.groupby('Dialogue_ID')['Speaker'].unique().apply(lambda x: sorted(x)).reset_index()
        data = data.merge(unique_speakers, on='Dialogue_ID', suffixes=('', '_unique'))
        return data

    def total_speakers_id(self, data):
        """
        Assigns a unique ID to each unique set of speakers across dialogues.
        If a combination is not found, it assigns 0.
        """
        if not self.speaker_combinations:
            # Create the mapping from the current data if not already provided
            self.speaker_combinations = {tuple(combo): i + 1 for i, combo in
                                         enumerate(data['Speaker_unique'].drop_duplicates().tolist())}
        data['Total_Speakers_id'] = data['Speaker_unique'].apply(lambda x: self.speaker_combinations.get(tuple(x), 0))
        return data

    def find_to_speakers(self, data):
        all_values = pd.Series([item for sublist in data['Speaker_unique'] for item in sublist])
        value_counts = all_values.value_counts()
        total_speakers = len(value_counts)
        threshold = self.percent_of_speakers * total_speakers
        self.top_speakers += value_counts[value_counts >= threshold].index.tolist()

    def encode_speakers_to_binary(self, data):
        def encode_row(row):
            row_set = set(row)
            # Initialize dictionary with zeros for common values and 'other'
            encoded = {val: 0 for val in self.top_speakers}
            encoded['other'] = 0

            # Encode the row
            for item in row_set:
                if item in self.top_speakers:
                    encoded[item] = 1
                else:
                    encoded['other'] = 1

            return pd.Series(encoded)

        binary_encoded_df = data['Speaker_unique'].apply(encode_row)
        return binary_encoded_df

    def run(self, data=None, train=True):
        """
        Runs the entire feature engineering
        """
        if data is None:
            data = pd.read_csv(self.path)
        data = self.calculate_utterance_length(data)
        data = self.calculate_utterance_word_length(data)
        data = self.dialogue_speakers(data)
        data = self.total_speakers_id(data)
        if train:
            self.find_to_speakers(data)
        data[self.top_speakers] = self.encode_speakers_to_binary(data)
        res = data[self.top_speakers + ['Total_Speakers_id', 'utterance_spoken_length(secs)', 'utterance_length(words)']]
        res.columns = [f'fe_features_{col}' for col in res.columns]
        res[['Utterance_ID', 'Dialogue_ID']] = data[['Utterance_ID', 'Dialogue_ID']]
        res = utils.file_key_generator(res)
        res = res.set_index('file_key')
        res = res.drop(columns=['Utterance_ID', 'Dialogue_ID'])
        return res


if __name__ == '__main__':

    MELD_train = FeatureEngineering('MELD.Raw/train/train_sent_emo.csv')
    MELD_train_fe = MELD_train.run()

    MELD_test = FeatureEngineering('MELD.Raw/test_sent_emo.csv',
                                   speaker_combinations=MELD_train.speaker_combinations)
    MELD_test_fe = MELD_test.run()
