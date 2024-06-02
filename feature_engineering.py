import pandas as pd


class FeatureEngineering:
    def __init__(self, path: str, speaker_combinations=None):
        self.path = path
        self.speaker_combinations = speaker_combinations if speaker_combinations is not None else {}

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

    def run(self):
        """
        Runs the entire feature engineering
        """
        data = pd.read_csv(self.path)
        data = self.calculate_utterance_length(data)
        data = self.calculate_utterance_word_length(data)
        data = self.dialogue_speakers(data)
        data = self.total_speakers_id(data)
        return data

if __name__ == '__main__':

    MELD_train = FeatureEngineering('MELD.Raw/train/train_sent_emo.csv')
    MELD_train_fe = MELD_train.run()

    MELD_test = FeatureEngineering('MELD.Raw/test_sent_emo.csv',
                                   speaker_combinations=MELD_train.speaker_combinations)
    MELD_test_fe = MELD_test.run()
