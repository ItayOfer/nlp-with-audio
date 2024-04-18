import pandas as pd
from torch.utils.data import DataLoader
from AudioFeaturesExtraction.AudioDataset import AudioDataset
import os
import pickle


class AudioFeaturesExtractor:
    def __init__(self):
        self.train_directory = '../train_audio'
        self.test_directory = '../test_audio'
        self.dev_directory = '../dev_audio'
        self.test_info_path = '../MELD.Raw/test_sent_emo.csv'
        self.train_info_path = '../MELD.Raw/train/train_sent_emo.csv'
        self.dev_info_path = '../MELD.Raw/dev_sent_emo.csv'

    def audio_files_paths_listing(self):
        # Sorting the files by their dia and utt
        sorted_train_files = sorted(os.listdir(self.train_directory))
        sorted_test_files = sorted(os.listdir(self.test_directory))
        sorted_dev_files = sorted(os.listdir(self.dev_directory))

        # Creating 3 lists of the audio files absolute paths
        train_files_paths = [os.path.join(self.train_directory, file) for file in sorted_train_files]
        test_files_paths = [os.path.join(self.test_directory, file) for file in sorted_test_files]
        dev_files_paths = [os.path.join(self.dev_directory, file) for file in sorted_dev_files]

        return train_files_paths, test_files_paths, dev_files_paths

    def file_key_generator(self):
        train_info = pd.read_csv(self.train_info_path)
        test_info = pd.read_csv(self.test_info_path)
        dev_info = pd.read_csv(self.dev_info_path)

        # Creating file_key which is a unique identifier for each scene.
        train_info['file_key'] = 'dia' + train_info['Dialogue_ID'].astype(str) + '_' + 'utt' + train_info[
            'Utterance_ID'].astype(str)
        test_info['file_key'] = 'dia' + test_info['Dialogue_ID'].astype(str) + '_' + 'utt' + test_info[
            'Utterance_ID'].astype(str)
        dev_info['file_key'] = 'dia' + dev_info['Dialogue_ID'].astype(str) + '_' + 'utt' + dev_info[
            'Utterance_ID'].astype(str)

        train_info = train_info.sort_values(by='file_key')
        test_info = test_info.sort_values(by='file_key')
        dev_info = dev_info.sort_values(by='file_key')

        return train_info, test_info, dev_info

    @staticmethod
    def label_generator(train_info, test_info, dev_info):
        labels_dict = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }

        # Encoding labels by labels_dict
        train_info['label'] = train_info['Sentiment'].map(labels_dict)
        test_info['label'] = test_info['Sentiment'].map(labels_dict)
        dev_info['label'] = dev_info['Sentiment'].map(labels_dict)

        return train_info, test_info, dev_info

    @staticmethod
    def creating_is_in_identifier(train_files_paths, test_files_paths, dev_files_paths):
        """This function takes 3 lists of audio paths, and creates is_in datasets.
        is_in datasets contain the file_key(unique identifier for each scene),
        and a column of 1s that indicates whether the observation is in the info and audio datasets."""

        is_in_train = pd.DataFrame({
            'file_key': [os.path.splitext(os.path.basename(file))[0] for file in train_files_paths],
            'is_in': [1 for _ in range(len(train_files_paths))]
        })

        is_in_test = pd.DataFrame({
            'file_key': [os.path.splitext(os.path.basename(file))[0] for file in test_files_paths],
            'is_in': [1 for _ in range(len(test_files_paths))]
        })

        is_in_dev = pd.DataFrame({
            'file_key': [os.path.splitext(os.path.basename(file))[0] for file in dev_files_paths],
            'is_in': [1 for _ in range(len(dev_files_paths))]
        })

        return is_in_train, is_in_test, is_in_dev

    @staticmethod
    def filtering_audio_and_info_data_by_is_in(is_in_train, is_in_test, is_in_dev,
                                               train_files_paths, test_files_paths, dev_files_paths,
                                               train_info, test_info, dev_info):

        # Ensure exact matches are possible
        matching_keys_train = set(is_in_train['file_key']).intersection(set(train_info['file_key']))

        # Proceed with filtering if there are matches
        if len(matching_keys_train) > 0:
            train_files_paths = [file for file in train_files_paths if
                                 os.path.splitext(os.path.basename(file))[0] in matching_keys_train]


        # Repeat for test and dev
        matching_keys_test = set(is_in_test['file_key']).intersection(set(test_info['file_key']))

        if len(matching_keys_test) > 0:
            test_files_paths = [file for file in test_files_paths if
                                os.path.splitext(os.path.basename(file))[0] in matching_keys_test]


        matching_keys_dev = set(is_in_dev['file_key']).intersection(set(dev_info['file_key']))

        if len(matching_keys_dev) > 0:
            dev_files_paths = [file for file in dev_files_paths if
                               os.path.splitext(os.path.basename(file))[0] in matching_keys_dev]

        return train_files_paths, test_files_paths, dev_files_paths, train_info, test_info, dev_info

    @staticmethod
    def creating_data_loaders(train_files_paths, test_files_paths, dev_files_paths,
                              train_info, test_info, dev_info):
        """This function takes the paths lists and the info datasets and returns a dataloader for each audio file"""

        train_dataset = AudioDataset(train_files_paths, train_info['label'].values.tolist())
        test_dataset = AudioDataset(test_files_paths, test_info['label'].values.tolist())
        devel_dataset = AudioDataset(dev_files_paths, dev_info['label'].values.tolist())

        print(f"Train Dataset Length: {len(train_dataset)}")  # Debug print
        print(f"Test Dataset Length: {len(test_dataset)}")  # Debug print
        print(f"Dev Dataset Length: {len(devel_dataset)}")  # Debug print

        train_data_loader = DataLoader(train_dataset, batch_size=1)
        test_data_loader = DataLoader(test_dataset, batch_size=1)
        dev_data_loader = DataLoader(devel_dataset, batch_size=1)

        return train_data_loader, test_data_loader, dev_data_loader

    @staticmethod
    def creating_audio_waveforms(data_loader: DataLoader):
        """
        get a data loader and return the waveforms and labels in a dictionary
        :param data_loader: DataLoader object
        :return: dict with waveforms and labels
        """
        res = {}
        for audio_path_name, waveform, label in data_loader:
            res[audio_path_name[0]] = {
                'waveforms': waveform[0],
                'label': label[0]
            }
        return res

    def run(self):
        train_files_paths, test_files_paths, dev_files_paths = self.audio_files_paths_listing()
        train_info, test_info, dev_info = self.file_key_generator()
        train_info, test_info, dev_info = self.label_generator(train_info, test_info, dev_info)
        is_in_train, is_in_test, is_in_dev = self.creating_is_in_identifier(train_files_paths,
                                                                            test_files_paths,
                                                                            dev_files_paths)
        (train_files_paths, test_files_paths, dev_files_paths,
         train_info, test_info, dev_info) = self.filtering_audio_and_info_data_by_is_in(
            is_in_train, is_in_test, is_in_dev,
            train_files_paths, test_files_paths, dev_files_paths,
            train_info, test_info, dev_info)
        train_data_loader, test_data_loader, dev_data_loader = self.creating_data_loaders(
            train_files_paths, test_files_paths,
            dev_files_paths, train_info, test_info,
            dev_info)
        train = self.creating_audio_waveforms(train_data_loader)
        test = self.creating_audio_waveforms(test_data_loader)
        dev = self.creating_audio_waveforms(dev_data_loader)

        return train, test, dev


if __name__ == '__main__':
    Audio_Features_Extractor = AudioFeaturesExtractor()
    train, test, dev = Audio_Features_Extractor.run()

    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open('dev_data.pkl', 'wb') as f:
        pickle.dump(dev, f)

