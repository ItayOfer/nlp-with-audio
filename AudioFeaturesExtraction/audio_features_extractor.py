import pandas as pd
from torch.utils.data import DataLoader
from AudioFeaturesExtraction.audio_dataset import AudioDataset
import os
import pickle


class AudioFeaturesExtractor:
    def __init__(self, path: str, info_file_name: str):
        self.path = path
        self.info_file_name = info_file_name
        self.labels_dict = {
            'negative': 0,
            'neutral': 1,
            'positive': 2}

    def audio_files_paths_listing(self) -> list:
        """

        :return:
        """
        # Sorting the files by their dia and utt
        sorted_files = sorted(os.listdir(self.path))
        # Creating list of the audio files absolute paths
        files_paths = [os.path.join(self.path, file) for file in sorted_files]
        return files_paths

    def file_key_generator(self):
        info_file = pd.read_csv(self.info_file_name)
        # Creating file_key which is a unique identifier for each scene.
        info_file['file_key'] = 'dia' + info_file['Dialogue_ID'].astype(str) + '_' + 'utt' + info_file[
            'Utterance_ID'].astype(str)

        info_file['label'] = info_file['Sentiment'].map(self.labels_dict)
        info_file = info_file.sort_values(by='file_key')
        return info_file

    @staticmethod
    def creating_is_in_identifier(files_paths: list):
        """
        This function takes 3 lists of audio paths, and creates is_in datasets.
        is_in datasets contain the file_key(unique identifier for each scene),
        and a column of 1s that indicates whether the observation is in the info and audio datasets.
        """

        is_in_data = pd.DataFrame({
            'file_key': [os.path.splitext(os.path.basename(file))[0] for file in files_paths],
            'is_in': [1 for _ in range(len(files_paths))]
        })
        return is_in_data

    @staticmethod
    def filtering_audio_and_info_data_by_is_in(is_in_data, files_paths, info_file):
        """

        :param is_in_data:
        :param files_paths:
        :param info_file:
        :return:
        """
        # Ensure exact matches are possible
        matching_keys = set(is_in_data['file_key']).intersection(set(info_file['file_key']))

        # Proceed with filtering if there are matches
        if len(matching_keys) > 0:
            files_paths = [file for file in files_paths if
                           os.path.splitext(os.path.basename(file))[0] in matching_keys]
        return files_paths

    @staticmethod
    def creating_data_loaders(files_paths: list, info_file: pd.DataFrame) -> DataLoader:
        """This function takes the paths lists and the info datasets and returns a dataloader for each audio file"""
        audio_dataset = AudioDataset(files_paths, info_file['label'].values.tolist())
        data_loader = DataLoader(audio_dataset, batch_size=1)
        return data_loader

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
        files_paths = self.audio_files_paths_listing()
        info_file = self.file_key_generator()
        is_in_path = self.creating_is_in_identifier(files_paths)
        files_paths = self.filtering_audio_and_info_data_by_is_in(is_in_path, files_paths, info_file)
        data_loader = self.creating_data_loaders(files_paths, info_file)
        res_dict = self.creating_audio_waveforms(data_loader)
        return res_dict
