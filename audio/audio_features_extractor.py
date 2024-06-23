import pandas as pd
from torch.utils.data import DataLoader

import utils
from audio.audio_dataset import AudioDataset
import os


class AudioFeaturesExtractor:
    """
    This class is responsible for dealing with the data loaders and returning audio wave forms.
    """

    def __init__(self, path: str, info_file_name: str):
        self.path = path
        self.info_file_name = info_file_name

    def audio_files_paths_listing(self) -> list:
        """
        This method is responsible for collecting the file paths of the audio data.
        """
        # Sorting the files by their dia and utt
        sorted_files = sorted(os.listdir(self.path))
        # Creating list of the audio files absolute paths
        files_paths = [os.path.join(self.path, file) for file in sorted_files]
        return files_paths


    def file_key_generator(self, df: pd.DataFrame):
        """
        This method is responsible for creating an information data frame which consists the file keys (paths) of audio.
        """
        # Creating file_key which is a unique identifier for each scene.
        df = utils.file_key_generator(df)
        info_file = df.sort_values(by='file_key')
        info_file = utils.set_target(info_file)
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
        This method is responsible for matching the audio data to the raw data via file_key.
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
        """
        This function takes the paths lists and the info datasets and returns a dataloader for each audio file
        """
        audio_dataset = AudioDataset(files_paths, info_file['label'].values.tolist())
        data_loader = DataLoader(audio_dataset, batch_size=1)
        return data_loader

    @staticmethod
    def creating_audio_waveforms(data_loader: DataLoader):
        """
        This method is responsible getting a data loader and return the waveforms and labels in a dictionary
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
        info_file = self.file_key_generator(df=pd.read_csv(self.info_file_name))
        is_in_path = self.creating_is_in_identifier(files_paths)
        files_paths = self.filtering_audio_and_info_data_by_is_in(is_in_path, files_paths, info_file)
        data_loader = self.creating_data_loaders(files_paths, info_file)
        res_dict = self.creating_audio_waveforms(data_loader)
        return res_dict
