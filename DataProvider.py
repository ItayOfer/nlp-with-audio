import os
import cv2

class DataProvider:
    def __init__(self):
        self.video_dev_path = 'MELD.Raw/dev_splits_complete'
        self.video_test_path = 'MELD.Raw/output_repeated_splits_test'
        self.video_train_path = 'MELD.Raw/train/train_splits'
        self.audio_dev_path = 'MELD.Raw/dev_audio'
        self.audio_test_path = 'MELD.Raw/test_audio'
        self.audio_train_path = 'MELD.Raw/train_audio'

    def run(self, path: str):
        """
        this method will run the data provider on the given path and do the following:
        1 run on data paths
        2 read the video data
        3 extract audio from video
        4 save the audio
        :param path: path to run the data provider on
        :return:
        """
        path_dict = {'dev': self.video_dev_path,
                     'test': self.video_test_path,
                     'train': self.video_train_path}

        path_to_run = path_dict[path]
        for file_name in os.listdir(path_to_run):
            check_files = file_name.split('.')
            if check_files[1] == 'mp4':
                video = self.get_video_data(file_name)
                audio = self.extract_audio_from_video(video)
                self.save_audio(audio, check_files[0])

    def get_video_data(self, file_name: str) -> object:
        ...

    def extract_audio_from_video(self, video: object) -> object:
        ...

    def save_audio(self, audio, file_name: str) -> None:
        pass


if __name__ == '__main__':
    data_provider = DataProvider()
    data_provider.run('dev')
    data_provider.run('test')
    data_provider.run('train')