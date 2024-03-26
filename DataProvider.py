import os
from moviepy.editor import VideoFileClip


class DataProvider:
    def __init__(self):
        self.video_dev_path = 'MELD.Raw/dev_splits_complete'
        self.video_test_path = 'MELD.Raw/output_repeated_splits_test'
        self.video_train_path = 'MELD.Raw/train/train_splits'
        self.audio_dev_path = 'dev_audio'
        self.audio_test_path = 'test_audio'
        self.audio_train_path = 'train_audio'
        self.video_path_dict = {'dev': self.video_dev_path,
                                'test': self.video_test_path,
                                'train': self.video_train_path}
        self.audio_path_dict = {'dev': self.audio_dev_path,
                                'test': self.audio_test_path,
                                'train': self.audio_train_path}

    def run(self):
        """
        this method will run the data provider on the given path and do the following:
        1 run on data paths
        2 read the video data
        3 extract audio from video
        4 save the audio
        :return:
        """
        for path in ['dev', 'test', 'train']:
            video_path = self.video_path_dict[path]
            audio_path = self.audio_path_dict[path]
            os.makedirs(audio_path, exist_ok=True)
            for file_name in os.listdir(video_path):
                check_files = file_name.split('.')
                if check_files[1] == 'mp4':
                    video_file_path = os.path.join(video_path, file_name)
                    audio_file_path = os.path.join(audio_path, check_files[0] + '.mp3')
                    DataProvider.extract_audio_from_video(video_file_path, audio_file_path)

    @staticmethod
    def extract_audio_from_video(video_file_path: str, audio_file_path: str) -> None:
        """
        Extracts audio from the video file and saves it to a specified path.
        :param video_file_path: Path to the video file.
        :param audio_file_path: Path where the extracted audio will be saved.
        """
        video = VideoFileClip(video_file_path)
        video.audio.write_audiofile(audio_file_path, codec='mp3')


if __name__ == '__main__':
    data_provider = DataProvider()
    data_provider.run()
