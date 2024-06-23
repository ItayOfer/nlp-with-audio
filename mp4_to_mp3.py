import os
from moviepy.editor import VideoFileClip
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, filename='data_provider.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


class AudioExtractor:
    def __init__(self):
        """
        This class will extract audio from video files and save them as mp3 files
        """
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

        self.unsuccessful_file_names = {'train': [], 'dev': [], 'test': []}

    def run(self):
        """
        this method will run the data provider on the given path and do the following:
        1 run on data paths
        2 read the video data
        3 extract audio from video
        4 save the audio
        :return:
        """
        for path in ['train', 'dev', 'test']:
            video_path = self.video_path_dict[path]
            audio_path = self.audio_path_dict[path]
            os.makedirs(audio_path, exist_ok=True)
            for file_name in os.listdir(video_path):
                check_files = file_name.split('.')
                if check_files[1] == 'mp4':
                    video_file_path = os.path.join(video_path, file_name)
                    audio_file_path = os.path.join(audio_path, check_files[0] + '.mp3')
                    try:
                        # Extract audio from video
                        video = VideoFileClip(video_file_path)
                        video.audio.write_audiofile(audio_file_path, codec='mp3')
                    except Exception:
                        error_message = f'Error extracting audio from {video_file_path} to {audio_file_path}'
                        logging.error(error_message)
                        self.unsuccessful_file_names[path].append(file_name)
        # Log unsuccessful files
        logging.info(f"unsuccessful to read {len(self.unsuccessful_file_names['train'])} from train")
        logging.info(f"unsuccessful to read {len(self.unsuccessful_file_names['dev'])} from dev")
        logging.info(f"unsuccessful to read {len(self.unsuccessful_file_names['test'])} from test")


if __name__ == '__main__':
    data_provider = AudioExtractor()
    data_provider.run()
