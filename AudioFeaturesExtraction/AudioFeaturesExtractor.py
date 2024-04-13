import os
import json
import librosa
import pandas as pd
import numpy as np
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


class AudioFeaturesExtractor:

    def __init__(self):
        self.dev_audio_path = 'dev_audio'
        self.test_audio_path = 'test_audio'
        self.train_audio_path = 'train_audio'
        self.dev_features_dict = {}
        self.test_features_dict = {}
        self.train_features_dict = {}

    @staticmethod
    def audio_feature_extraction(audio_path:str):
        """This function takes the path of an audio file,
        and returns a dictionary with the audio features for the specific file"""

        y, sr = librosa.load(audio_path)
        # Ensure y is at least 1024 samples long
        if len(y) < 1024:
            y = librosa.util.fix_length(y, size=1024)

        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo
        rms_energy = librosa.feature.rms(y=y)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C7'))
        pitch = f0

        features_dict = {
            'mfccs': mfccs,
            'chroma': chroma,
            'spectral_contrast': spectral_contrast,
            'tonnetz': tonnetz,
            'tempo': tempo,
            'rms_energy': rms_energy,
            'zero_crossing_rate': zero_crossing_rate,
            'pitch': pitch
        }

        return features_dict

    def process_audio_files(self, audio_path, target_dict):
        """This function performs audio feature extraction for each file in the directory"""
        allowed_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.aif', '.ogg'}

        for filename in os.listdir(audio_path):
            if os.path.splitext(filename)[1].lower() in allowed_extensions:
                file_path = os.path.join(audio_path, filename)
                if os.path.isfile(file_path):
                    features = self.audio_feature_extraction(file_path)
                    target_dict[filename] = features

    @staticmethod
    def save_features_dict(features_dict, save_path, filename):
        """This function saves a unique dict for each directory"""
        for key in features_dict:
            for feature in features_dict[key]:
                if isinstance(features_dict[key][feature], np.ndarray):
                    features_dict[key][feature] = features_dict[key][feature].tolist()

        full_path = os.path.join(save_path, f"{filename}.json")

        with open(full_path, 'w') as f:
            json.dump(features_dict, f, indent=4)

    def process_all_paths_and_save(self):
        self.process_audio_files(self.dev_audio_path, self.dev_features_dict)
        self.save_features_dict(self.dev_features_dict, self.dev_audio_path, 'dev_features')

        self.process_audio_files(self.test_audio_path, self.test_features_dict)
        self.save_features_dict(self.test_features_dict, self.test_audio_path, 'test_features')

        self.process_audio_files(self.train_audio_path, self.train_features_dict)
        self.save_features_dict(self.train_features_dict, self.train_audio_path, 'train_features')


def main():
    extractor = AudioFeaturesExtractor()
    extractor.process_all_paths_and_save()


if __name__ == '__main__':
    main()
