import pandas as pd
import pickle
import numpy as np
import librosa


class AudioWaveformsConverter:
    def __init__(self, path, sample_rate):
        self.path = path
        self.sample_rate = sample_rate

    def data_loader(self):
        with open(self.path, 'rb') as file:
            audio_dict = pickle.load(file)
        return audio_dict

    def extract_features(self,waveform):
        """
        Calculate various spectral, rhythmic, and tonal features and return them in a dictionary.
        """
        # Decompose into harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(waveform)

        # Temporal and spectral features
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=self.sample_rate)
        onset_env = librosa.onset.onset_strength(y=waveform, sr=self.sample_rate)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sample_rate)
        fourier_tempogram = librosa.feature.fourier_tempogram(onset_envelope=onset_env, sr=self.sample_rate)
        tempogram_ratio = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sample_rate, win_length=16)
        p_features = librosa.feature.poly_features(S=librosa.stft(waveform), sr=self.sample_rate, order=2)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=self.sample_rate)

        # Basic spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=self.sample_rate)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=waveform)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=self.sample_rate)[0]
        rms_energy = librosa.feature.rms(y=waveform)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        mfccs = librosa.feature.mfcc(y=waveform, sr=self.sample_rate)
        chroma = librosa.feature.chroma_stft(y=waveform, sr=self.sample_rate)

        # Initialize the feature dictionary
        features = {
            'centroid_mean': np.mean(spectral_centroid),
            'bandwidth_mean': np.mean(spectral_bandwidth),
            'flatness_mean': np.mean(spectral_flatness),
            'rolloff_mean': np.mean(spectral_rolloff),
            'rms_energy_mean': np.mean(rms_energy),
            'zcr_mean': np.mean(zcr),
            'tempo': tempo,
            'tempogram_mean': np.mean(tempogram),
            'fourier_tempogram_mean': np.mean(fourier_tempogram),
            'tempogram_ratio_mean': np.mean(tempogram_ratio),
            'tonnetz_mean': np.mean(tonnetz)
        }

        # Adding MFCCs and Chroma features
        for i in range(mfccs.shape[0]):
            features[f'mfccs_mean_{i}'] = np.mean(mfccs[i, :])

        for i in range(chroma.shape[0]):
            features[f'chroma_mean_{i}'] = np.mean(chroma[i, :])

        # Adding Polynomial features
        for i in range(p_features.shape[0]):
            features[f'poly_features_{i}'] = np.mean(p_features[i, :])

        return features

    def process_audio_data(self, audio_dict):
        """
        Process each audio scene, extract features, and compile them into a single DataFrame.
        """
        feature_list = []

        for scene_id, data in audio_dict.items():
            waveform = data['waveforms'][0].numpy()  # Ensure waveform is a NumPy array
            features = self.extract_features(waveform)
            features['scene_id'] = scene_id  # Add scene_id to the features dictionary
            feature_list.append(features)

        # Create a DataFrame from the list of feature dictionaries
        combined_features_df = pd.DataFrame(feature_list)
        return combined_features_df

    def run(self):
        audio_dict = self.data_loader()
        audio_df = self.process_audio_data(audio_dict)
        return audio_df


if __name__ == '__main__':
    sample_rate = 44100
    train = AudioWaveformsConverter('train_data.pkl', sample_rate)
    # test = AudioWaveformsConverter('test_data.pkl', sample_rate)
    # dev = AudioWaveformsConverter('dev_data.pkl', sample_rate)

    train_audio_df = train.run()
    # test_audio_df = test.run()
    # dev_audio_df = dev.run()

    with open('train_audio_df.pkl', 'wb') as f:
        pickle.dump(train_audio_df, f)
#    with open('test_audio_df.pkl', 'wb') as f:
#        pickle.dump(test_audio_df, f)
#    with open('dev_audio_df.pkl', 'wb') as f:
#        pickle.dump(dev_audio_df, f)