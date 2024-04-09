import librosa
import pandas as pd
import numpy as np

class AudioFeaturesExtractor:

    def __init__(self,audio_path):
        self.audio_path = audio_path
        self.audio_features_dict = {}

    def audio_feature_extraction(self):

        y,sr = librosa.load(self.audio_path)

        self.audio_features_dict['mfccs'] = librosa.feature.mfcc(y=y, sr=sr)
        self.audio_features_dict['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr)
        self.audio_features_dict['spectral_contrast'] = librosa.feature.spectral_contrast(y=y, sr=sr)
        self.audio_features_dict['tonnetz'] = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        self.audio_features_dict['tempo'] = librosa.beat.beat_track(y=y, sr=sr)
        self.audio_features_dict['rms_energy'] = librosa.feature.rms(y=y)
        self.audio_features_dict['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2')
                                                     , fmax=librosa.note_to_hz('C7'))

        features_dict = {
            'MFCCs': mfccs,
            'Chroma': chroma,
            'Spectral_Contrast': spectral_contrast,
            'Tonnetz': tonnetz,
            'Tempo': tempo,
            'RMS_Energy': rms_energy,
            'Zero_Crossing_Rate': zero_crossing_rate,
            'Pitch': f0
        }

y, sr = librosa.load('../train_audio/dia0_utt0.mp3')
# Mel-Frequency Cepstral Coefficients (MFCCs)
# Chroma Feature
# Spectral Contrast
# Tonnetz (Tonal Centroid Features)
# Tempo
# Energy and RMS
# Zero Crossing Rate
# Pitch

# For demonstration, let's create a feature with a single value per audio file
single_value_features = {
    'tempo': tempo,
    'overall_rms_energy': np.mean(rms_energy)
}

# Multi-value features (we're using the mean across time frames here for simplicity)
multi_value_features = {
    'mfccs': np.mean(mfccs, axis=1),
    'chroma': np.mean(chroma, axis=1),
    'spectral_contrast': np.mean(spectral_contrast, axis=1),
    'tonnetz': np.mean(tonnetz, axis=1),
    'zero_crossing_rate': np.mean(zero_crossing_rate, axis=1)
}

print(single_value_features)
print(multi_value_features)
# Combine single and multi-value features into a DataFrame
#features_data = {**single_value_features}
#features_data_b = {**multi_value_features}
#df = pd.DataFrame(features_data, index=[0])  # Use index=[0] to indicate a single observation
#df_b = pd.DataFrame(features_data_b)
# If you have multiple audio files, you would append each new set of features as a new row

#print(df.head())
#print(df_b.head())