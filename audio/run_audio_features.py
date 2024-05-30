import pickle
from audio.audio_features_extractor import AudioFeaturesExtractor
from audio.audio_wave_forms import AudioWaveformsConverter


def run_audio_fe():
    """
    This function runs the AudioFeaturesExtractor class on the train, test and dev datasets.
    :return:
    """
    train_afe = AudioFeaturesExtractor('../train_audio', '../MELD.Raw/train/train_sent_emo.csv')
    test_afe = AudioFeaturesExtractor('../test_audio', '../MELD.Raw/test_sent_emo.csv')
    dev_afe = AudioFeaturesExtractor('../dev_audio', '../MELD.Raw/dev_sent_emo.csv')
    train_dict = train_afe.run()
    test_dict = test_afe.run()
    dev_dict = dev_afe.run()

    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_dict, f)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_dict, f)
    with open('dev_data.pkl', 'wb') as f:
        pickle.dump(dev_dict, f)


def run_audio_wave_form(sample_rate=16000):
    """
    This function runs the AudioWaveformsConverter class on the train, test and dev datasets.
    :param sample_rate: the sample rate of the audio files
    :return:
    """
    train = AudioWaveformsConverter('train_data.pkl', sample_rate)
    test = AudioWaveformsConverter('test_data.pkl', sample_rate)
    dev = AudioWaveformsConverter('dev_data.pkl', sample_rate)
    print('train start')
    train_audio_df = train.run()
    print('train done')
    print('test start')
    test_audio_df = test.run()
    print('test done')
    print('dev start')
    dev_audio_df = dev.run()
    print('dev done')
    dev_audio_df.to_csv('dev_fe.csv')
    train_audio_df.to_csv('train_fe.csv')
    test_audio_df.to_csv('test_fe.csv')


if __name__ == '__main__':
    # run_audio_fe()  # create the dict files and saves them as a pickle file
    run_audio_wave_form()  # create the csv data files
