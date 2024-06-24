# Sentiment Analysis with F.R.I.E.N.D.S Dataset

## Overview
For the first time dealing with audio data, we came up with [MELD](https://affective-meld.github.io/), a dataset that contains a collection of video clips from the popular television show F.R.I.E.N.D.S.
Each video clip is some utterance within a scene from the famous show, along with accompanying tabular data providing information about the scenes' utterances, sentiment, emotion, and more.
When humans assess sentiment, they naturally rely on various cues such as the speaker's tonne, energy, face expressions, etc...
Hence, when dealing with sentiment analysis, we believe that the data you should be looking for, should be similar to the one that we humans use.
Performing sentiment analysis for parts of series like F.R.I.E.N.D.S when relying on textual data alone (and without any video/audio information), may be very difficult (we believe that good amount of sarcasm may mislead such NLP model).
In our Medium (place holder for medium article) we ellaborately explained our motivation for the following research questions:

1. Can the inclusion of audio features improve the performance of an existing NLP model?
2. Can these features improve the performance of any predictive model used on this data?

This repository contains the codebase of our work.

## Contents
- [mp4_to_mp3.py](https://github.com/lvyor307/nlp-with-audio/blob/main/mp4_to_mp3.py) - Convert our mp4 data to mp3 format.
- [utils.py](https://github.com/lvyor307/nlp-with-audio/blob/main/utils.py) - Contains our utility functions.
- [audio](https://github.com/lvyor307/nlp-with-audio/tree/main/audio) - Directory that contains the scripts that perform the relevant actions on the audio datasets, in order to attain the required audio features.
- [modelling](https://github.com/lvyor307/nlp-with-audio/tree/main/modelling) - Directory that contains several scripts that represent out different models.
- [text](https://github.com/lvyor307/nlp-with-audio/tree/main/text) - Directory that contains the script that performs some manipulation on the textual data.
- [DescriptiveStatistics](https://github.com/lvyor307/nlp-with-audio/tree/main/DescriptiveStatistics) - Directory that contains the script that generate most of the plots as part of our EDA section.


## Usage
1. Clone the following repository to your local machine.
2. Download [MELD](https://affective-meld.github.io/).
3. Extract the data to your PyCharm project.
4. Install the necessary packages by checking `requirements.txt`
5. Execute the provided script `mp4_to_mp3.py` to convert to audio files to mp3 dormat. You do not need to input specific paths.
6. (optional) Exectue the provided script `audio/run_audio_features.py` to attain the audio features from the audio data.
   Similarly, you may just examine `audio/train_fe.csv` and `audio/text_fe.csv` to attain the desired output.
7. Run the different scripts within `modelling/` directory to attain the results of the different models.

