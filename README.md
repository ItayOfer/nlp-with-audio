# FRIENDS Scenes Dataset

## Overview
The FRIENDS Scenes Dataset contains a collection of video clips from the popular television show FRIENDS. Each video clip is some scene from the famous show, along with accompanying tabular data providing information about the scenes' utterances, sentiment, emotion, and more.
You can download the datasets here: 
https://affective-meld.github.io/. The datasets are already divided to train, test and dev.
## Contents
- [Introduction](#introduction)
- [Dataset Structure](#dataset-structure)
- [Data Fields](#data-fields)
- [Script for Audio Extraction](#script-for-audio-extraction)


## Introduction
FRIENDS is a beloved sitcom known for its humor, memorable characters, iconic moments, and lots of sarcasm.
We aim to use these datasets for sentiment and emotions analysis. We also aim to compare the results of a NLP model with an Audio featured model.
We believe that the sarcasm of this show might mislead the NLP model.

## Dataset Structure
The dataset is organized into two main components:
1. Video Clips: This directory contains video files (.mp4 format) of individual scenes extracted from FRIENDS episodes.
2. Tabular Data: This dataset also includes a CSV file providing structured information about each scene, including utterance(written), sentiment analysis, emotion classification, and more.

## Data Fields
The tabular data includes the following fields(and more):
- **Dialogue_ID**: Unique identifier for the dialogue in the scene.
- **Utterance_ID**: Unique Identifier for the utterance in scene.
- **Sentiment**: Sentiment analysis of the dialogue (e.g., positive, negative, neutral).
- **Emotion**: Emotion classification of the dialogue (e.g., happy, sad, angry).
- **Speaker**: Character speaking the dialogue.


## Script for Audio Extraction
To facilitate further analysis or applications requiring audio data, we provide a Python script for extracting audio from the video clips. This script utilizes the `moviepy.editor` library to extract audio in MP3 format from the MP4 video files.

### Usage
1. Install the necessary packages by checking `requirements.txt`.
2. Extract the data to your PyCharm project.
3. Execute the provided script `DataProvider.py`. You do not need to input specific paths.

