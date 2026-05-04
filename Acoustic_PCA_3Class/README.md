# Acoustic PCA 3-Class Classifier

![Acoustic PCA 3-Class GUI](GUI.png)

## Overview

This prototype demonstrates an ecoacoustic audio analysis workflow for visualising and classifying WAV recordings. It combines PCA-based feature-space visualisation with a 3-class Random Forest classifier.

The system classifies uploaded audio into:

- Bird / Keep
- No-bird / Background
- Silence / Drop

## System Functionality

The GUI includes the following buttons:

- **Load Saved Map**  
  Loads the saved acoustic feature map and displays the PCA map.

- **Load 3-Class Classifier**  
  Loads the trained Random Forest model and scaler.

- **Load WAV + Predict**  
  Uploads a WAV file, extracts acoustic features, projects the audio windows into PCA space, and predicts the class.

- **START Trajectory**  
  Animates the WAV file windows on the PCA map.

- **STOP**  
  Stops the trajectory animation.

## Methodology

```text
WAV audio
→ window segmentation
→ feature extraction
→ PCA visualisation
→ Random Forest classification
→ Bird / No-bird / Silence decision


Dataset sources:

DCASE 2018 Bird Audio Detection Task
https://dcase.community/challenge2018/task-bird-audio-detection
Kaggle Bird Audio Detection competition
https://www.kaggle.com/competitions/bird-audio-detection
