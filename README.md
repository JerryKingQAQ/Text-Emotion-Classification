# Emotional-Text-Classification
# Text Emotion Classification Project README

## Overview

This repository is focused on text emotion classification, employing three different models: GCNN, DNN, and GCNN_LSTM. Each model provides a unique approach to understanding and classifying the emotions conveyed in text data. The project includes comprehensive code for these models and scripts for both training and testing.

## Models Description

1. **GCNN (Graph Convolutional Neural Network):** A model that leverages graph-based learning in processing text data, ideal for capturing contextual information and relationships between words.

2. **DNN (Deep Neural Network):** A traditional deep learning approach that uses layers of neural networks to extract features and perform classification.

3. **GCNN_LSTM (Graph Convolutional Neural Network with Long Short-Term Memory):** A hybrid model combining GCNN's ability to capture contextual data with LSTM's proficiency in handling sequences, making it powerful for sequential text data.

## Usage Instructions

### Training

- To train a model, navigate to `train.py`.
- Select your desired dataset and model.
- Execute the script to start the training process.
- The system will automatically save the trained model.

### Testing

- After training, run `main.py` to test the model.
- The script will load the trained model and run it on the test dataset.
- You will receive a classification report detailing the model's performance.

## Installation

1. Clone the repository:

```git clone [[repository_url]](https://github.com/JerryKingQAQ/Emotional-Text-Classification)https://github.com/JerryKingQAQ/Emotional-Text-Classification```
