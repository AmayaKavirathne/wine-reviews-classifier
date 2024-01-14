# wine-reviews-classifier

This repository contains a Python script for classifying wine reviews as either low tier(low rated) or high tier(high rated) based on their points. The classification is performed using two different models: a dense neural network and a Long Short-Term Memory (LSTM) neural network.

# Requirements
Make sure you have the following dependencies installed:

- numpy
- pandas
- matplotlib
- tensorflow
- tensorflow_hub

# Dataset
The script uses a wine reviews dataset (wine-reviews.csv). It includes information about the country, description, points, price, variety, and winery of each wine. The dataset is loaded into a Pandas DataFrame, and null values in the 'description' and 'points' columns are dropped.

# Data Preprocessing
The script preprocesses the data by classifying it into two categories: low tier (points < 90) and high tier (points >= 90). The relevant columns ('description' and 'label') are selected for further processing.

# Model Development - Dense Neural Network
## Data Embedding\
The script utilizes transfer learning with an embedding layer from TensorFlow Hub to convert wine descriptions into numerical representations.

## Model Architecture
The dense neural network model consists of:

1. An embedding layer for text-to-vector transformation.
2. Two dense layers with ReLU activation and dropout for non-linearity and regularization.
3. The final output layer with a sigmoid activation function for binary classification.
   
## Training and Evaluation
The model is compiled using binary cross-entropy loss and the Adam optimizer. Training is performed for 5 epochs, and the accuracy and loss are plotted over time. The script evaluates the model on the training, validation, and test datasets.

# Model Development - LSTM Neural Network
## Text Vectorization
For the LSTM model, a text vectorization layer is used to convert text data into sequences of integers.

## Model Architecture
The LSTM model consists of:

1. A text vectorization layer to preprocess input text.
2. An embedding layer to map integer-encoded words to dense vectors.
3. An LSTM layer for sequence modeling.
4. Two dense layers with ReLU activation and dropout.
5. The final output layer with a sigmoid activation function.

## Training and Evaluation
Similar to the dense model, the LSTM model is compiled with binary cross-entropy loss and the Adam optimizer. Training is performed for 5 epochs, and accuracy and loss are plotted over time. The script evaluates the LSTM model on the training, validation, and test datasets.
