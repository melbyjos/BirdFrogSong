# Bird and Frog Song Classification

This repository contains explorations of audio recording data with a focus on classifying bird and frog songs.

Much of the work included here was conducted as part of the Erdos Institute Data Science Bootcamp during May and June of 2023. This is collaborative work of Joseph Melby, Gautam Prakriya, and Florian Stecker.

**Team Members:** Joseph Melby, Gautam Prakriya, and Florian Stecker

**Project Goals:** Classify frog and bird species in audio recordings from Puerto Rico.

**Societal Impact:** Classifying bird, frog, and other animal species by their communication techniques is of interest to a wide variety of stakeholders, including conservation-focused organizations and governments, endangered and threatened species advocates, and even hobbyists throughout the general public. Our motivation for this project aligns with the mission of these stakeholders: the more we know about the communication dynamics of these species of rainforest dwellers, the better we can act to protect their homes. This project gave us a deeper appreciation of the important efforts of conservationists and scientists dedicated to understanding these frog and bird species.

**Description of Dataset:** The dataset came from a Kaggle competition called “Rainforest Connection Species Audio Detection” and consisted of 4727 minutes of sound recordings from various locations in the Puerto Rican rainforest, together with 1216 labels identifying subsections of 0.2 - 8s long as the song of one of 24 selected species. Of the 24 species, 11 are frogs and 13 are birds, and many of them are endemic to Puerto Rico.

**Model 1:** For each of the 1216 labels we extracted a 2 second sound clip from the recordings; if the song was longer than 2 seconds, we chose a random 2 second subsegment, if it was shorter than 2 seconds, we added a random amount of context before and after to pad it to 2 seconds. Then we converted the clips into 64x44 pixel log-mel-spectrograms, flattened them, and ran various classification algorithms on the resulting 2816-dimensional vectors. We found the best results using a logistic regression classifier. In particular, we were able to tune our model using various regularization techniques to tame its tendency to overfit. Given this, the model was able to identify the correct species in 68% of the test cases. We also tested SVM and KNN models to a lesser degree of accuracy.

**Model 2:** In order to improve our results, we explored existing models for these datasets, namely using the BirdNET model. We extracted 3 second sound clips in the same manner as above, and transformed these into 1024-dimensional feature vectors using the pretrained BirdNET neural network. The BirdNET model applies a convolutional neural network to mel-spectrograms in order to classify a recording into one of over 3300 bird species. It has been trained on over 30000 hours of sound data and is freely available online. To adapt the model to the Puerto Rican species in our dataset (which are not present in the original BirdNET training data), we cut of the final layer of the neural network to obtain 1024-dimensional feature embeddings, and then train a custom classifier on these embeddings. Again, a regularized logistic regression model gives the best results, identifying the correct species in 82.5% of test cases.

**Takeaways:** It is clear that this project and data merit far more time and exploration than was feasible in the time we had, but we are motivated to continue to learn what we can from the process.This project gave great insight into the challenges faced by conservation-minded parties to understand audible animal communication, especially the data collection challenges therein. With time, we would have loved to explore many different data preprocessing avenues, including noise removal and data augmentation techniques, more sophisticated image classification techniques, and applications of our methods to other animal sound datasets.

## Getting started

The birdsong data itself is not included in the repository. It was downloaded from three sources:

- The competition [Rainforest Connection Species Audio Detection](https://www.kaggle.com/competitions/rfcx-species-audio-detection/data) on kaggle.com.

  This contains recordings from the Puerto Rican rainforest and is the main dataset used in this repository. It has 1216 labeled songs of 24 species (11 frogs and 13 birds, many of them endemic to Puerto Rico), and over 70 hours of unlabeled soundscapes from Puerto Rico.

  To use it, download the zip file from Kaggle and unpack it into the directory `kaggle-data`.

- The [BirdCLEF 2023 challenge](https://www.kaggle.com/competitions/birdclef-2023/data), also on kaggle.com.

  This dataset contains 192 hours of labeled bird songs, uploaded by volunteers to the [xeno-canto.org](xeno-canto.org).

  To use it, download the zip file from Kaggle and unpack it into the directory `BirdCLEF`.

- Grasshopper recordings from [xeno-canto.org](xeno-canto.org).

  This dataset contains 82 hours of grasshopper recordings, and can be downloaded from [xeno-canto.org](xeno-canto.org) using their API.

Most of the code comes in the form of Jupyter notebooks. We use the python data science packages `sklearn`, `pandas`, and `keras` (which is part of `tensorflow`), as well as the audio library `librosa`.

Some of our classifiers use the embeddings provided by the pretrained neural network BirdNET (version 2.3). It can be downloaded from its [Github repository](https://github.com/kahst/BirdNET-Analyzer/tree/main/checkpoints).

## Data preprocessing

In order to generate spectrogram datasets, start with the Generate_spectrograms.ipynb notebook. This will generate the spectrograms for the identified bird and frog songs identified in the dataset. Once the data is generated, you can explore it using the BirdNET Spectrograms Confusion Matrices.ipynb notebook. This notebook makes heavy use of the classifiers.py module, which contains a custom class for analyzing data using multiple classification algorithms.

In order to generate the embedding data, run the BirdNET embeddings of Kaggle data.ipynb notebook. This will load the pre-trained BirdNET model with the base layer removed and embed the bird song dataset into 1024-dimensional space. From there, you can explore this data using the BirdNET embeddings Confusion Matrices.ipynb notebook. This again makes use of the classifiers.py module, and it allows you to introduce a custom classifier as an addition to the last layer of the BirdNET neural network.

## Classification

(todo: describe how to use classifiers to study preprocessed data)
