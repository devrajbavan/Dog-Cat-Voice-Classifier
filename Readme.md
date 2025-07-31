# **Cat vs. Dog Voice Classifier**

This project uses deep learning to distinguish between the sounds of cats meowing and dogs barking. It involves a complete pipeline that starts with raw audio files, converts them into visual spectrograms, and then trains a Convolutional Neural Network (CNN) to perform the classification.

## **Project Description**

The primary goal of this project is to build an accurate audio classification model. Raw audio waveforms of cat meows and dog barks are transformed into Mel spectrograms, which are 2D image representations of the sound. These images are then used as input to a CNN, which learns to identify the unique visual patterns associated with each animal's voice. The final model can take a new audio clip, convert it, and predict whether it contains a cat or a dog.

## **Model Explanation**

The classification model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture is designed to effectively process the 2D spectrogram images:

1. Input Layer: Accepts spectrogram images of size (300, 300, 3).  
2. Convolutional Layers: The model consists of three sequential Conv2D layers with increasing filters (16, 32, 64). These layers apply filters to detect low-level features like edges and textures in the spectrograms. Each is followed by a relu activation function.  
3. Pooling Layers: After each Conv2D layer, a MaxPooling2D layer is used to downsample the feature maps. This reduces the spatial dimensions, making the model more computationally efficient and helping it generalize by retaining the most important features.  
4. Flatten Layer: This layer converts the 2D feature maps into a 1D vector to be fed into the dense layers.  
5. Dense Layers:  
   * A fully connected Dense layer with 128 neurons and a relu activation function acts as a classifier.  
   * The final Dense output layer has a single neuron with a sigmoid activation function. This outputs a probability score between 0 (Cat) and 1 (Dog).

The model is compiled using the adam optimizer and BinaryCrossentropy loss function, which are standard choices for binary classification tasks.

## **Tech Used (Script-wise)**

This project is composed of several scripts, each with a specific purpose and technology stack.  
1\. Audio to Spectrogram Conversion

* Purpose: Converts .wav or .mp3 audio files into clean PNG spectrogram images without axes or backgrounds.  
* Technologies:  
  * Python: The core programming language.  
  * librosa: For loading audio files and generating Mel spectrograms.  
  * matplotlib: To render and save the spectrograms as PNG images.  
  * os: For navigating the file system.

2\. Model Training (DvC\_train.py)

* Purpose: Trains the CNN using the generated spectrograms.  
* Technologies:  
  * TensorFlow & Keras: For building, compiling, and training the neural network.  
  * ImageDataGenerator: To load images from directories in batches, apply normalization (rescale=1./255), and split data for training and validation.

3\. Automated Testing and Evaluation

* Purpose: Runs inference on a test set of audio files, generates a confusion matrix, and provides a classification report.  
* Technologies:  
  * TensorFlow & Keras: For loading the pre-trained model (meow\_vs\_Bark.h5).  
  * librosa & matplotlib: To perform on-the-fly audio-to-spectrogram conversion for test files.  
  * numpy: For numerical operations and data manipulation.  
  * scikit-learn: To compute and display the confusion\_matrix and classification\_report for model performance evaluation.

## **Data Used**

Find converted data through the link below:
https://drive.google.com/file/d/1a8tW9lpqB6_vrMqe8bCb5ryWspRKmKV9/view?usp=sharing

## **Directory Structure**

Here is the recommended directory structure for the project:  
text

`/Dogs_vs_Cats_Voice_Classifier`  
`│`  
`├── train/`  
`│   ├── cats/`  
`│   │   ├── cat_spectrogram_1.png`  
`│   │   └── ...`  
`│   └── dogs/`  
`│       ├── dog_spectrogram_1.png`  
`│       └── ...`  
`│`  
`├── voice_data/`  
`│   ├── cat990.wav`  
`│   ├── dog1.wav`  
`│   └── ...`  
`│`  
`├── scripts/`  
`│   ├── audio_conversion.py`  
`│   ├── DvC_train.py`  
`│   └── automated_test.py`  
`│`  
`├── meow_vs_Bark.h5`  
`├── TrueLabels.txt`  
`└── README.md`

* train/: Contains the spectrogram images used for training the model, organized into cats and dogs subdirectories.  
* voice\_data/: Holds the raw audio test files (.wav, .mp3).  
* scripts/: Contains all the Python scripts for the project.  
* meow\_vs\_Bark.h5: The saved, trained Keras model.  
* TrueLabels.txt: The file containing the ground truth labels for the test audio data.  
* README.md: This file.

## **Accomplishments**

* End-to-End Pipeline: Successfully created a full machine learning pipeline that processes raw audio, converts it into a machine-readable format (spectrograms), and trains a model for classification.  
* Automated Evaluation: Developed an automated script that evaluates the model's performance on a test dataset, providing clear metrics like a confusion matrix and a classification report.  
* Efficient Data Handling: Implemented a system that correctly reads ground truth labels from an external file and matches them with predictions for a robust evaluation.  
* Practical Application: Built a functional classifier that can be used to predict whether a new, unseen audio clip is from a cat or a dog.

