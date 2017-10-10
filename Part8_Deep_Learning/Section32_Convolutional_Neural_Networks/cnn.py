# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git#egg=Theano

# Install Tensorflow
# https://www.tensorflow.org/install/install_windows
# Using anaconda, so follow the steps below
    # conda create -n tensorflow python=3.6
    # activate tensorflow
    # Installing CPU version for this
        # CPU version: pip install --upgrade tensorflow
        # GPU version: pip install --upgrade tensorflow-gpu
    
    # deactivate
    
# The above didn't work, so run this from Anaconda Prompt as administrator
# pip install --upgrade tensorflow

# Keras is a library that runs on Tensorflow and Theano
# Makes our code simpler

# Installing Keras (use Anaconda Prompt as administrator)
# pip install --upgrade keras

# Part 1 - Building the CNN
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.metrics import confusion_matrix
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
# Using 2D packages for images


# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part8_Deep_Learning/\
Section32_Convolutional_Neural_Networks")

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
# For Theano backend, input_shape = (3, 64, 64)
# For Tensorflow backend, input_shape = (64, 64, 3)
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3),
                             activation="relu",
                             input_shape=(64, 64, 3)))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer to account for overfitting and improve
# accuracy of the model
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3),
                             activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
# Full Connection layer
classifier.add(Dense(units=128, activation="relu"))

# Output layer
classifier.add(Dense(units=1, activation="sigmoid"))

# Compiling the CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

# Part 2 - Fitting the CNN to the images
# Go to https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set, steps_per_epoch=(8000/32), epochs=25,
                         validation_data=test_set, validation_steps=(2000/32))
