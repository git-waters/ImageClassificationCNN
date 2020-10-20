#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
import random
import imutils
import cv2
import numpy as np
import tensorflow as tf
from kerastuner import RandomSearch
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, ZeroPadding2D, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow import keras
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.callbacks import EarlyStopping
import time

LOG_DIR = f"{int(time.time())}"
# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
# tf.set_random_seed(SEED)
tf.random.set_seed(SEED)
batch_size = 32
img_height = 64
img_width = 64
train_path = "D:/Work/Uni/COMP 309/Project Assignment/ProjectTemplate_python3.8/Train_data"
num_of_classes = 3
datadir = "D:/Work/Uni/COMP 309/Project Assignment/ProjectTemplate_python3.8/Train_data"
categories = ["cherry", "strawberry", "tomato"]
test_datadir = "D:/Work/Uni/COMP 309/Project Assignment/ProjectTemplate_python3.8/data/test"
print("GPU Free: ", len(tf.config.experimental.list_physical_devices("GPU")))


def construct_mlp_model():
    model = Sequential()
    model.add(Conv2D(32, 3, data_format='channels_last', activation='relu', padding='same',
                     input_shape=(img_width, img_height, 3)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    # model = Sequential()
    # model.add(Dense(units=64, activation='relu', input_dim=100))
    # model.add(Dense(units=10, activation='softmax'))
    # model.compile(loss='categorical_crossentropy',
    # optimizer='sgd',
    # metrics=['accuracy'])
    # return model

    model = Sequential()
    # Input Layer
    model.add(Conv2D(64, 3, data_format='channels_last', activation='relu', padding='same',
                     input_shape=(img_width, img_height, 3)))
    model.add(MaxPool2D(pool_size=2, strides=2))
    # Hidden Layer 1
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Hidden Layer 2
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, activation='relu', padding='same', strides=2))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Hidden Layer 3
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same', strides=2))
    model.add(MaxPool2D(pool_size=2, strides=2))


    # Fully Connected Layer
    model.add(Flatten())
    # 512 Neuron Layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # Output Layer
    model.add(Dense(num_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_training_data():
    # Enrich Data
    # Straw = 884 Cherry = 876 Tomato = 2159
    # Create Image Data Generator
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=50,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3,
        fill_mode="nearest")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=datadir,
        target_size=(img_height, img_width),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
        subset="training"

    )

    valid_generator = train_datagen.flow_from_directory(
        directory=datadir,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",  # Set as validation data
        seed=SEED
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_datadir,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )

    # valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    # datadir,
    # validation_split=0.2,
    # subset="validation",
    # label_mode="categorical",
    # seed=SEED,
    # image_size=(img_height, img_width),
    # batch_size=batch_size)

    # class_names = train_dataset.class_names
    # print(class_names)

    # Normalize
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    # normalized_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_dataset))
    # first_image = image_batch[0]
    # Pixel values are now [0,1]
    # print(np.min(first_image), np.max(first_image))

    # Configuring for performance (Avoiding I/O blocking)
    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # valid_dataset = valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    x, y = train_generator.next()
    for i in range(0, 1):
        image = x[i]
        plt.imshow(image)
        plt.show()

    return train_generator, valid_generator, test_generator


def train_model(model, train, validation):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Add your code here

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, restore_best_weights=True)

    model.fit_generator(
        generator=train,
        validation_data=validation,
        epochs=1000,
        callbacks=monitor

    )
    # Preprocessing (Enrichment)
    # Preprocessing (Normalisation)

    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("model/model.h5")
    print("Model Saved Successfully.")


def evaluate_model(model, test_data):
    score = model.evaluate_generator(test_data)
    print("Accuracy = ", score[1])


if __name__ == '__main__':
    train, validation, test = create_training_data()
    model = construct_model()
    model = train_model(model, train, validation)
    save_model(model)
    evaluate_model(model, test)
