# coding: utf-8
import cv2
import csv
import os
import numpy as np
from keras import Sequential, optimizers
from keras.layers import Lambda, Dense, Flatten, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from sklearn import model_selection

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

LEFT_ANGLE_CORRECTION = 0.4
RIGHT_ANGLE_CORRECTION = -0.3
REUSE = False
EPOCHES = 1
LOG = "./backup/all.csv"

# fetch data from the driving log
def data(file):
    lines = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines
        
# generator for fit_generator
def generate_data(data, batchSize = 32):
    while True:
        data = shuffle(data)
        for i in range(0, len(data), int(batchSize/4)):
            X_batch = []
            y_batch = []
            chunk = data[i: i+int(batchSize/4)]
            for line in chunk:
                center_image = cv2.imread(line[0])
                flipped_image = np.fliplr(center_image)
                left_image = cv2.imread(line[1])
                right_image = cv2.imread(line[2])
                angle = float(line[3])

                #center image
                X_batch.append(center_image)
                y_batch.append(angle)
                
                #flipped
                X_batch.append(flipped_image)
                y_batch.append(-angle)
                
                # left camera image and left-corrected angle
                X_batch.append(left_image)
                y_batch.append(angle + LEFT_ANGLE_CORRECTION)

                # right camera image and right-corrected angle
                X_batch.append(right_image)
                y_batch.append(angle + RIGHT_ANGLE_CORRECTION)
            # converting to numpy array
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield shuffle(X_batch, y_batch)

#Diving data among training and validation set

lines = data(LOG)
training_data, validation_data = train_test_split(lines, test_size = 0.2)

version = "-nvidia-gen-sharp"

# Keras requires numpy array
if not REUSE:
    # Build NVIDIA MODEL
    print("--- New model ---")
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer="adam")
    model.fit_generator(generate_data(training_data, 64), 
                        samples_per_epoch = len(training_data)*4, 
                        nb_epoch = EPOCHES, 
                        validation_data=generate_data(validation_data, 64), 
                        nb_val_samples=len(validation_data))

else:
    print("--- Loading model {} ---".format(version))
    model = load_model("../model{}.h5".format(version))
    model.fit_generator(generate_data(training_data), 
                        samples_per_epoch = len(training_data)*4, 
                        nb_epoch = EPOCHES, 
                        validation_data=generate_data(validation_data), 
                        nb_val_samples=len(validation_data))


save_version = "-demo"
model.save("../model{}.h5".format(save_version))
print("Successfully saved the model in ../model{}.h5".format(save_version))