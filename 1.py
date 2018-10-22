# import tensorflow as tf

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# val_loss, val_acc = model.evaluate(x_test, y_test)
# print (val_loss, val_acc)

# import matplotlib.pyplot as plt
# # plt.imshow(x_train[0], cmap = plt.cm.binary)
# # plt.show()
# print(x_train[0])

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard

# FIRST PART OF DATA SET TO CLEAN UP DATA

# DATADIR = "PetImages"
# CATEGORIES = ["Dog", "Cat"]
# IMG_SIZE = 50

# training_data = []

# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category)
#         class_name = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 training_data.append([new_array, class_name])
#             except Exception as e:
#                 pass

# create_training_data()
# # print(len(training_data))

# random.shuffle(training_data)

# # for sample in training_data:
# #     print(sample[1])

# X = []
# y = []

# for features, label in training_data:
#     X.append(features)
#     y.append(label)

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# pickle_out = open("X.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)

# print(X[1])

# END FIRST PART OF PROJ

NAME = "Cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=8, validation_split=0.1, callbacks=[tensorboard])



