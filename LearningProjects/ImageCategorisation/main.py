import time

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

number = int(input("Which model are you using: "))
checkpoint_path = f"training_{number}/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

from matplotlib import pyplot as plt
trainData = tf.Variable(np.load("Training_Data.npy"))
evalData = tf.Variable(np.load("Eval_Data.npy"))
trainLabels = tf.Variable(np.load("Training_Labels.npy"))
evalLabels = tf.Variable(np.load("Eval_Labels.npy"))

class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

if number == 1:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(500, 500)),  # input layer (1)
        keras.layers.Dense(200, 200, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(10, activation='softmax')])  # output layer (3)


model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

trainData /= 255.0
evalData /= 255.0

def train_a_neural_network():
    numOfEpoch = int(input("How many epoch: "))
    # This is a function to save the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # Trains the model for 10 epochs and saves the model after each epoch
    model.fit(trainData, trainLabels, epochs=numOfEpoch, callbacks=[cp_callback])
    # Evaluate the model which has benn created


def use_model(display=False):
    model.load_weights(checkpoint_path)
    test_loss, test_acc = model.evaluate(evalData,  evalLabels, verbose=1)
    predictions = model.predict(evalData)
    print('Test accuracy:', test_acc)
    print("Test loss", test_loss)
    if display:
        for x in range(len(evalData.shape[0])):
            plt.title(f"Prediction:{class_names[np.argmax(predictions[x])]}       Label:{class_names[evalLabels[x]]}")
            plt.imshow(evalData[x])
            plt.show(block=False)
            plt.figure()
            time.sleep(0.2)
            plt.close()



train_a_neural_network()
# use_model(True)
