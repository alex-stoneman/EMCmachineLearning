import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

from matplotlib import pyplot as plt
trainData = np.load("Training_Data.npy")
evalData = np.load("Eval_Data.npy")
trainLabels = np.load("Training_Labels.npy")
evalLabels = np.load("Eval_Labels.npy")

# print(trainData.shape, trainData[0])
# print(evalData.shape, evalData[0])
# print(trainLabels.shape, trainLabels[0])
# print(evalLabels.shape, evalLabels[0])
class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

number = int(input("Which model are you using: "))
checkpoint_path = f"training_{number}/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if number == 1:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(500, 500, 3)),  # input layer (1)
        keras.layers.Dense(200, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(10, activation='softmax')])  # output layer (3)


model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

trainData = trainData / 255.0
evalData = evalData / 255.0

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
        for x in range(len(evalData)):
            try:
                print("Prediction:", class_names[np.argmax(predictions[x])])
                print("Label:", class_names[evalLabels[x]])
                print("------------")
            except TypeError:
                print(np.argmax(predictions[x]))
                print(evalLabels[x])
                raise TypeError

            # plt.title(f"Prediction:{class_names[np.argmax(predictions[x])]}       Label:{class_names[evalLabels[x]]}")
            # plt.imshow(evalData[x])
            # plt.show(block=False)
            # plt.figure()
            # time.sleep(0.2)
            # plt.close()



# train_a_neural_network()
use_model(True)


# Might need to change the labels from (n, 1) to be (n)