import time

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from matplotlib import pyplot as plt
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = tf.keras.datasets.fashion_mnist  # load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training
# print(train_images.shape)
# print(train_images[0, 23, 23])
# print(train_labels)
#
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
        keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(10, activation='softmax') # output layer (3)
    ])
model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
train_images = train_images / 255.0
test_images = test_images / 255.0

def train_a_neural_network():
    # This is a function to save the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # Trains the model for 10 epochs and saves the model after each epoch
    model.fit(train_images, train_labels, epochs=10, callbacks=[cp_callback])
    # Evaluate the model which has benn created


def use_model():
    model.load_weights(checkpoint_path)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
    print('Test accuracy:', test_acc)
    print("Test loss", test_loss)
    # Predict the labels for the training set
    predictions = model.predict(test_images)
    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])
    for x in range(len(test_images)):
        plt.imshow(test_images[x])
        plt.title(f"{class_names[np.argmax(predictions[x])]} = {class_names[test_labels[x]]}")
        plt.show()
        plt.close()

# train_a_neural_network()
use_model()