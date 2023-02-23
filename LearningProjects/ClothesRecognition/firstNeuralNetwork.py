import time, os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

number = int(input("Which model are you using: ")) # which file to save a model to / load a model from
checkpoint_path = f"training_{number}/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = tf.keras.datasets.fashion_mnist  # load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training

print(train_images.shape)
print(train_images[0])
print(train_images[0][0])
print(train_images[0][0][0])
# print(train_images[0, 23, 23])
# print(train_labels)
#
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

if number in [11, 12, 13, 14, 15]:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer
        keras.layers.Dense(128, activation='relu'),  # hidden layer
        keras.layers.Dense(10, activation='softmax')  # output layer
    ])
elif number in [21, 22, 23, 24, 25]:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer
        keras.layers.Dense(128, activation='relu'),  # hidden layer
        keras.layers.Dense(128, activation='relu'),  # hidden layer
        keras.layers.Dense(10, activation='softmax')  # output layer
    ])
else:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer
        keras.layers.Dense(128, activation='relu'),  # hidden layer
        keras.layers.Dense(128, activation='relu'),  # hidden layer
        keras.layers.Dense(128, activation='relu'),  # hidden layer
        keras.layers.Dense(10, activation='softmax')  # output layer
        ])


model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

train_images = train_images / 255.0 # Transform values to the range of 0 - 1
test_images = test_images / 255.0

def train_a_neural_network():
    numOfEpoch = int(input("How many epoch: "))
    # This is a function to save the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # Trains the model for 10 epochs and saves the model after each epoch
    model.fit(train_images, train_labels, epochs=numOfEpoch, callbacks=[cp_callback])
    # Evaluate the model which has been created


def use_model(display=False):
    model.load_weights(checkpoint_path)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
    predictions = model.predict(test_images)
    print('Test accuracy:', test_acc)
    print("Test loss", test_loss)
    input("Press enter to begin test")
    if display: # Display the images with the correct label and the prediction made
        for x in range(len(test_images)):
            plt.title(f"{class_names[np.argmax(predictions[x])]} = {class_names[test_labels[x]]}")
            plt.imshow(test_images[x], cmap=plt.cm.binary)
            plt.axis("off")
            plt.show(block=False)
            plt.figure()
            time.sleep(0.2)
            plt.close()


# train_a_neural_network()
use_model(True)