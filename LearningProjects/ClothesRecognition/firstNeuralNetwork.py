import time, os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import csv


def train_a_neural_network(numOfEpoch=None):
    if numOfEpoch is None:
        numOfEpoch = int(input("How many epoch: "))
    #  This is a function to save the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # Trains the model for 10 epochs and saves the model after each epoch
    model.fit(train_images, train_labels, epochs=numOfEpoch, callbacks=[cp_callback])

   #  model.fit(train_images, train_labels, epochs=numOfEpoch)


def use_model(display=False):
    model.load_weights(checkpoint_path)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    predictions = model.predict(test_images)
    print('Test accuracy:', test_acc)
    print("Test loss", test_loss)
    input("Press enter to begin test")
    if display:  # Display the images with the correct label and the prediction made
        for x in range(len(test_images)):
            plt.title(
                f"Model prediction: {class_names[np.argmax(predictions[x])]}, Label: {class_names[test_labels[x]]}")
            plt.imshow(test_images[x], cmap=plt.cm.binary)
            plt.axis("off")
            plt.show(block=False)
            plt.figure()
            time.sleep(0.2)
            plt.close()


epochBands = [1, 3, 5, 10, 30, 50, 100]
# n = int(input("Which model are you using: "))
data = [[0 for a in range(7)] for b in range(6)]
for n in range(1):
    number = f"{n:0>3}"  # which file to save a model to / load a model from
    if int(number[1]) <= 6 and int(number[2]) <= 5:
        print(number)
        epochNum = epochBands[int(number[1])]
        checkpoint_path = f"training_{number}/cp.ckpt"
        #  checkpoint_dir = os.path.dirname(checkpoint_path)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        fashion_mnist = tf.keras.datasets.fashion_mnist  # load dataset
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training

        if number[0] == "0":
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),  # input layer
                keras.layers.Dense(128, activation='relu'),  # hidden layer
                keras.layers.Dense(10, activation='softmax')  # output layer
            ])
        elif number[0] == "1":
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

        train_images = train_images / 255.0  # Transform values to the range of 0 - 1
        test_images = test_images / 255.0

        train_a_neural_network(epochNum)
        use_model(True)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
        data[2 * int(number[0])][int(number[1])] += test_acc / 5
        data[2 * int(number[0]) + 1][int(number[1])] = epochNum



# 7h and 44 minutes!
#  2023-02-25 19:21:32.110821
#  2023-02-26 03:05


# need to correct values:
# accidentally did 6 values per test
# currently finding average by dividing by 5
# times all values by 5/6


# 1.018779993057251,1.0390999913215637,1.047000002861023,1.0611599802970888,1.063639986515045,1.0635200023651121,1.0617999792098998
#
# 1,3,5,10,30,50,100
#
# 1.0130599975585939,1.0401000022888183,1.0511600017547607,1.0605200052261352,1.0688400268554688,1.0671000003814697,1.0651399970054627
#
# 1,3,5,10,30,50,100
#
# 1.0081600189208983,1.0384399890899658,1.0474999904632567,1.0605999946594238,1.0680400013923645,1.0659600019454958,1.067799985408783
#
# 1,3,5,10,30,50,100

