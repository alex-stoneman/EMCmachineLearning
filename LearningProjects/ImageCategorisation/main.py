import time

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pygame

trainImages = np.load("Training_Data.npy")
evalImages = np.load("Eval_Data.npy")
trainLabels = np.load("Training_Labels.npy")
evalLabels = np.load("Eval_Labels.npy")

# print(trainImages.shape, trainImages[0])
# print(evalImages.shape, evalImages[0])
# print(trainLabels.shape, trainLabels[0])
# print(evalLabels.shape, evalLabels[0])
class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

number = int(input("Which model are you using: "))
checkpoint_path = f"training_{number}/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if number == 1:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(50, 50)),  # input layer (1)
        keras.layers.Dense(64, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(6, activation='softmax')])  # output layer (3) - 6 labels
else:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(50, 50)),  # input layer (1)
        keras.layers.Dense(40, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(40, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(6, activation='softmax')])  # output layer (3) - 6 labels


model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

trainData = trainImages / 255.0
evalData = evalImages / 255.0

def train_a_neural_network():
    numOfEpoch = int(input("How many epoch: "))
    # This is a function to save the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # Trains the model for 10 epochs and saves the model after each epoch
    model.fit(trainData, trainLabels, epochs=numOfEpoch, callbacks=[cp_callback], batch_size=4)
    # Evaluate the model which has benn created
    print("Ends here")

def use_model(display=False):
    model.load_weights(checkpoint_path)
    test_loss, test_acc = model.evaluate(evalData,  evalLabels, verbose=1)
    predictions = model.predict(evalData)
    print('Test accuracy:', test_acc)
    print("Test loss", test_loss)

    pygame.init()
    screen = pygame.display.set_mode((200, 200))
    font = pygame.font.SysFont("freesansbold.ttf", 24)
    if display:
        for x in range(len(evalData)):
            # try:
            #     print("Prediction:", class_names[np.argmax(predictions[x])])
            #     print("Label:", class_names[evalLabels[x]])
            #     print("------------")
            # except TypeError:
            #     print(np.argmax(predictions[x]))
            #     print(evalLabels[x])
            #     raise TypeError
            screen.fill((255, 255, 255))
            temp = pygame.pixelcopy.make_surface(evalImages[x])
            screen.blit(temp, (75, 150))

            img = font.render('hello', True, "red")
            screen.blit(img, (50, 20))

            pygame.display.flip()



train_a_neural_network()
use_model(True)

# pygame.init()
# pygame.font.init()
# screen = pygame.display.set_mode((200, 200))
# font = pygame.font.SysFont("freesansbold.ttf", 24)
# for item in evalImages:
#     # screen.fill((255, 255, 255))
#     print("e")
#     temp = pygame.pixelcopy.make_surface(item)
#     screen.blit(temp, (75, 150))
#
#     # img = font.render('hello', True, "red")
#     # screen.blit(img, (50, 20))
#
#     pygame.display.flip()
#     time.sleep(1)
#     print("new")