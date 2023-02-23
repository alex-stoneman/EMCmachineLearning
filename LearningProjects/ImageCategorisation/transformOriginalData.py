import os, pygame
import time

import numpy as np

from matplotlib import pyplot as plt
for file in ["Eval_Data.npy", "Eval_Labels.npy", "Training_Data.npy", "Training_Labels.npy"]:
    try:
        os.remove(file)
    except FileNotFoundError:
        pass

pygame.init()
width, height = (50, 50)
screen = pygame.display.set_mode((width, height))
trainImageData = np.zeros((278, width, height), np.int8) # There are 555 images
evalImageData = np.zeros((277, width, height), np.int8)
trainImageLabels = np.zeros(278, np.int8)
evalImageLabels = np.zeros(277, np.int8)
labels = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

data_fir = os.listdir("Chess-image-dataset")
count = 0
for item in data_fir:
    for image in os.listdir(f"Chess-image-dataset/{item}"):
        chessImage = pygame.image.load(f"Chess-image-dataset/{item}/{image}")
        try:
            chessImage = pygame.transform.smoothscale(chessImage, (width, height))
        except ValueError:
            chessImage = pygame.transform.scale(chessImage, (width, height))
            print(f"Chess-image-dataset/{item}/{image}")
        # screen.blit(chessImage, (0,0))
        # pygame.display.flip()
        values = pygame.surfarray.array3d(chessImage)
        # luminosity filter
        averages = [[(r * 0.298 + g * 0.587 + b * 0.114) for (r, g, b) in col] for col in values]
        arr = np.array([[avg for avg in col] for col in averages])
        if count % 2 == 0:
            trainImageData[count//2] = arr
            trainImageLabels[count//2] = labels.index(item)

        else:
            evalImageData[(count-1)//2] = arr
            evalImageLabels[(count-1)//2] = labels.index(item)
        count += 1

# for item in evalImageData:
#     print(item)
#     temp = pygame.pixelcopy.make_surface(item)
#     screen.blit(temp, (0, 0))
#     pygame.display.flip()
#     # time.sleep(1)
#     break
# pygame.display.quit()



np.save("Training_Data.npy", trainImageData)
np.save("Training_Labels.npy", trainImageLabels)
np.save("Eval_Data.npy", evalImageData)
np.save("Eval_Labels.npy", evalImageLabels)