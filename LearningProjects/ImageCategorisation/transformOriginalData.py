import os, pygame
import numpy as np
pygame.init()
width, height = (500, 500)
screen = pygame.display.set_mode((width, height))
trainImageData = np.zeros((278, 500, 500, 3), np.int8) # There are 555 images
evalImageData = np.zeros((277, 500, 500, 3), np.int8)
trainImageLabels = np.zeros((278, 1), np.int8)
evalImageLabels = np.zeros((277, 1), np.int8)
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
        if count % 2 == 0:
            trainImageData[count//2] = values
            trainImageLabels[count//2] = item
        else:
            evalImageData[(count-1)//2] = values
            evalImageLabels[(count-1)//2] = item
        count += 1


np.save("Training_Data.npy", trainImageData)
np.save("Training_Labels.npy", trainImageLabels)
np.save("Eval_Data.npy", evalImageData)
np.save("Eval_Labels.npy", evalImageLabels)