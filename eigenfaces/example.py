import sys
import numpy as np
from eigen import read_images
from model import EigenfacesModel

imageFile = '/Users/Abhi/Desktop/FacialRecognition/data/att_faces/'

if __name__ == '__main__':
    # read images
    [X, y] = read_images(imageFile)

    # compute the eigenfaces model
    trainingX, trainingY, testX, testY = [], [], [], []

    # Seperate training and test data
    start, end = 0, 5
    while end <= 400:
        # training/test X
        trainingX += X[start:end]
        testX += X[end:end + 5]

        # training/test y
        trainingY += y[start:end]
        testY += y[end:end + 5]

        # Increment the values
        start += 10
        end += 10

    # model the eigenfaces
    model = EigenfacesModel(trainingX, trainingY)

    # Get Prediction
    numberCorrect, numberToCheck = 0, 0
    for i in range(0, len(testX)):
        prediction = model.predict(testX[i])
        if testY[i] == prediction:
            numberCorrect += 1

    # Print the accuracy
    print 'Number Correct:', numberCorrect, 'out of:', len(testX)
