import sys
import numpy as np
from eigen import read_images
from model import EigenfacesModel

if __name__ == '__main__':
    # read images
    [X, y] = read_images('/Users/Abhi/Desktop/FacialRecognition/data/att_faces/')

    # compute the eigenfaces model
    S = []
    M = []
    A = []
    B = []

    # Seperate training and test
    start = 0
    end = 5
    while end <= 400:
        # X
        S = S + X[start:end]
        A = A + X[end:end + 5]
        # Y
        M = M + y[start:end]
        B = B + y[end:end + 5]
        # Increment the values
        start = start + 10
        end = end + 10

    # model the eigenfaces
    model = EigenfacesModel(S, M)
    numberCorrect = 0
    numberToCheck = 0
    start = 5
    end = 10
    # Get Prediction
    while end <= 400:
        for i in range(start, end):
            print start, end
            numberToCheck = numberToCheck + 1
            prediction = model.predict(X[i])
            # print "expected =", y[i], "/", "predicted =", prediction
            if y[i] == prediction:
                numberCorrect = numberCorrect + 1
        start = start + 10
        end = end + 10

    print 'Number Correct:', numberCorrect, 'out of:', numberToCheck
