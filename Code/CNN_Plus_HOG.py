import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import time
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.applications import VGG16
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from tqdm import tqdm
from math import *
import ModeA_HOG_SVM


def combine_CNN_HOG(epredscnn, gpredscnn, TestPath):
    epredshog, gpredshog = ModeA_HOG_SVM.HOGSVM(TestPath)
    FinalExPreds = [x+y for x,y in zip(epredscnn,epredshog)]
    FinalGenPreds = [x+y for x,y in zip(gpredscnn,gpredshog)]


    expressionLabels = ['Neutral','Anger','Disgust','Fear','Happy','Sadness','Surprise']
    GenderLabels = ['Male','Female']

    totalPreds = sum(FinalExPreds)
    for i in range(7): print('{}: {} %'.format(expressionLabels[i],round(FinalExPreds[i]/totalPreds,4)*100))
    mxIndx = np.argmax(np.array(FinalExPreds))
    print('Combined CNN + HOG Winning expression: {} with Accuracy: {} %'.format(expressionLabels[mxIndx],round(FinalExPreds[mxIndx]/totalPreds,4)*100))
    
    totalX = sum(FinalGenPreds)
    mxIndx2 = np.argmax(np.array(FinalGenPreds))
    print('Combined Winning Gender: {} with Accuracy: {} %'.format(GenderLabels[mxIndx2],round(FinalGenPreds[mxIndx2]/totalX,4)*100))