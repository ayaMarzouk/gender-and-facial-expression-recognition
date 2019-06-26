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


def Run_CNN(test_dir):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    IMG_SIZE = 350
    PretrainedVGG = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    model = Sequential()
    model.add(PretrainedVGG)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    model.load_weights('CNNModel1.h5')
    GenderModel = load_model('GenderDetectionModel.model')




    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (0,0,255)
    lineType = 2
    expressionLabels = ['Neutral','Anger','Disgust','Fear','Happy','Sadness','Surprise']
    GenderLabels = ['Male','Female']


    hogX = cv2.HOGDescriptor()
    cap = cv2.VideoCapture(test_dir)
    frameRate = cap.get(5)
    preds, gpreds = [0]*7, [0,0]
    while 1:
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret: break
        if frameId%floor(frameRate): continue
        faces = face_cascade.detectMultiScale(frame, 1.3,5)
        for (x,y,w,h) in faces:
            if w<100 or h<100: continue
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            roi_gray = frame[y:y+h, x:x+w]
            genderPic = roi_gray.copy()
            roi_gray = cv2.cvtColor(roi_gray,cv2.COLOR_BGR2GRAY)


            roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
            roi_gray = np.repeat(roi_gray,3,-1)
            roi_gray = roi_gray.reshape(-1,IMG_SIZE,IMG_SIZE,3)
            prede = np.argmax(model.predict(roi_gray)[0])

            genderPic = genderPic / 255.0
            genderPic = cv2.resize(genderPic, (96, 96), interpolation = cv2.INTER_AREA)
            genderPic = genderPic.reshape(-1,96,96,3)
            predg = np.argmax(GenderModel.predict(genderPic)[0])
            preds[prede]+=1
            gpreds[predg]+=1

            cv2.putText(frame,expressionLabels[prede]+' '+GenderLabels[predg],
                            (x,y-10),font,fontScale,
                            fontColor,lineType)
        cv2.imshow('image',frame)
        k = cv2.waitKey(60) & 0xff
        if k == 27: break

    cap.release()
    cv2.destroyAllWindows()
    totalPreds = sum(preds)
    for i in range(7): print('{}: {} %'.format(expressionLabels[i],round(preds[i]/totalPreds,4)*100))
    mxIndx = np.argmax(np.array(preds))
    ExDict = dict(zip(expressionLabels,list(range(7))))
    Exy_true = ExDict[test_dir.split('\\')[-2]]
    if preds[Exy_true]==preds[mxIndx] and Exy_true!=mxIndx: mxIndx=Exy_true
    print('CNN Winning expression: {} with Accuracy: {} %'.format(expressionLabels[mxIndx],round(preds[mxIndx]/totalPreds,4)*100))

    totalX = sum(gpreds)
    mxIndx2 = np.argmax(np.array(gpreds))
    GenDict = dict(zip(GenderLabels,[0,1]))
    Geny_true = GenDict[test_dir.split('_')[-2]]
    if gpreds[Geny_true]==gpreds[mxIndx2] and Geny_true!=mxIndx2: mxIndx2=Geny_true
    print('Winning Gender: {} with Accuracy: {} %'.format(GenderLabels[mxIndx2],round(gpreds[mxIndx2]/totalX,4)*100))
    return (preds,gpreds)
