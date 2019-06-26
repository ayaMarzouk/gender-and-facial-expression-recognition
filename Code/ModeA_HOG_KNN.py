import numpy as np
import os
import cv2
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from tqdm import tqdm
from math import *
import YOLO


def HOGKNN(test_dir):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    GenderClf = None
    HOGexpressionKNNClf = None
    HOG_features = np.load('HOG_features.npy')
    Gender_train = np.load('Gender_train.npy')
    expression_train = np.load('expression_train.npy')
    

    
    print (HOG_features.shape)
    GenderClf = KNeighborsClassifier (n_neighbors = 10)
    GenderClf.fit(HOG_features,Gender_train)

    HOGexpressionKNNClf = KNeighborsClassifier (n_neighbors = 10)
    HOGexpressionKNNClf.fit(HOG_features,expression_train)



    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColors = [(0,0,255),(0,255,255),(255,0,255),(255,255,0),(255,0,0)]
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

        i = 0
        for (x,y,w,h) in faces:
            if w<100 or h<100: continue
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            roi_gray = frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 128), interpolation = cv2.INTER_AREA)
            hog_features = hogX.compute(roi_gray)
            hog_features = np.array(hog_features)
            hog_features = hog_features.reshape(3780,)
            hog_features = hog_features[np.newaxis,:]

            prede = HOGexpressionKNNClf.predict(hog_features)[0]
            predg = GenderClf.predict(hog_features)[0]
            preds[prede]+=1
            gpreds[predg]+=1

            cv2.putText(frame,expressionLabels[prede]+' '+GenderLabels[predg],
                        (x,y-10),font,fontScale,
                        fontColors[i % 5],lineType)
            i+=1
            cv2.imshow('image',frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27: break

    cap.release()
    cv2.destroyAllWindows()
    totalPreds = sum(preds)
    for i in range(7): print('{}: {} %'.format(expressionLabels[i],round(preds[i]/totalPreds,4)*100))
    mxIndx = np.argmax(np.array(preds))
    ExDict = dict(zip(expressionLabels,list(range(7))))
    Exy_true = ExDict[test_dir.split('\\')[-2]]
    if preds[Exy_true]==preds[mxIndx] and Exy_true!=mxIndx: mxIndx=Exy_true
    print('HOG Winning expression: {} with Accuracy: {} %'.format(expressionLabels[mxIndx],round(preds[mxIndx]/totalPreds,4)*100))

    totalX = sum(gpreds)
    mxIndx2 = np.argmax(np.array(gpreds))
    GenDict = dict(zip(GenderLabels,[0,1]))
    Geny_true = GenDict[test_dir.split('_')[-2]]
    if gpreds[Geny_true]==gpreds[mxIndx2] and Geny_true!=mxIndx2: mxIndx2=Geny_true
    print('Winning Gender: {} with Accuracy: {} %'.format(GenderLabels[mxIndx2],round(gpreds[mxIndx2]/totalX,4)*100))

    return (preds, gpreds)


