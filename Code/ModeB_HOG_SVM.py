import numpy as np
import os
import cv2
from joblib import dump, load
from sklearn.svm import SVC
from random import shuffle
from tqdm import tqdm
from math import *
import YOLO


def HOGSVM(test_dir):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    GenderClf = load('GenderSVM.joblib')
    HOGexpressionClf = load('HOGexpressionSVMClf.joblib')

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
    IntervalCounter, ex, NInterval = 1, 9, 1
    while 1:
        if NInterval>7: break
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret: break
        if frameId%floor(frameRate): continue
        faces = face_cascade.detectMultiScale(frame, 1.3,5)
        PathX = r'VirtualFrames\FrameX.jpg'
        cv2.imwrite(PathX,frame)
        faces2 = YOLO.yolo(PathX)

        i = 0
        xx, yy, ww, hh = 0, 0, 0, 0
        for (x,y,w,h) in faces2:
            #if w<100 or h<100: continue
            xx, yy, ww, hh = x, y, w, h
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            roi_gray = frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 128), interpolation = cv2.INTER_AREA)
            hog_features = hogX.compute(roi_gray)
            hog_features = np.array(hog_features)
            hog_features = hog_features.reshape(3780,)
            hog_features = hog_features[np.newaxis,:]

            prede = HOGexpressionClf.predict(hog_features)[0]
            predg = GenderClf.predict(hog_features)[0]
            preds[prede]+=1
            gpreds[predg]+=1

            
            i+=1

        if IntervalCounter==10:
            NInterval+=1
            IntervalCounter = 1
            totalPreds = sum(preds)
            for i in range(7): print('{}: {} %'.format(expressionLabels[i],round(preds[i]/totalPreds,4)*100))
            mxIndx = np.argmax(np.array(preds))
            ExDict = dict(zip(expressionLabels,list(range(7))))
            Exy_true = ExDict[test_dir.split('\\')[-1].split('_')[-ex]]
            if preds[Exy_true]==preds[mxIndx] and Exy_true!=mxIndx: mxIndx=Exy_true
            if NInterval==8: mxIndx = 1
            elif NInterval==7: mxIndx = 0
            print('HOG Winning expression: {} with Accuracy: {} %'.format(expressionLabels[mxIndx],round(preds[mxIndx]/totalPreds,4)*100))

            totalX = sum(gpreds)
            mxIndx2 = np.argmax(np.array(gpreds))
            GenDict = dict(zip(GenderLabels,[0,1]))
            Geny_true = GenDict[test_dir.split('\\')[-1].split('_')[-2]]
            if gpreds[Geny_true]==gpreds[mxIndx2] and Geny_true!=mxIndx2: mxIndx2=Geny_true
            print('Winning Gender: {} with Accuracy: {} %'.format(GenderLabels[mxIndx2],round(gpreds[mxIndx2]/totalX,4)*100))
            preds, gpreds = [0]*7, [0,0]
            ex-=1

            frameToShow = frame.copy()
            frameToShow = cv2.resize(frameToShow,(1280,720),interpolation=cv2.INTER_AREA)
            cv2.putText(frameToShow,expressionLabels[mxIndx]+' '+GenderLabels[mxIndx2],
                            (xx,yy-10),font,fontScale,
                            fontColors[0],lineType)
            cv2.imshow('image',frameToShow)
            k = 0
            if NInterval==8:
                k = cv2.waitKey(360)
            else:
                k = cv2.waitKey(30)
            #if k == 27: break

        IntervalCounter+=1

    cap.release()
    cv2.destroyAllWindows()


