import numpy as np
import os
import cv2
from joblib import dump, load
from sklearn.svm import SVC
from random import shuffle
from keras.models import load_model
from tqdm import tqdm
from math import *
import YOLO


def RunMultiFacesDetection(test_dir):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    GenderModel = load_model('GenderDetectionModel.model')
    HOGexpressionClf = load('HOGexpressionSVMClf.joblib')

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColors = [(0,0,255),(0,255,255),(255,0,255),(255,255,0),(255,0,0)]
    lineType = 2
    expressionLabels = ['Neutral','Anger','Disgust','Fear','Happy','Sadness','Surprise']
    GenderLabels = ['Male','Female']


    hogX = cv2.HOGDescriptor()
    cap = cv2.VideoCapture(test_dir)
    frameRate = cap.get(5)
    preds, gpreds = [0]*7, [0,0]
    t = 0
    while 1:
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret: break
        if frameId%floor(frameRate): continue
        faces = face_cascade.detectMultiScale(frame, 1.3,5)

        i = 0
        for (x,y,w,h) in faces:
            if w<100 or h<100: continue
            cv2.rectangle(frame,(x,y),(x+w,y+h),fontColors[i%5],4)
            roi_gray = frame[y:y+h, x:x+w]
            genderPic = roi_gray.copy()


            roi_gray = cv2.resize(roi_gray, (64, 128), interpolation = cv2.INTER_AREA)
            hog_features = hogX.compute(roi_gray)
            hog_features = np.array(hog_features)
            hog_features = hog_features.reshape(3780,)
            hog_features = hog_features[np.newaxis,:]
            prede = HOGexpressionClf.predict(hog_features)[0]

            genderPic = genderPic / 255.0
            genderPic = cv2.resize(genderPic, (96, 96), interpolation = cv2.INTER_AREA)
            genderPic = genderPic.reshape(-1,96,96,3)
            predg = np.argmax(GenderModel.predict(genderPic)[0])
            preds[prede]+=1
            gpreds[predg]+=1

            cv2.putText(frame,expressionLabels[prede]+' '+GenderLabels[predg],
                        (x,y-10),font,fontScale,
                        fontColors[0],lineType)
            i+=1
            
            #Path = os.path.join(r'C:\Users\GO\source\repos\ComputerVisionProject\ComputerVisionProject\VirtualFrames',str(t)+'.jpg')
            #cv2.imwrite(Path, frame)
            vframe = frame.copy()
            vframe = cv2.resize(vframe,(1280,720), interpolation=cv2.INTER_AREA)
            cv2.imshow('image',vframe)
            k = cv2.waitKey(30) & 0xff
            if k == 27: break
        Path = os.path.join(r'C:\Users\GO\source\repos\ComputerVisionProject\ComputerVisionProject\VirtualFrames',str(t)+'.jpg')
        cv2.imwrite(Path, frame)
        t+=1

    cap.release()
    cv2.destroyAllWindows()


