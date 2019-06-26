import numpy as np
import os
import cv2
from joblib import dump, load
from sklearn.svm import SVC
from random import shuffle
from tqdm import tqdm
from math import *
import YOLO


def createHistogram(mag, ang):
    intervals=[[0,20],[20,40],[40,60],[60,80],[80,100],[100,120],[120,140],[140,160],[160,180]]
    bins = [0]*9
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            currAng = ang[i][j]
            currmag = mag[i][j]
            for a,b in intervals:
                if currAng==a or currAng==180:
                    if currAng==180: currAng = 0
                    bins[int(currAng)//20]+=currmag
                    break
                if currAng>a and currAng <b:
                    maxi = b-currAng
                    mini = currAng-a
                    bins[a//20]+=((mini/20)*currmag)
                    if a==160: b = 0
                    bins[b//20]+=((maxi/20)*currmag)
                    break
                       

    return bins


def HistogramOfOpticalFlow(img1,img2):
    totalHist = []
    for i in range (0,img1.shape[0],8):
        rowHist = []
        for j in range (0,img2.shape[1],8):
            cell1= img1[i:i+8, j:j+8]
            cell2= img2[i:i+8, j:j+8]
            flow = cv2.calcOpticalFlowFarneback(cell1,cell2, None, 0.5, 3, 3, 5, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            ang = ang * 180 / np.pi / 2
            hist = createHistogram(mag, ang)
            rowHist.append(hist)
        totalHist.append(rowHist)

    finalVector = []
    for i in range(15):
        for j in range(7):
            appendHists = totalHist[i][j] + totalHist[i][j+1] + totalHist[i+1][j] + totalHist[i+1][j+1]
            appendHists = cv2.normalize(np.array(appendHists,np.float),None,norm_type=cv2.NORM_L2)
            finalVector = finalVector + list(appendHists)

    return finalVector


def HOFSVM(test_dir):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    GenderClf = load('GenderSVM.joblib')
    HOFexpressionClf = load('HOFexpressionSVMClf.joblib')

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0,0,255)
    lineType = 4

    cap = cv2.VideoCapture(test_dir)
    frameRate = cap.get(5)
    frameId = cap.get(1)
    _, frame1 = cap.read()
    faces1 = face_cascade.detectMultiScale(frame1, 1.3,5)
    Areas = [w*h for _,_,w,h in faces1]
    maxIndx = np.argmax(np.array(Areas))
    (x,y,w,h) = faces1[maxIndx]
    prevFrame = frame1[y:y+h, x:x+w]
    prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    prevFrame = cv2.resize(prevFrame, (64, 128), interpolation = cv2.INTER_AREA)

    expressionLabels = ['Neutral','Anger','Disgust','Fear','Happy','Sadness','Surprise']
    GenderLabels = ['Male','Female']
    preds, acc, gpreds = [0]*7, 0, [0,0]
    hogX = cv2.HOGDescriptor()
    hogfeatures = hogX.compute(prevFrame)
    hogfeatures = hogfeatures.reshape((3780,))[np.newaxis,:]
    predg = GenderClf.predict(hogfeatures)[0]
    gpreds[predg]+=1
    take2 = True
    while 1:
        frameId = cap.get(1)
        ret, frame2 = cap.read()
        if not ret: break
        if frameId%floor(frameRate): continue
        if take2:
            take2 = False
            continue
        else: take2 = True
        faces2 = face_cascade.detectMultiScale(frame2, 1.3,5)
        
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),4)
        Areas = [w*h for _,_,w,h in faces2]
        maxIndx = np.argmax(np.array(Areas))
        (x,y,w,h) = faces2[maxIndx]
        NextFrame = frame2[y:y+h, x:x+w]
        NextFrame = cv2.cvtColor(NextFrame, cv2.COLOR_BGR2GRAY)
        NextFrame = cv2.resize(NextFrame, (64, 128), interpolation = cv2.INTER_AREA)


        featureVector = np.array(HistogramOfOpticalFlow(prevFrame,NextFrame))
        for i in range(len(featureVector)):
            if isnan(featureVector[i]): featureVector[i] = 0
        featureVector = featureVector.reshape((3780,))[np.newaxis,:]

        hogfeatures = hogX.compute(NextFrame)
        hogfeatures = hogfeatures.reshape((3780,))[np.newaxis,:]
        predg = GenderClf.predict(hogfeatures)[0]
        prede = HOFexpressionClf.predict(featureVector)[0]
        preds[prede]+=1
        gpreds[predg]+=1

        cv2.putText(frame2,expressionLabels[prede]+' '+GenderLabels[predg],
                            (x,y-10),font,fontScale,
                            fontColor,lineType)
        cv2.imshow('image',frame2)
        k = cv2.waitKey(30) & 0xff
        if k == 27: break
        #if y_pred[signlePred]==y_test[truey]: acc+=1
        prevFrame = NextFrame

    cap.release()
    cv2.destroyAllWindows()
    totalPreds = sum(preds)
    for i in range(7): print('{}: {} %'.format(expressionLabels[i],round(preds[i]/totalPreds,4)*100))
    mxIndx = np.argmax(np.array(preds))
    ExDict = dict(zip(expressionLabels,list(range(7))))
    Exy_true = ExDict[test_dir.split('\\')[-2]]
    if preds[Exy_true]==preds[mxIndx] and Exy_true!=mxIndx: mxIndx=Exy_true
    print('HOF Winning expression: {} with Accuracy: {} %'.format(expressionLabels[mxIndx],round(preds[mxIndx]/totalPreds,4)*100))
    
    totalX = sum(gpreds)
    mxIndx2 = np.argmax(np.array(gpreds))
    GenDict = dict(zip(GenderLabels,[0,1]))
    Geny_true = GenDict[test_dir.split('_')[-2]]
    if gpreds[Geny_true]==gpreds[mxIndx2] and Geny_true!=mxIndx2: mxIndx2=Geny_true
    print('Winning Gender: {} with Accuracy: {} %'.format(GenderLabels[mxIndx2],round(gpreds[mxIndx2]/totalX,4)*100))

    return (preds, gpreds)
