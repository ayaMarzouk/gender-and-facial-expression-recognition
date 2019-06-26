import numpy as np

def combine_HOG_HOF(epredshog, gpredshog,epredshof, gpredshof):
    FinalExPreds = [x+y for x,y in zip(epredshog,epredshof)]
    FinalGenPreds = [x+y for x,y in zip(gpredshog,gpredshof)]


    expressionLabels = ['Neutral','Anger','Disgust','Fear','Happy','Sadness','Surprise']
    GenderLabels = ['Male','Female']

    totalPreds = sum(FinalExPreds)
    for i in range(7): print('{}: {} %'.format(expressionLabels[i],round(FinalExPreds[i]/totalPreds,4)*100))
    mxIndx = np.argmax(np.array(FinalExPreds))
    print('Combined HOG + HOF using SVM Winning expression: {} with Accuracy: {} %'.format(expressionLabels[mxIndx],round(FinalExPreds[mxIndx]/totalPreds,4)*100))
    
    totalX = sum(FinalGenPreds)
    mxIndx2 = np.argmax(np.array(FinalGenPreds))
    print('Combined Winning Gender: {} with Accuracy: {} %'.format(GenderLabels[mxIndx2],round(FinalGenPreds[mxIndx2]/totalX,4)*100))
