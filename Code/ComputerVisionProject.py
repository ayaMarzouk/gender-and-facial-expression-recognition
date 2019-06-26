import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
from os import system
import ModeA_HOG_SVM
import ModeA_HOF_SVM
import ModeA_Combined_HOG_HOF
import ModeA_HOG_KNN
import ModeA_HOF_KNN
import ModeB_HOG_SVM
import ModeB_HOF_SVM
import ModeB_HOG_KNN
import ModeB_HOF_KNN
import HOG_HOF_Features
import CNN_Model
import CNN_Plus_HOG
import DetectMultiFaces



def from_rgb_to_hexa(rgb):
    return "#%02x%02x%02x" % rgb


TestPath = None
def VideoBrowsing():
    global TestPath
    TestPath = askopenfilename(initialdir = r'C:\Users\GO\source\repos\ComputerVisionProject\ComputerVisionProject',title = "Select a video",filetypes = [('MP4','*.mp4'),('MOV','*.MOV')])
    TestPath = TestPath.replace('/','\\')


################################### SVM Classifier ################################

hogSVMpreds, hogSVMgpreds = None, None
def HOGSVM():
    global TestPath, hogSVMpreds, hogSVMgpreds
    _ = system('cls')
    ModeType = TestPath.split('\\')[7].split('_')[0]
    if ModeType=='ModeA':
        (hogSVMpreds, hogSVMgpreds) = ModeA_HOG_SVM.HOGSVM(TestPath)
    elif ModeType=='ModeB':
        ModeB_HOG_SVM.HOGSVM(TestPath)

hofSVMpreds, hofSVMgpreds = None, None
def HOFSVM():
    global TestPath, hofSVMpreds, hofSVMgpreds
    _ = system('cls')
    ModeType = TestPath.split('\\')[7].split('_')[0]
    if ModeType=='ModeA':
        (hofSVMpreds, hofSVMgpreds) = ModeA_HOF_SVM.HOFSVM(TestPath)
    elif ModeType=='ModeB':
        ModeB_HOF_SVM.HOFSVM(TestPath)


def HOG_HOF_SVM():
    global hogSVMpreds, hogSVMgpreds, hofSVMpreds, hofSVMgpreds
    _ = system('cls')
    ModeA_Combined_HOG_HOF.combine_HOG_HOF(hogSVMpreds, hogSVMgpreds, hofSVMpreds, hofSVMgpreds)


################################### KNN Classifier ################################

hogKNNpreds, hogKNNgpreds = None, None
def HOGKNN():
    global TestPath, hogKNNpreds, hogKNNgpreds
    _ = system('cls')
    ModeType = TestPath.split('\\')[7].split('_')[0]
    if ModeType=='ModeA':
        (hogKNNpreds, hogKNNgpreds) = ModeA_HOG_KNN.HOGKNN(TestPath)
    elif ModeType=='ModeB':
        ModeB_HOG_KNN.HOGKNN(TestPath)


hofSVMpreds, hofSVMgpreds = None, None
def HOFKNN():
    global TestPath, hofKNNpreds, hofKNNgpreds
    _ = system('cls')
    ModeType = TestPath.split('\\')[7].split('_')[0]
    if ModeType=='ModeA':
        (hofKNNpreds, hofKNNgpreds) = ModeA_HOF_KNN.HOFKNN(TestPath)
    elif ModeType=='ModeB':
        ModeB_HOF_KNN.HOFKNN(TestPath)


def HOG_HOF_KNN():
    global hogKNNpreds, hogKNNgpreds, hofKNNpreds, hofKNNgpreds, var1, var2
    _ = system('cls')
    ModeA_Combined_HOG_HOF.combine_HOG_HOF(hogKNNpreds, hogKNNgpreds, hofKNNpreds, hofKNNgpreds)


################################### Advanced Techniques ################################

def HOG_HOF_AllFeatures():
    global TestPath
    _ = system('cls')
    HOG_HOF_Features.HOG_HOF_Features(TestPath)


epredscnn, gpredscnn = None, None
def CNNModel():
    global TestPath, epredscnn, gpredscnn
    _ = system('cls')
    epredscnn, gpredscnn = CNN_Model.Run_CNN(TestPath)


def CNNPlusHOG():
    global TestPath, epredscnn, gpredscnn
    _ = system('cls')
    CNN_Plus_HOG.combine_CNN_HOG(epredscnn, gpredscnn, TestPath)


def MultiFacesDetection():
    global TestPath
    _ = system('cls')
    DetectMultiFaces.RunMultiFacesDetection(TestPath)


def AdvancedTechniques():
    global root
    subRoot = Toplevel(root)
    subRoot.title('Advanced Techniques')
    subRoot.geometry('1920x1080')
    image = PhotoImage(file='modern cool.png')
    c = Canvas(subRoot, width = 1920, height = 1080, bg='black')
    c.pack()
    c.create_image(0, 0, image=image, anchor=NW)
    c.create_rectangle(0 ,0, 282, 1080, fill=from_rgb_to_hexa((41, 44, 51)))

    HOGHOFFeaturesB = Button(subRoot, text='HOG + HOF Features', font=('DS-Digital',26), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=HOG_HOF_AllFeatures)
    HOGHOFFeaturesB.place(relx=0.002, rely=0.01, width=281 ,height=60)

    CNNB = Button(subRoot, text='Use CNN', font=('DS-Digital',38), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=CNNModel)
    CNNB.place(relx=0.002, rely=0.1, width=281 ,height=60)

    CNNPlusHOGB = Button(subRoot, text='CNN + HOG', font=('DS-Digital',38), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=CNNPlusHOG)
    CNNPlusHOGB.place(relx=0.002, rely=0.19, width=281 ,height=60)

    MultiFacesB = Button(subRoot, text='Detect Multi-Faces', font=('DS-Digital',24), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=MultiFacesDetection)
    MultiFacesB.place(relx=0.002, rely=0.28, width=281 ,height=60)

    tk.mainloop()


root = Tk()
root.title('Computer Vision Project')
root.geometry('1920x1080')


image = PhotoImage(file='pic.png')
c = Canvas(root, width = 1920, height = 1080, bg='black')
c.pack()
c.create_image(0, 0, image=image, anchor=NW)
c.create_rectangle(1920 ,0, 1070, 1080, fill=from_rgb_to_hexa((41, 44, 51)))


BrowseingB = Button(root, text='Browse Video', font=('DS-Digital',34), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=VideoBrowsing)
BrowseingB.place(relx=0.784, rely=0.01, width=282 ,height=60)
#BrowseingB.config(bd=6, highlightcolor=from_rgb_to_hexa((62, 120, 138)))

HOGSVMB = Button(root, text='HOG SVM', font=('DS-Digital',38), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=HOGSVM)
HOGSVMB.place(relx=0.784, rely=0.1, width=282, height=60)

HOFSVMB = Button(root, text='HOF SVM', font=('DS-Digital',38), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=HOFSVM)
HOFSVMB.place(relx=0.784, rely=0.19, width=282, height=60)

CombinedSVMB = Button(root, text='HOG + HOF SVM', font=('DS-Digital',36), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=HOG_HOF_SVM)
CombinedSVMB.place(relx=0.784, rely=0.28, width=282, height=60)

HOGKNNB = Button(root, text='HOG KNN', font=('DS-Digital',38), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=HOGKNN)
HOGKNNB.place(relx=0.784, rely=0.37, width=282, height=60)

HOFKNNB = Button(root, text='HOF KNN', font=('DS-Digital',38), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=HOFKNN)
HOFKNNB.place(relx=0.784, rely=0.46, width=282, height=60)

CombinedKNNB = Button(root, text='HOG + HOF KNN', font=('DS-Digital',36), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=HOG_HOF_KNN)
CombinedKNNB.place(relx=0.784, rely=0.55, width=282, height=60)

AdvancedTechB = Button(root, text='Advanced Techniques', font=('DS-Digital',22), fg=from_rgb_to_hexa((62, 120, 138)), bg=from_rgb_to_hexa((41, 44, 51)), command=AdvancedTechniques)
AdvancedTechB.place(relx=0.784, rely=0.64, width=282, height=60)

tk.mainloop()