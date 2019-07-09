# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:55:00 2019

@author: Ogün Can KAYA
"""


from PyQt5.QtWidgets import QApplication, QFrame,QPushButton, QHBoxLayout, QGroupBox, QVBoxLayout,QLabel,QDialog,QMessageBox
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import QRect,QTimer
from PyQt5 import QtCore
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
import cv2
import facedataset
import facerecognition
import facetraining
import numpy as np
import os
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import time
import json
from PyQt5.uic import loadUi
# %%
camType=0
class Second(QDialog):
    
    def __init__(self, parent=None):
        super(Second, self).__init__(parent)
        loadUi('addImage.ui',self)
        
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.FRGraph = FaceRecGraph();
        self.MTCNNGraph = FaceRecGraph();
        self.aligner = AlignCustom();
        self.extract_feature = FaceFeature(self.FRGraph)
        self.face_detect = MTCNNDetect(self.MTCNNGraph, scale_factor=2);
        self.person_imgs = {"Left" : [], "Right": [], "Center": []};
        self.person_features = {"Left" : [], "Right": [], "Center": []};
        self.init_ui()
        self.count=0
   
    def init_ui(self):
        self.title = "Yüz Ekleme"
        self.top = 200
        self.left = 650
        self.width = 640
        self.height = 640
#        self.imageData=QLabel(self)
#        pixmap=QPixmap("face.png")
#        self.imageData.setPixmap(pixmap)
#        self.imageData.setAlignment(QtCore.Qt.AlignCenter)
#        self.vbox.addWidget(self.labelImage)
        imageData=cv2.imread("face.png",2)
        qformat=QImage.Format_Indexed8
        
#        qformat=QImage.Format_RGB888
        outImage=QImage(imageData,imageData.shape[1],imageData.shape[0],imageData.strides[0],qformat)
        outImage=outImage.rgbSwapped()
        self.labelImage.setPixmap(QPixmap.fromImage(outImage))
        self.labelImage.setScaledContents(True)

#        self.face_id=self.personelId(self.path)
        self.setFixedSize(self.width,self.height)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle(self.title)
        self.image=None
#        self.imageData.hide()
        
        self.addImage.clicked.connect(self.clickMethod)
    
    def clickMethod(self):
        self.capture=cv2.VideoCapture(camType)
#        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
#        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
        
        
        ret,self.image=self.capture.read()
        self.image=cv2.flip(self.image,1)
#        detect_image=self.detect_face(self.image)
        self.displayImage(self.image,1)

        detect_name=self.detect_person(self.image)
        
        if detect_name=="Unknown":
           
            self.timer=QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(5)
        else:
            buttonReply = QMessageBox.question(self, 'Uyarı', "Yüz Kayıtlı!", QMessageBox.Cancel)
            if buttonReply == QMessageBox.Cancel:
                self.close()
                self.destroy()
    
    
    
    def unique(self,list1): 
        unique_list = [] 
        for x in list1: 
            if x not in unique_list: 
                unique_list.append(x)
        
        return unique_list
    
    
    def personelId(self,path):
    
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        ids=[]
        for imagePath in imagePaths:
            ids.append(os.path.split(imagePath)[-1].split(".")[1])
#            print(os.path.split(imagePath)[-1])
        unique_id=self.unique(ids)
        if len(unique_id)==0:
            return 0
        return int(unique_id[-1])+1
    
    def adjust_gamma(self,face_id,name,count,resim,gamma=2.5):
    	# build a lookup table mapping the pixel values [0, 255] to
    	# their adjusted gamma values
        self.count=count
        self.count+=1
        
        img = cv2.imread("dataset/User." + str(self.face_id) +"." +str(name) +'.' + str(resim) + ".jpg",0)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    	# apply gamma correction using the lookup table
        cv2.imwrite("dataset/User." + str(self.face_id) +"." +str(name) +'.' + str(self.count) + ".jpg",  cv2.LUT(img, table))
        return self.count

    def newImage(self,face_id,name,count,resim):
        self.count=count
        img = cv2.imread("dataset/User." + str(self.face_id) +"." +str(name) +'.' + str(resim) + ".jpg",0)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        self.count+=1
        cv2.imwrite("dataset/User." + str(self.face_id) +"." +str(name) +'.' + str(self.count) + ".jpg", cl1)
        #    bright=cv2.addWeighted(img,2,np.zeros(img.shape,img.dtype),0,50)
        #    count+=1
        #    cv2.imwrite("dataset/User." + str(face_id) +"." +str(name) +'.' + str(count) + ".jpg", bright)
        equ = cv2.equalizeHist(img)
        self.count+=1
        cv2.imwrite("dataset/User." + str(self.face_id) +"." +str(name) +'.' + str(self.count) + ".jpg", equ)
        self.count=self.adjust_gamma(face_id,name,self.count,resim)
        print(count)
        return self.count
    def closeEvent(self, event):
        name=self.lineEdit.text()
        f = open('./facerec_128D.txt','r');
        data_set = json.loads(f.read());
        for pos in self.person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
            self.person_features[pos] = [np.mean(self.extract_feature.get_features(self.person_imgs[pos]),axis=0).tolist()]
#            print(person_features)
        data_set[name] = self.person_features;
        f = open('./facerec_128D.txt', 'w');
        f.write(json.dumps(data_set))
        self.close()
        self.window = Window()
#        self.window.doTraining()
        self.window.show()
    def detect_face(self,img):
        
        name=self.lineEdit.text()        
        f = open('./facerec_128D.txt','r');
        data_set = json.loads(f.read());
#        person_imgs = {"Left" : [], "Right": [], "Center": []};
#        person_features = {"Left" : [], "Right": [], "Center": []};
        rects, landmarks = self.face_detect.detect_face(img, 80);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            self.count+=1
            aligned_frame, pos = self.aligner.align(160,img,landmarks[:,i]);
            print(aligned_frame)
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                self.person_imgs[pos].append(aligned_frame) 
                self.displayImage(aligned_frame,1)
                
                if(self.count>=80):
                    self.count=0
                    self.timer.stop()
                    self.close()
        
        
        
        
        return img

    def detect_person(self,img):
        person_name=""
#        self.timer.stop()
        rects, landmarks = self.face_detect.detect_face(img,80);#min face size is set to 80x80
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = self.aligner.align(160,img,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0):
            features_arr = self.extract_feature.get_features(aligns)
            person_name = self.findPeople(features_arr,positions)
        return person_name
    
    def findPeople(self,features_arr, positions, thres = 0.6, percent_thres = 70):
        f = open('./facerec_128D.txt','r')
        data_set = json.loads(f.read());
        returnRes = ""
        for (i,features_128D) in enumerate(features_arr):
            result = "Unknown";
            smallest = sys.maxsize
            for person in data_set.keys():
                person_data = data_set[person][positions[i]];
                for data in person_data:
                    distance = np.sqrt(np.sum(np.square(data-features_128D)))
                    if(distance < smallest):
                        smallest = distance;
                        result = person;
            percentage =  min(100, 100 * thres / smallest)
            if percentage <= percent_thres :
                result = "Unknown"
            returnRes=result
        return returnRes
    

    def update_frame(self):
        ret,self.image=self.capture.read()
        self.image=cv2.flip(self.image,1)
        detect_image=self.detect_face(self.image)
        self.displayImage(detect_image,1)
    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage=outImage.rgbSwapped()
        
        if window==1:
            self.labelImage.setPixmap(QPixmap.fromImage(outImage))
            self.labelImage.setScaledContents(True)
        if window==2:
            self.processedLabel.setPixmap=(QPixmap.fromImage(outImage))
            self.processedImage.setScaledContents(True)
        
class Window(QtWidgets.QWidget):
    
    def __init__(self):
        super(Window,self).__init__()
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
        self.FRGraph = FaceRecGraph();
        self.MTCNNGraph = FaceRecGraph();
        self.aligner = AlignCustom();
        self.extract_feature = FaceFeature(self.FRGraph)
        self.face_detect = MTCNNDetect(self.MTCNNGraph, scale_factor=2); #scale_factor, rescales image for faster detection
#        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.InitWindow()
    
    def InitWindow(self):
        
        self.title = "Yüz Tanıma"
        self.top = 200
        self.left = 650
        self.width = 640
        self.height = 640
        self.setFixedSize(self.width,self.height)
        self.image=None

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.timer=QTimer(self)
        self.run_button = QtWidgets.QPushButton('Yüzü Bul')
        self.addImage = QtWidgets.QPushButton('Veri Ekle')
        self.doTrain = QtWidgets.QPushButton('Train Et')

        self.run_button.clicked.connect(self.findImage)
        self.addImage.clicked.connect(self.imageAdd)
        
        self.doTrain.clicked.connect(self.doTraining)
        self.vbox = QVBoxLayout()
        
        
        print(camType)
#        first=FirstScreen()
#        print(first.camType)
#        first=FirstScreen()
#        print(first.buttonClick.camType)
        
        
        self.imageBox = QLabel(self)
        self.imageBox.resize(460, 330)

        
        self.vbox.addWidget(self.imageBox)
        self.vbox.addWidget(self.run_button)
        self.vbox.addWidget(self.addImage)
        self.vbox.addWidget(self.doTrain)

        self.setLayout(self.vbox)
        self.timer.stop()
        
    def unique(self,list1): 
        unique_list = [] 
        for x in list1: 
            if x not in unique_list: 
                unique_list.append(x) 
        return unique_list
    def closeEvent(self, event):
#        reply = QMessageBox.question(self, 'Quit', 'Are You Sure to Quit?', QMessageBox.No | QMessageBox.Yes)
#        if reply == QMessageBox.Yes:
#            event.accept()
#            self.close()
#        else:
#            event.ignore()
        
        self.close()
        self.firstScreen=FirstScreen()
        self.firstScreen.show()
        
    def faceNames(self,path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        name=[]
        ids=[]
        for imagePath in (imagePaths):
            name.append(os.path.split(imagePath)[-1].split(".")[2])
            ids.append(os.path.split(imagePath)[-1].split(".")[1])
        unique_ids=self.unique(ids)
        return name,unique_ids
    
    def getName(self,searchId,path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in (imagePaths):
            if(searchId==os.path.split(imagePath)[-1].split(".")[1]):
                return os.path.split(imagePath)[-1].split(".")[2]
    
    def update_frame(self):
        
        ret,self.image=self.capture.read()
        self.image=cv2.flip(self.image,1)
        detect_image=self.detect_face(self.image)
        self.displayImage(detect_image,1)
    
    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage=outImage.rgbSwapped()
        if window==1:
           
            self.imageBox.setPixmap(QPixmap.fromImage(outImage))
            self.imageBox.setScaledContents(True)
        if window==2:
            self.processedLabel.setPixmap=(QPixmap.fromImage(outImage))
            self.processedImage.setScaledContents(True)
    
    def getImagesAndLabels(self,path):
        
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        names=[]
    
        for imagePath in imagePaths:
    
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
    
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            names = (os.path.split(imagePath)[-1].split(".")[2])
            faces = self.detector.detectMultiScale(img_numpy)
    
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids,names
    
    
    def findImage(self):
#        self.timer.stop()
        
        
        self.capture=cv2.VideoCapture(camType)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
#        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
    
    
    def imageAdd(self):
#        self.run_button.clicked.disconnect(self.record_video.start_recording)
        self.timer.stop()
#        self.close()
        self.setVisible(False)
        self.firstScreen=FirstScreen()
#        self.firstScreen.close()
        self.firstScreen.setVisible(False)
        self.SW = Second()
        self.SW.show()
        
        
    
    def detect_face(self,img):
#        self.timer.stop()
        rects, landmarks = self.face_detect.detect_face(img,80);#min face size is set to 80x80
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = self.aligner.align(160,img,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0):
            features_arr = self.extract_feature.get_features(aligns)
            recog_data = self.findPeople(features_arr,positions)
            for (i,rect) in enumerate(rects):
                cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #draw bounding box for the face
                cv2.putText(img,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        return img
    
    def findPeople(self,features_arr, positions, thres = 0.6, percent_thres = 70):
        f = open('./facerec_128D.txt','r')
        data_set = json.loads(f.read());
        returnRes = [];
        for (i,features_128D) in enumerate(features_arr):
            result = "Unknown";
            smallest = sys.maxsize
            for person in data_set.keys():
                person_data = data_set[person][positions[i]];
                for data in person_data:
                    distance = np.sqrt(np.sum(np.square(data-features_128D)))
                    if(distance < smallest):
                        smallest = distance;
                        result = person;
            percentage =  min(100, 100 * thres / smallest)
            if percentage <= percent_thres :
                result = "Unknown"
            returnRes.append((result,percentage))
        return returnRes
    def doTraining(self):
        self.timer.stop()
        path = 'dataset\\'
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print ("\n [BILGI] Yüzler eğitiliyor. Biraz Zaman Alabilir. Lütfen bekleyiniz...")
        faces,ids,names = self.getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        
        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
        
        # Print the numer of faces trained and end program
        print("\n [BILGI] {0} yüz eğitildi..".format(len(np.unique(ids))))
    
class FirstScreen(QDialog):
    
    def __init__(self, parent=None):
        super(FirstScreen, self).__init__(parent)
        loadUi('firstScreen.ui',self)
        self.init_ui()
        self.count=0

        
    def init_ui(self):
        self.title = "Hoşgeldiniz"
        self.top = 200
        self.left = 650
        self.width = 640
        self.height = 640
        self.setFixedSize(self.width,self.height)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.setWindowTitle(self.title)
        self.nextButton.clicked.connect(self.buttonClick)
        self.rbnLaptop.clicked.connect(self.rbnLaptop_click)
        self.rbnUsb.clicked.connect(self.rbnUsb_click)
        self.rbnHls.clicked.connect(self.rbnHls_click)
        self.rbnRtsp.clicked.connect(self.rbnRtsp_click)
        self.rbnRtmp.clicked.connect(self.rbnRtmp_click)
        self.lineIP.setDisabled(1)
        self.lineUser.setDisabled(1)
        self.linePassword.setDisabled(1)
        self.lblIP.setDisabled(1)
        self.lblUsername.setDisabled(1)
        self.lblPassword.setDisabled(1)
    def rbnLaptop_click(self):
        self.lineIP.setDisabled(1)
        self.lineUser.setDisabled(1)
        self.linePassword.setDisabled(1)
        self.lblIP.setDisabled(1)
        self.lblUsername.setDisabled(1)
        self.lblPassword.setDisabled(1)
    def rbnUsb_click(self):
        self.lineIP.setDisabled(1)
        self.lineUser.setDisabled(1)
        self.linePassword.setDisabled(1)
        self.lblIP.setDisabled(1)
        self.lblUsername.setDisabled(1)
        self.lblPassword.setDisabled(1)
    def rbnHls_click(self):
        self.lineIP.setEnabled(1)
        self.lineUser.setDisabled(1)
        self.linePassword.setDisabled(1)
        self.lblIP.setEnabled(1)
        self.lblUsername.setDisabled(1)
        self.lblPassword.setDisabled(1)
    def rbnRtsp_click(self):
        self.lineIP.setEnabled(1)
        self.lineUser.setEnabled(1)
        self.linePassword.setEnabled(1)
        self.lblIP.setEnabled(1)
        self.lblUsername.setEnabled(1)
        self.lblPassword.setEnabled(1)
    def rbnRtmp_click(self):
        self.lineIP.setEnabled(1)
        self.lineUser.setEnabled(1)
        self.linePassword.setEnabled(1)
        self.lblIP.setEnabled(1)
        self.lblUsername.setEnabled(1)
        self.lblPassword.setEnabled(1)
    def closeEvent(self, event):
        self.close()
    def buttonClick(self):
        global camType
        if(self.rbnLaptop.isChecked()):
            #dizüstü
            camType=0
            self.window=Window()
            self.window.show()
            
        elif(self.rbnUsb.isChecked()):
            #USB
            camType=1
            self.window=Window()
            self.window.show()
        elif(self.rbnHls.isChecked()):
            #hls
            pass
        elif(self.rbnRstp.isChecked()):
            #rtsp
            pass
        elif(self.rbnRtmp.isChecked()):
            #rtmp
            pass
        else:
              buttonReply = QMessageBox.question(self, 'Uyarı', "Lütfen bir kamera seçiniz!", QMessageBox.Cancel ) 
        self.close()
#        self.show()	
if __name__ == "__main__":
    app = QApplication(sys.argv)
#    main_window = QtWidgets.QMainWindow()
    main_widget = FirstScreen()
#    main_window.setCentralWidget(main_widget)
    main_widget.show()
    sys.exit(app.exec_())

    