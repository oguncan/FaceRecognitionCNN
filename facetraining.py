# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:55:00 2019

@author: Ogün Can KAYA
"""

#################### MEVCUT RESİMLERİ EĞİTMEMİZİ SAĞLAR ######################################

import cv2
import numpy as np
from PIL import Image
import os
import time
from sklearn import svm


detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    names=[]

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        names = (os.path.split(imagePath)[-1].split(".")[2])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids,names

def main():
    # Path for face image database
    path = 'dataset\\'
    start = time.time()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    
    print ("\n [BILGI] Yüzler eğitiliyor. Biraz Zaman Alabilir. Lütfen bekleyiniz...")
    faces,ids,names = getImagesAndLabels(path)
#    model = svm.LinearSVC()
#    model.fit(faces, np.array(ids))
#    recognizer.train(faces, np.array(ids))
    
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
    
    # Print the numer of faces trained and end program
    print("\n [BILGI] {0} yüz eğitildi. Programdan çıkılıyor.".format(len(np.unique(ids))))
    end = time.time()
    print(end - start)

if __name__=="__main__":
    main()