# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:55:00 2019

@author: Ogün Can KAYA
"""

#################### YENİ BİR YÜZ EKLEYEBİLMEYİ SAĞLAR ######################################

import cv2
import os
import numpy as np
import time


def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

def personelId(path):
    
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    ids=[]
    for imagePath in imagePaths:
        ids.append(os.path.split(imagePath)[-1].split(".")[1])
        print(os.path.split(imagePath)[-1])
    unique_id=unique(ids)
    if len(unique_id)==0:
        return 0
    return int(unique_id[-1])+1
    
def adjust_gamma(face_id,name,count,resim,gamma=2.5):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    count+=1
    img = cv2.imread("dataset/User." + str(face_id) +"." +str(name) +'.' + str(resim) + ".jpg",0)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    cv2.imwrite("dataset/User." + str(face_id) +"." +str(name) +'.' + str(count) + ".jpg",  cv2.LUT(img, table))
   
    return count

def newImage(face_id,name,count,resim):
    img = cv2.imread("dataset/User." + str(face_id) +"." +str(name) +'.' + str(resim) + ".jpg",0)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    count+=1
    cv2.imwrite("dataset/User." + str(face_id) +"." +str(name) +'.' + str(count) + ".jpg", cl1)
#    bright=cv2.addWeighted(img,2,np.zeros(img.shape,img.dtype),0,50)
#    count+=1
#    cv2.imwrite("dataset/User." + str(face_id) +"." +str(name) +'.' + str(count) + ".jpg", bright)
    equ = cv2.equalizeHist(img)
    count+=1
    cv2.imwrite("dataset/User." + str(face_id) +"." +str(name) +'.' + str(count) + ".jpg", equ)
    count=adjust_gamma(face_id,name,count,resim)
    return count

def main(): 

    start = time.time()
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # For each person, enter one numeric face id
    path="dataset\\"
    face_id=personelId(path)
    #face_id = input('\n user id giriniz:   ')
    name= input("\nLütfen isminizi giriniz: ")
    print("\n [BILGI] Hazırlanıyor. Kameraya Bir Süre Bakın ve Bekleyiniz ...")
    # Initialize individual sampling face count
    count = 0
    resim=0
    while(True):
    
        ret, img = cam.read()
        img = cv2.flip(img, 1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            resim=count
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) +"." +str(name) +'.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
            count=newImage(face_id,name,count,resim) 
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 200: 
            
             break

    # Do a bit of cleanup
    print("\n [BILGI] Programdan Çıkılıyor")
    cam.release()
    cv2.destroyAllWindows()
    end = time.time()
    print(end - start)

if __name__=="__main__":
    main()
