# # -*- coding: utf-8 -*-
# """
# Created on Tue Dec 28 18:29:31 2021

# @author: Ishraq
# """

# for data manipulation
import numpy as np
import pandas as pd

# for image loading
from PIL import Image, ImageOps
import glob
import random
import os
from skimage import io

# # for creating visualizations
# import matplotlib.pyplot as plt
# import cv2
# # function to load the data sets
# def loadData(dataPath):
    
#     # load the training data details
#     trainXImg = []
#     #trainY = []
#     for file in glob.glob(dataPath):
         
#         img=cv2.imread(file)#, cv2. IMREAD_GRAYSCALE,cv2.IMREAD_COLOR
        
#         #img=img.reshape(-1)
#         img = cv2.resize(img, (256,256))
#         trainXImg.append(np.asarray(img))# save the image to the array
#         #trainY.append(0)
#         #trainY.append(1)
    
            
#     print("finished loading EOSINOPHILlTrain images")
#     return np.asarray(trainXImg)


# # get the folders of the images for training and testing sets
# EOSINOPHILlTrain = "C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/EOSINOPHIL/*.png"
# LYMPHOCYTE_Train = "C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/LYMPHOCYTE/*.jpeg"
# MONOCYTElTrain = "C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/MONOCYTE/*.jpeg"
# PNEU_Train = "C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/NEUTROPHIL/*.jpeg"
# #Normal_Test = "archive/chest_xray/test/NORMAL/*.jpeg"
# #PNEU_Test = "archive/chest_xray/test/PNEUMONIA/*.jpeg"
# # call the function
# EOSINOPHILlTrainXImg = loadData(EOSINOPHILlTrain)
# #PNEUtrainXImg = loadData(PNEU_Train)
# #NORMALTstImg=loadData(Normal_Test)
# #PNEUTstImg=loadData(PNEU_Test)
# #NORtrainXImg.resize()
# EOSINOPHILlTrainXImg= np.resize(EOSINOPHILlTrainXImg,(10,256,256,3))#
# #PNEUtrainXImg= np.resize(PNEUtrainXImg,(3875,224,224,1))
# #PNEUTstImg=np.resize(PNEUTstImg,(390,224,224,1))
# #NORMALTstImg=np.resize(NORMALTstImg,(234,224,224,1))
# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator( rescale=1./255,
#                                   #rotation_range=20,)
#                                  # width_shift_range=0.2,
#                                   height_shift_range=0.2,
#                                   #shear_range=0.2,
#                                   #zoom_range=0.2,
#                                   horizontal_flip=True,)
#       #validation_split=0.3,
#                                   #fill_mode='nearest',)

# #to grnerate 30.000 CXR imgs --> 100*100 for train & 50*100 for tst
# i=0
# for batch in train_datagen.flow(EOSINOPHILlTrainXImg,#labels="inferred",PNEUtrainXImg
#                           batch_size=10,
#                           #class_mode="binary",
#                           #label_mode="int",
#                           #class_names=['NORMAL',"PNEUMONIA"],
#                           #color_mode='rgb',#image_size=(224,224)),#:
#                           #target_size=(224,224),
#                           save_to_dir='C:/Users/admin/Desktop/Images (Blood Cell types)/Augmented/EOSINOPHILl/',
#                           save_prefix=i):
      
#       i += 1
#       if i > 10:#100*50=5000
#           break
# #test_datagen = ImageDataGenerator(rescale=1./255,)        
# # i=0
# # for batch in test_datagen.flow(PNEUTstImg,#labels="inferred",
# #                           batch_size=50,
# #                           #class_mode="binary",
# #                           #label_mode="int",
# #                           #class_names=['NORMAL',"PNEUMONIA"],
# #                           #color_mode='rgb',#image_size=(224,224)),#:
# #                           #target_size=(224,224),
# #                           save_to_dir='30,000(70%&30%)/test/Pneumonia/',
# #                           save_prefix='Pneu_test_aug'):
      
# #       i += 1
# #       if i > 90: #13*30=390
# #         break 

from keras.preprocessing.image import ImageDataGenerator
from skimage import io
datagen = ImageDataGenerator(        
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,)
       # brightness_range = (0.5, 1.5))
###############################
# # get the folders of the images for training and testing sets
EOSINOPHILlTrain = "c"
# LYMPHOCYTE_Train = "C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/LYMPHOCYTE/*.jpeg"
# MONOCYTElTrain = "C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/MONOCYTE/*.jpeg"
# PNEU_Train = "C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/NEUTROPHIL/*.jpeg"
# #Normal_Test = "archive/chest_xray/test/NORMAL/*.jpeg"
# #PNEU_Test = "archive/chest_xray/test/PNEUMONIA/*.jpeg"
# # call the function
# EOSINOPHILlTrainXImg = loadData(EOSINOPHILlTrain)      
       
import numpy as np
import os
from PIL import Image
img_path= 'C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/NEUTROPHIL/*jpg'
SIZE = 224
dataset = []
#my_images = os.listdir(image_directory)
#for i, image_name in enumerate(my_images):
for file in glob.glob(img_path):    
    #if (file.split('.')[0] == 'jpg'):        
        image = io.imread(file)        
        image = Image.fromarray(image, 'RGB')        
        image = image.resize((SIZE,SIZE)) 
        dataset.append(np.array(image))
       
x = np.array(dataset)
i = 0
for batch in datagen.flow(x, batch_size=10,
                          save_to_dir= r'C:/Users/admin/Desktop/Images (Blood Cell types)/Augmented/NEUTROPHIL/',
                          save_prefix='NEUTROPHIL',
                          save_format='jpg'):    
    i += 1    
    if i > 10:  #MONOCYTE=17       
        break