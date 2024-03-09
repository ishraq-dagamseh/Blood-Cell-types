# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 07:43:40 2023

@author: admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from skimage.io import imread, imshow
import cv2
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,confusion_matrix,classification_report,roc_auc_score
################
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
datagen = ImageDataGenerator(
        #rescale=1./255,        
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,)

#########################
labels_df = pd.read_csv('C:/Users/admin/Desktop/Images (Blood Cell types)/bdcbtae/train.csv')

# Create a set of all the image IDs that you want to use
image_ids = set(labels_df['Image'])
###################
#path = "NewWBpics/"
path= 'C:/Users/admin/Desktop/Images (Blood Cell types)/Images (Blood Cell types)/MONOCYTE/*jpg'
import glob

img_data = []
#img_labels = []
#for file in os.listdir(path):
for file in glob.glob(path): 
    # if file.endswith(".jpg"):
    #     # Get the ID of the image from the file name
    #     image_id = file.replace("BloodImage_","").replace(".jpg","") #remove the prefix 
    #     # remove leading zeroes
    #     image_id = str(int(image_id))
    #     image_id = int(image_id)

    #     if image_id in image_ids:
            
            # Load the image
    image = io.imread(file) 
            #img = cv2.imread(os.path.join(path, file))
           # image = Image.fromarray(img, 'RGB')        
            #image = image.resize((224,224)) 
            #dataset.append(np.array(image))
            #img=img.reshape(-1)
            #print(img)
            #img = cv2.resize(img, (256,256)) # resizing the image
    img_data.append(np.array(image))
            
x = np.array(img_data)
i = 0
for batch in datagen.flow(x, batch_size=10,
                          save_to_dir= r'C:\Users\admin\Desktop\Images (Blood Cell types)\Aug_withoutRescale&without resize/MONOCYTE',
                          #save_prefix='NEUTROPHIL',
                          save_format='jpg'):    
    i += 1    
    if i > 17:  #MONOCYTE=17 else 10      
        break            