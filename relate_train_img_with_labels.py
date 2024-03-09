# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:07:59 2023

@author: Ishraq
"""
import numpy as np # linear algebra
import pandas as pd

# Train_data
labels_df = pd.read_csv('C:/Users/admin/Desktop/Images (Blood Cell types)/bdcbtae//train.csv')

# Create a set of all the image IDs that you want to use
image_ids = set(labels_df['Image'])

# Create a dictionary to map the image IDs to the labels
labels_dict = labels_df.set_index('Image').to_dict()['Category']

############################
import os
import cv2
import matplotlib
import shutil 
input_folder = "C:/Users/admin/Desktop/Images (Blood Cell types)/bdcbtae/Images/"
output_folder = "C:/Users/admin/Desktop/Images (Blood Cell types)/working/"
#for filename in os.listdir(input_folder):
    # Check if file is an image
    #if not filename.endswith('.jpeg'):
#path = "/kaggle/input/bdcbtae/Images"
img_data = []
img_labels = []
for file in os.listdir(input_folder):
        # Get the ID of the image from the file name
      image_id = file.replace("BloodImage_","").replace(".jpg","") #remove the prefix 
        # remove leading zeroes
      image_id = str(int(image_id))
      image_id = int(image_id)
      

      if image_id in image_ids:
          #if labels_df['Image'] in 
            # Load the image
          #img = cv2.imread(os.path.join(input_folder, file))
          #print(img)
          #img = cv2.resize(img, (256,256)) # resizing the image
          #img=img.reshape(-1)
          #image = img.copy()
          #show(img)
          #img_data.append(img)
            # Get the label for the image
          #label = labels_dict.get(image_id, None)
          #img_labels.append(label)
          #output_path = os.path.join(output_folder, file)
          #cv2.imwrite(output_path, image)
          #creating subfolders
          #for i in label:
              #os.makedirs(os.path.join('C:/Users/admin/Desktop/Images (Blood Cell types)/train_data', i))
  
          # moving the image files to their respective categories
          for c in labels_dict:
              if c=='NEUTROPHIL':
              # Category Name

                 #for i in os.listdir(input_folder):#_data:#list(labels_df[labels_df['Category']==c]['Image']):
                  # Image Id
                  in_path = os.path.join(input_folder, c)
                  dst_path = os.path.join(output_folder, c) 
                  #get_image = os.path.join('train', c) # Path to Images
                  
                  move_image_to_cat = shutil.move(input_folder,dst_path)#'train_data_/'+c)
                  move_image_to_cat
                   #print("Saved image to: {}".format(output_path))
                  
       ############################
    #     for f in label:
    #          src_path = os.path.join(source, img)
    #         dst_path = os.path.join(destination, f)
    # shutil.move(src_path, dst_path)






