# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:29:49 2023

@author: admin
"""

import shutil, os
import pandas as pd

labels_df = pd.read_csv("C:/Users/admin/Desktop/Images (Blood Cell types)/bdcbtae/train.csv")
labels = labels_df.sort_values('Category')
# Create a set of all the image IDs that you want to use
# Create a set of all the image IDs that you want to use
#image_ids = set(labels_df['Image'])

# Create a dictionary to map the image IDs to the labels
#labels_dict = labels_df.set_index('Image').to_dict()['Category']


# Create a dictionary to map the image IDs to the labels
#labels = labels.set_index('Image').to_dict()['Category']
class_names = list(labels.Class.unique())



train_images = 'C:/Users/admin/Desktop/Images (Blood Cell types)/bdcbtae/Images'

train_cat = 'C:/Users/admin/Desktop/Images (Blood Cell types)/bdcbtae/train_'

#creating subfolders
for i in class_names:
    os.makedirs(os.path.join('C:/Users/admin/Desktop/Images (Blood Cell types)/train_', i))

# moving the image files to their respective categories
for c in class_names:
    # Category Name

    for i in list(labels_df[labels_df['Category']==c]['Image']):
        # Image Id
    
        get_image = os.path.join('train', i) # Path to Images
        move_image_to_cat = shutil.move(get_image, 'train_/'+c)