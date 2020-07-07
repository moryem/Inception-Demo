# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:51:22 2020

@author: Mor
"""

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

def split_sick_healthy(x, y):
# =============================================================================
#  Split into sick and healthy    
# =============================================================================
    x_sick = x[np.where(y[:,1]==1)[0],:,:,:]   
    y_sick = y[np.where(y[:,1]==1)[0],:]   
    x_healthy = x[np.where(y[:,1]==0)[0],:,:,:]
    y_healthy = y[np.where(y[:,1]==0)[0],:]
    
    return x_sick, x_healthy, y_sick, y_healthy

def augment(x, y, type_name):
# =============================================================================
#   Apply augmentation, more on sick patients    
# =============================================================================
        
    # create the data generator
    datagen = ImageDataGenerator(
        rotation_range = 90,
        horizontal_flip = True,
        vertical_flip = True)

    
    # concatenate 
    counter = 1
    
    for x_batch, y_batch in datagen.flow(x, y, batch_size = x.shape[0]):
        x = np.concatenate((x,x_batch),axis=0)
        y = np.concatenate((y,y_batch),axis=0)
        counter += 1
        if (type_name == 'sick'):
            if (counter == 10): # end after 12 variations
                break
        else:
            if (counter == 2): # end after 2 variations
                break            
    
    return x, y

def apply_aug(x, y):
# =============================================================================
#   Apply augmentation, merge and save    
# =============================================================================
    
    if os.path.exists('X.npy'):
        new_x = np.load('X.npy')
        new_y = np.load('Y.npy')
    else:
        # Split to sick and healthy cases
        x_sick, x_healthy, y_sick, y_healthy = split_sick_healthy(x, y)
        
        # Oversample (more on sick cases)
        x_aug_sick, y_aug_sick = augment(x_sick, y_sick, 'sick')
        x_aug_healthy, y_aug_healthy = augment(x_healthy, y_healthy, 'healthy')
        
        # Merge all cases together
        new_x = np.concatenate((x_aug_sick, x_aug_healthy), axis = 0)
        new_y = np.concatenate((y_aug_sick, y_aug_healthy), axis = 0)
        
        # Shuffle the data
        new_x, new_y = shuffle(new_x, new_y, random_state=22)
        
        np.save('X',new_x)
        np.save('Y',new_y)
    
    return new_x, new_y
    
    
    
