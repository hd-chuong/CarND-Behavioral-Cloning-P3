
# coding: utf-8

# In[1]:


import csv
import os
import pandas as pd
import numpy as np
import cv2


# In[2]:


dataset_names = [
#                  "./training_data2", 
#                  "./training_proper",
#                  "./training_veer",
#                  "./training_veer_counter",
#                 "./data/data",
#                     "training_proper2",
#                     "training_proper3"
                "./training_right_turn",
                ]
csv_filename = "driving_log.csv"
image_folder = "IMG"


# In[3]:


image_paths = []
left_image_paths = []
right_image_paths = []
angles = []
throttles = []
breaks = []
speeds = []
for dataset_name in dataset_names:
    csv_path = os.path.join(dataset_name, csv_filename)
    dataset = pd.read_csv(csv_path, header=None)
    dataset = dataset.rename(columns={0: 'center img', 
                                     1: 'left img', 
                                     2: 'right img',
                                     3: 'steering angle', 
                                     4: 'throttle', 
                                     5: 'break', 
                                     6: 'speed'})
    
    paths = list(dataset['center img'].apply(lambda full_path: os.path.split(full_path)))
    image_names = [pair[1] for pair in paths]
    image_paths.extend([os.path.join(dataset_name, image_folder, image_name) for image_name in image_names])
    
    right_paths = list(dataset['right img'].apply(lambda full_path: os.path.split(full_path)))
    right_image_names = [pair[1] for pair in right_paths]
    right_image_paths.extend([os.path.join(dataset_name, image_folder, image_name) for image_name in right_image_names])

    left_paths = list(dataset['left img'].apply(lambda full_path: os.path.split(full_path)))
    left_image_names = [pair[1] for pair in left_paths]
    left_image_paths.extend([os.path.join(dataset_name, image_folder, image_name) for image_name in left_image_names])
    
    
    angles.extend(list(dataset['steering angle']))
    throttles.extend(list(dataset['throttle']))
    breaks.extend(list(dataset['break']))
    speeds.extend(list(dataset['speed']))


# In[4]:


backup = pd.DataFrame(list(zip(image_paths, left_image_paths, right_image_paths, angles, throttles, breaks, speeds)) )
backup.to_csv('./backup/right-turn.csv', header=None, index=None)


# In[5]:


backup.tail()

