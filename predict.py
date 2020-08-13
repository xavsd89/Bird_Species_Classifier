# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:37:10 2020

@author: viper
"""

test_dir = 'test_bird'
model_name = 'bird_model.h5'
label_name = 'bird_cv_labels.pkl'

import tensorflow as tf
from keras.models import load_model
import os

model = load_model(model_name)
#model.summary()
print('Load model')

import pickle

with open(os.path.join(script_dir, label_name), 'rb+') as encoder_file:
    lb = pickle.load(encoder_file)
    print('Load LabelEncoder.')

lb_name_mapping = dict(zip(lb.transform(lb.classes_), lb.classes_))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from imutils import paths

img_size = 224

for imgFile in np.sort(list(paths.list_images(test_dir))):
    image = cv2.imread(imgFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))

    data = np.array(image)
    plt.imshow(data)
    plt.show()
    
    data = np.expand_dims(data, axis=0)

    preds = model.predict(data)
    top_three_prob = np.sort(preds[0])[-1:-4:-1]
    top_three_indices = np.argsort(preds[0])[-1:-4:-1]

    print('Original:', imgFile.split(os.path.sep)[-1])
    for i in range(len(top_three_prob)): 
        print('{:02d}: {:{}{}{}} |{:05.2f}%|'.format(
        	i+1, 
        	lb_name_mapping[top_three_indices[i]], '.', '<', 30, top_three_prob[i]*100))
    print()