# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:20:52 2020

@author: viper
"""
from __future__ import print_function
import pandas as pd
import shutil
import os
import sys
import csv

#%%
labels = pd.read_csv('trainLabels.csv')

# Create `train_sep` directory
train_dir = 'images_train/'
train_sep_dir = 'train_sep/'
if not os.path.exists(train_sep_dir):
    os.mkdir(train_sep_dir)

for filename, class_name in labels.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(train_sep_dir + class_name):
        os.mkdir(train_sep_dir + class_name)
    src_path = train_dir + filename + '.jpg'
    dst_path = train_sep_dir + class_name + '/' + filename + '.jpg'
    try:
        shutil.copy(src_path, dst_path)
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))
#%%
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import VGG16
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
import keras.utils 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imutils import paths

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import seed

import cv2
import os

np.random.seed(42)
#%%
import random

images_dir = 'train_sep'
script_dir = ''
label_name = 'bird_cv_labels.pkl'
model_name = 'bird_model.h5'

imagePaths = sorted(list(paths.list_images(images_dir)))
random.shuffle(imagePaths)

data = []
labels = []

# size is 224 x 224 pixels as required by VGG 16 model
img_size = 224

# number of epochs to run
num_epochs = 75

print("Load images.")
#%%
import tqdm

# loop over the image paths and load the images into array
for imagePath in tqdm.tqdm(imagePaths):
	# extract the class label from the directory name
	label = imagePath.split(os.path.sep)[-2]

	# load the image and resize it 
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (img_size, img_size))

	# update the data and labels lists
	data.append(image)
	labels.append(label)

# convert the data and labels to numpy array
data = np.array(data)
labels = np.array(labels)
#%%
print('Total rows of data: {}'.format(len(data)))
print('Total rows of labels: {}'.format(len(labels)))
#%%

# the labels are the bird names so need to One-Hot encoding on the categories
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = keras.utils.to_categorical(labels)

# create a dictionary, mapping the label name to its number encoded
lb_name_mapping = dict(zip(lb.transform(lb.classes_), lb.classes_))

# record the number of unique labels
img_classes=len(lb_name_mapping)

print('Number of unique labels(classes): {}'.format(img_classes))

print(labels.shape)
print(img_classes)
#%%
# split the data into training and testing using 75% of the data for training 
# and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)
#%%
# initialize the training data augmentation variable
# this is because there are imbalances in the dataset
# some bird species have few images while some 
# have many images.
train_datagen = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	vertical_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation variables 
test_datagen = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation variables
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
train_datagen.mean = mean
test_datagen.mean = mean
#%%
# load VGG16, ensuring the head fully connected layer sets are left off
# adjust the size of the input image tensor to the network
baseModel = VGG16(weights="imagenet", 
                  include_top=False, 
                  input_tensor=Input(shape=(img_size, img_size, 3)))
    
print("Summary.")
print(baseModel.summary())

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(img_size, activation="relu")(headModel)
headModel = Dropout(rate=0.5)(headModel)
headModel = Dense(img_classes, activation="softmax")(headModel)

# place the head fully connected model on top of the base model 
# (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

#freeze all the vgg base model layers
for layer in baseModel.layers:
	layer.trainable = False

#%%
# compile the model 
print("Compiling model.")
model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])

callbacks_list = [
 	        #keras.callbacks.EarlyStopping(monitor='acc', patience=1),
 			keras.callbacks.ModelCheckpoint(filepath=os.path.join(script_dir, model_name), monitor='val_loss', save_best_only=True)
]
#%%
# train the head of the network for a few epochs 
# (all other layers are frozen) 
print("Training.")
history = model.fit_generator(
	train_datagen.flow(trainX, trainY, batch_size=32), steps_per_epoch=len(trainX) // 32,
	validation_data=test_datagen.flow(testX, testY), validation_steps=len(testX) // 32,
	epochs=num_epochs,
    callbacks= callbacks_list)


#%%
print('Save model.')
model.save('bird_model.h5')
#saving model to folder
pickle.dump(model, open('model.pkl','wb'))
#create a model object 'loaded_model'
loaded_model = pickle.load(open('model.pkl', 'rb'))
#%%
# save the label encoder
import pickle
with open(os.path.join(script_dir, label_name), 'wb+') as encoder_file:
    pickle.dump(lb, encoder_file)
    print('Save LabelEncoder.')
#%%
# Validation
predictions = model.predict(testX, batch_size=32)

fig = plt.figure(figsize=(16,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(np.arange(0, num_epochs), history.history["loss"], label="train_loss")
ax1.plot(np.arange(0, num_epochs), history.history["val_loss"], label="val_loss")
ax1.title.set_text('Training and Validation Losses')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(np.arange(0, num_epochs), history.history["accuracy"], label="train_acc")
ax2.plot(np.arange(0, num_epochs), history.history["val_accuracy"], label="val_acc")
ax2.title.set_text('Training and Validation Accuracies')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print('Accuracy  = {:.2f}'.format(accuracy_score(testY.argmax(axis=1), predictions.argmax(axis=1))))
#%%
#TESTING THE MODEL 
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