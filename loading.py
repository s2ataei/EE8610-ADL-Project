# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:53:18 2019

@author: s2ataei
"""

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras 
from keras.callbacks import TensorBoard
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import adam, sgd
from keras import losses
from keras.utils import np_utils
import scipy.io as spio
import time

print(tf.__version__)

images = np.empty((7352,36,128,1))
images_test = np.empty((2947,36,128,1))
imgpath = r"C:\Users\s2ataei\stuff\activity images"
#for x in range (7352):
#    y= x+1
#    images[x, :, :, 0] = mpimg.imread(os.path.join(imgpath,str(y)+'.jpg'))
#    #temp = spio.loadmat(os.path.join(imgpath,str(y)+'.mat'), squeeze_me=True)
#    #images[x,:,:,0] = temp['temp']
#    
imgpath_test = r"C:\Users\s2ataei\stuff\original_test .mat"    
for x in range (2947):
    y= x+1
    #images[x, :, :, 0] = mpimg.imread(os.path.join(imgpath,str(y)+'.jpg'))
    temp = spio.loadmat(os.path.join(imgpath_test,str(y)+'.mat'), squeeze_me=True)
    images_test[x,:,:,0] = temp['temp']   
    
    
labels = np.genfromtxt(r"C:\Users\s2ataei\stuff\UCI HAR Dataset\train\y_train-categorical.txt",delimiter=' ')
labels_test = np.genfromtxt(r"C:\Users\s2ataei\stuff\UCI HAR Dataset\test\y_test_categorical.txt",delimiter=' ')
#labels = labels.astype(np.int32)  
labels_test = np_utils.to_categorical(labels_test, num_classes = 6)
labels = np_utils.to_categorical(labels, num_classes = 6)



#labels_overfit = labels [0:100,:]
#images_overfit = images [0:100,:,:,:]

model = Sequential()

model.add(Conv2D(5, (5, 5), input_shape=(36,128,1), data_format="channels_last", activation = "sigmoid"))
model.add(AveragePooling2D(pool_size=(4, 4), strides = None))
model.add(Conv2D(10, (5, 5), activation ="sigmoid"))
#model.add(Dropout(0.5))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(120))
model.add(Dense(6, activation="softmax"))



#optimizer = adam(lr=0.1, beta_1=0.9, beta_2=0.999)
optimizer = sgd(lr=0.1, momentum=0.001, decay=0.00001, nesterov=False)
loss = keras.losses.categorical_crossentropy

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.summary() 
#
#rng_state = np.random.get_state()
#np.random.shuffle(labels)
#np.random.set_state(rng_state)
#np.random.shuffle(images)





#images_normal = images / (np.max(images) - np.min(images))

tboard = keras.callbacks.TensorBoard(log_dir='./logs21/')
tboard2 = keras.callbacks.TensorBoard(log_dir='./logs22/')
tboard_jpeg = keras.callbacks.TensorBoard(log_dir='./jpeg test/')

#model.fit(x=images, y=labels, batch_size=32, epochs=400, verbose=1, shuffle=True, validation_split=0.2, callbacks=[tboard_jpeg])
#

#model.load_weights('95%.h5')
model.evaluate(x=images_test, y=labels_test, verbose=1) 


