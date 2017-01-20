# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

os.getcwd()

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score

seed = 128
rng = np.random.RandomState(seed)


os.chdir("C:\\Users\\Admin\\Desktop\\identify the digits")

data_dir=os.getcwd()

data_dir


X_train = pd.read_csv(os.path.join(data_dir,'train.csv'))
X_test = pd.read_csv(os.path.join(data_dir,'test.csv'))

sub=pd.read_csv('sample_submission.csv')


X_train.head()

temp = []
for img_name in X_train.filename:
    image_path=os.path.join(data_dir,'Train','Images','train',img_name)
    img = imread(image_path,flatten=True)
    img=img.astype('float32')
    temp.append(img)
    
train_x = np.stack(temp)

train_x /=255.0

train_x = train_x.reshape(-1,784).astype('float32')    



temp = []
for img_name in X_test.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    
test_x = np.stack(temp)

test_x /= 255.0
test_x = test_x.reshape(-1, 784).astype('float32')



train_y = keras.utils.np_utils.to_categorical(X_train.label.values)



split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]



X_train.label.ix[split_size:]


# define vars
input_num_units = 784
hidden_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128

# import keras modules

from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential([
  Dense(output_dim=hidden_num_units, input_dim=input_num_units, activation='relu'),
  Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train model;
trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))


########model evalution 

pred = model.predict_classes(test_x)

sub.filename = X_test.filename; sub.label = pred

sub.to_csv(os.path.join(data_dir , 'sub_05.csv'),index=False)