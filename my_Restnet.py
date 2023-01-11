# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:07:35 2022

@author: 06006637
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from time import time
import warnings
warnings.filterwarnings('ignore')



"1-1設定訓練及測試圖檔之資料夾 路徑/名稱"
data_dir_path = '.'
train_dir = '/'.join((data_dir_path, 'train'))
testset_dir = '/'.join((data_dir_path, 'testset'))

"1-2設定模型訓練相關參數"
num_classes = 2  #不同類別數量
epochs = 100
batch_size = 32
img_size = 224 # 使用Restnet50 image size default設定為224

"1-3分類mapping代碼txt檔之路徑"
target_label_file_name = 'mapping.txt'
target_label_file_path = '/'.join((data_dir_path, target_label_file_name))


"2-1 train資料集建立"
## 建立train與 valid資料集、灰階轉彩圖、Normalization
def load_data(Gray2RGB=False, mean_proc=False, test_size=0.25, img_size=img_size):
    """ Load target labels """
    with open(target_label_file_path) as f:
        all_lines = [line.split(', ') for line in f.read().splitlines()]

    target_labels = dict()
    for line in all_lines:
        target_class, target_label = line
        target_labels[target_class] = target_label

    """ Create training data list """
    train_list = []
    img_paths = []
    img_labels = []
    for key in target_labels.keys():
        for img_path in glob('{}/{}/*.jpg'.format(train_dir, key)):
            train_list.append([img_path, target_labels[key]])
            img_paths.append(img_path)
            img_labels.append(target_labels[key])
               
    """ Split the list into training set and validation set """
    train_img_paths, valid_img_paths, y_train, y_valid = train_test_split(img_paths, img_labels, test_size=test_size)
  ##將圖片由路徑讀取   
    X_train = []
    for path in train_img_paths:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (img_size, img_size))
  ##將灰階影片轉成彩色        
        if Gray2RGB == True:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = img.astype(np.float32)
  ## Normalization            
        if mean_proc == 'VGG16_ImageNet':
            img = img - np.array([123.68, 116.779, 103.939]) ##將R、G、B分別減去各自統計出來的平均值
            img = img[:,:,::-1]  # RGB to BGR
            img = (img - np.min(img)) / np.max(img)
        if mean_proc == 'DenseNet':
            img /= 255.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = (img - mean) / std
        else:
            img /= 255.
            img -= 0.5
            img *= 2.
        X_train.append(img)
    X_train = np.array(X_train, dtype=np.float32)
    
    X_valid = []
    if float(test_size) != 0.:  #判斷test size不等於0時才準備驗證集資料
        for path in valid_img_paths:
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (img_size, img_size))

            if Gray2RGB == True:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            
            img = img.astype(np.float32)
            
            if mean_proc == 'VGG16_ImageNet':
                img = img - np.array([123.68, 116.779, 103.939])
                img = img[:,:,::-1]  # RGB to BGR
                img = (img - np.min(img)) / np.max(img)
            if mean_proc == 'DenseNet':
                img /= 255.
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                img = (img - mean) / std
            else:
                img /= 255.
                img -= 0.5
                img *= 2.
            X_valid.append(img)
    X_valid = np.array(X_valid, dtype=np.float32)

    if Gray2RGB == False:
        X_train = np.reshape(X_train, X_train.shape+(1,))
        X_valid = np.reshape(X_valid, X_valid.shape+(1,))
        
  # Convert class vectors to binary class matrices.    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)
    
    return X_train, y_train, X_valid, y_valid

"2-2 test資料集建立"
testset_list = []
test_id_list = []
for img_path in glob('{}/*.jpg'.format(testset_dir)):
    testset_list.append(img_path)
    id = img_path.split('/')[-1].split('.')[0]
    test_id_list.append(id)
testset_df = pd.DataFrame({'id': test_id_list, 'path': testset_list}).sort_values(by='id')

## 定義 test資料集的載入、Normalization
def load_test_data(Gray2RGB=False, mean_proc=False, img_size=img_size):
    img_path_list = []
    for img_path in glob('{}/*.jpg'.format(testset_dir)):
        img_path_list.append(img_path)
    X_test = []
    X_id = []
    for path in img_path_list:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (img_size, img_size))
        
        if Gray2RGB == True:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = img.astype(np.float32)

        if mean_proc == 'VGG16_ImageNet':
            img = img - np.array([123.68, 116.779, 103.939])
            img = img[:,:,::-1]  # RGB to BGR
            img = (img - np.min(img)) / np.max(img)
        if mean_proc == 'DenseNet':
            img /= 255.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = (img - mean) / std
        else:
            img /= 255.
            img -= 0.5
            img *= 2.
            
        img_id = path.split('/')[-1].split('.')[0]
        X_test.append(img)
        X_id.append(img_id)
        
    X_test = np.array(X_test, dtype=np.float32)
    
    if Gray2RGB == False:
        X_test = np.reshape(X_test, X_test.shape+(1,))
    
    return X_test, X_id


"3-1 Modeling"
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

## ResNet50套件載入
from tensorflow.keras.applications.resnet50 import ResNet50

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  

X_train, y_train, X_valid, y_valid = load_data(Gray2RGB=True,test_size=0.1, mean_proc=None, img_size=224)

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)

# Fine-tuning實例
model_name = 'ResNet50-Fine-tuning'

img_rows, img_cols, img_channel = 224, 224, 3


base_model = ResNet50(weights ='imagenet', include_top=False, input_shape =(img_rows, img_cols, img_channel))

#for layer in base_model.layers:
#    layer.trainable = False

x = base_model.output
x = Flatten()(x) 
x = Dropout(0.75)(x)
#x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()


"3-2 Model Training"
# Data generator with augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    )
##選擇 Optimizer
optimizer = keras.optimizers.Adam(lr=10e-6)

## 設定儲存路徑
model_path = './saved_models/{}.h5'.format(model_name)
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)

##設定 earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

##最後將模型進行compile

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

# Fit the model on the batches generated by datagen.flow().
batch_size = 10
aug_ratio = 1
epochs = 100
steps_per_epoch = int(aug_ratio * X_train.shape[0] / batch_size)
validation_steps = int(aug_ratio * X_valid.shape[0] / batch_size)

model_history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
                                    epochs = epochs,
                                    validation_data = (X_valid, y_valid),
                                    callbacks = [checkpoint, earlystop],
                                    steps_per_epoch=steps_per_epoch,
                                    validation_steps=validation_steps)

"3-3 Training Result Plotting"
#loss plot
training_loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.plot(training_loss, label="training_loss")
plt.plot(val_loss, label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.show()

#accuracy plot
training_acc = model_history.history['acc']
val_acc = model_history.history['val_acc']

plt.plot(training_acc, label="training_acc")
plt.plot(val_acc, label="validation_acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.show()


"3-4 Test data evaluate "
X_test, X_id = load_test_data(Gray2RGB=True, mean_proc=None)

model = load_model(model_path)

scores = model.evaluate(X_valid, y_valid, verbose=1)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])

y_test_pred_prob = model.predict(X_test)
y_test_pred = y_test_pred_prob.argmax(axis=-1)
y_test_pred_df = pd.DataFrame({'id': np.array(X_id), 'class':y_test_pred}).sort_values(by='id')
y_test_pred_df.to_csv('./submissions/{}.csv'.format(model_name), index=False)

