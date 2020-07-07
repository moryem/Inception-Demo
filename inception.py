# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:16:38 2020

@author: Mor
"""

import os
os.chdir(r'C:\Users\Dror\Desktop\Thesis Code\Inception')
from preprocess import apply_aug
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from plot_confusion_matrix import plot_confusion_matrix
import numpy as np

#%% Load data

x = np.load('mammos.npy')
y = np.load('labels.npy')

x_train = x[:round(x.shape[0]*0.7),:,:,:]
y_train = y[:round(x.shape[0]*0.7),:]
x_test = x[round(x.shape[0]*0.7):,:,:,:]
y_test = y[round(x.shape[0]*0.7):,:]

# Apply augmentation
x_train, y_train = apply_aug(x_train, y_train)

# Preprocessing the data, so that it can be fed to the pre-trained inception v-3 model. 
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#%% Define Hyper-Parameters

num_of_clss = 2         # number of classes
lr = 1e-3               # learning rate 
beta_1 = 0.9            # beta 1 - for adam optimizer
beta_2 = 0.99           # beta 2 - for adam optimizer
epsilon = 1e-7          # epsilon - for adam optimizer
epochs = 20             # number of epochs
bs = 32                 # batch size
dp = 0.6               # dropout probability

#%% Build Model using Transfer Learning

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
        
base_model = InceptionV3(weights="imagenet", include_top=False, 
                   input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))

idx = getLayerIndexByName(base_model, 'mixed7')
# Freeze the layers
for layer in base_model.layers[:idx]:
    layer.trainable = False
for layer in base_model.layers[idx:]:
    layer.trainable = True
# Extract the last layer from mixed8
last = base_model.get_layer('mixed8').output

# Add more layers on top of it
#x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(last)
#x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#x = BatchNormalization()(x)
#x = Dropout(dp)(x)
#x = MaxPooling2D((2, 2))(x)
x = Flatten()(last)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Dropout(dp)(x)
x = Dense(32)(x)
x = BatchNormalization()(x)
x = Dropout(dp)(x)
out = Dense(num_of_clss, activation='softmax')(x)

model = Model(base_model.input, out)

model.summary()

#%% Compile and train
 
#define the optimizer and compile the model
adam = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# add early stopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
class_weight = {0:1., 1:2.}

# Train the model, iterating on the data in batches
history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=bs, 
                    callbacks=[monitor], class_weight=class_weight)

# Visualize
# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show(); plt.close()

#%% Evaluate

y_pred = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test)

# Print results
print('test loss:', test_loss)
print('test accuracy:', test_acc)

# Confusion Matrix
cm = confusion_matrix(y_test[:,1], np.around(y_pred[:,1]))
labels = ['Healthy','Breast Cancer']
plot_confusion_matrix(cm,labels,title='Confusion Matrix - Test Set',normalize=True)
#plt.savefig('conf_mat')
plt.show(); plt.close()

# AUC and ROC curve
fpr, tpr, thresholds = roc_curve(y_test[:,1], y_pred[:,1], pos_label=1)
auc = roc_auc_score(y_test, y_pred)
print('AUC value is:', auc)

plt.plot(fpr, tpr, lw=1, label='ROC curve (auc = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='darkorange', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
#plt.savefig('ROC Curve.png'); 
plt.show(); plt.close()
