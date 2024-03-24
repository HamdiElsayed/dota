#!/usr/bin/env python
# coding: utf-8

# Transformer Dose Calculation 
## Import libraries and define auxiliary functions
import h5py
import json
import random
import pandas 
import sys
sys.path.append('./src')
import numpy as np
from generators import DataGenerator
from models import dota_energies
from preprocessing import DataRescaler
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.config import list_physical_devices
print(list_physical_devices('GPU'))

## Define hyperparameters
# Training parameters

'''Initializing Batch size num epochs learning rate and weight decay. This is the same in my model'''
batch_size = 8
num_epochs = 30
learning_rate = 0.001
weight_decay = 0.0001

'''Initializing the number of tokens, projection dimension, number of heads, number of transformers and kernel size. This is the same in my model'''
# Load model and data hyperparameters
with open('./hyperparam.json', 'r') as hfile:
    param = json.load(hfile)
    


# Load data files
path = r"data"
data_df=pd.read_pickle(os.path.join(path,'main_picklefile.pkl'))
path_ckpt = './weights/ckpt/weights.ckpt'
filename = path + 'train.h5'
train_split = 0.7
val_split = 0.15
test_split = 0.15
with h5py.File(filename, 'r') as fh:
    listIDs = [*range(fh['geometry'].shape[-1])]

# Training, validation, test split.
random.seed(333)
random.shuffle(listIDs)
trainIDs = listIDs[:int(round(train_split*len(listIDs)))]
valIDs = listIDs[int(round(train_split*len(listIDs))):]
    
# Calculate or load normalization constants.
scaler = DataRescaler(path, filename=filename)
scaler.load(inputs=True, outputs=True)
scale = {'y_min':scaler.y_min, 'y_max':scaler.y_max,
        'x_min':scaler.x_min, 'x_max':scaler.x_max,
        'e_min':70, 'e_max':220}

# Initialize generators.
train_gen = DataGenerator(trainIDs, batch_size, filename, scale, num_energies=2)
val_gen = DataGenerator(valIDs, batch_size, filename, scale, num_energies=1)

## Define and train the transformer.


# Load weights from checkpoint.
random.seed()
transformer.load_weights(path_ckpt)

# Compile the model.
optimizer = LAMB(learning_rate=learning_rate, weight_decay_rate=weight_decay)
transformer.compile(optimizer=optimizer, loss='mse', metrics=[])

# Callbacks.
# Save best model at the end of the epoch.
checkpoint = ModelCheckpoint(
    filepath=path_ckpt,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min')

# Learning rate scheduler. Manually reduce the learning rate.
sel_epochs = [4,8,12,16,20,24,28]
lr_scheduler = LearningRateScheduler(
    lambda epoch, lr: lr*0.5 if epoch in sel_epochs else lr,
    verbose=1)

history = transformer.fit(
    x=train_gen,
    validation_data=val_gen,
    epochs=num_epochs,
    verbose=1,
    callbacks=[checkpoint, lr_scheduler]
    )

# Save last weights and hyperparameters.
path_last = './weights/weights.ckpt'
transformer.save_weights(path_last)
