#!/usr/bin/env python
# coding: utf-8

# Transformer Dose Calculation 
## Import libraries and define auxiliary functions
import h5py
import json
import random
import pandas as pd
import sys
import os

sys.path.append('./src')
import numpy as np
#from generators import DataGenerator
from generator_hamdi import DataGenerator
from models import dota_energies
from preprocessing import DataRescaler
from preprocessing_hamdi import get_scaling_factors
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.config import list_physical_devices
import json
import tensorflow as tf
print(list_physical_devices('GPU'))

## Define hyperparameters
# Training parameters

gpu_index = 0

# Set the GPU to be visible and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
print('Available GPUs: ', gpus)
mem_growth = tf.config.experimental.get_memory_growth(gpus[gpu_index])
print('Memory growth: ', mem_growth)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
    mem_growth = tf.config.experimental.get_memory_growth(gpus[gpu_index])
    print('GPU set to be visible and memory growth set to: ', mem_growth)

'''Initializing Batch size num epochs learning rate and weight decay. This is the same in my model'''
batch_size =  8
num_epochs = 30
learning_rate = 0.001
weight_decay = 0.0001

'''Initializing the number of tokens, projection dimension, number of heads, number of transformers and kernel size. This is the same in my model'''
# Load model and data hyperparameters
with open('./hyperparam.json', 'r') as hfile:
    param = json.load(hfile)

preprocess=False
# Load data files
path = r"/tudelft.net/staff-umbrella/simelectrons/OneGeometryOneEnergy"
data_df=pd.read_pickle(os.path.join(path,'main_picklefile_correct.pkl'))
path_ckpt = './weights/ckpt/weights.ckpt'
train_split = 0.7
val_split = 0.15
test_split = 0.15

listIDs=data_df['cropped_geometry_name'].tolist()

# Training, validation, test split.
random.seed(333)
random.shuffle(listIDs)

trainIDs = listIDs[:int(round(train_split*len(listIDs)))]
valIDs = listIDs[int(round(train_split*len(listIDs))):]
testIDs = listIDs[int(round((train_split + val_split)*len(listIDs))):]

test_df=data_df[data_df['cropped_geometry_name'].isin(testIDs)]
test_df.to_pickle(os.path.join(path,'test_picklefile.pkl'))



# Calculate or load normalization constants.

if preprocess==True:
    scale = get_scaling_factors(path,'GeometryAll','DoseAll', data_df)
    scale_json = json.dumps(scale)
    with open('scale.json', 'w') as file:
        file.write(scale_json)

with open('scale.json', 'r') as file:
    scale_json = file.read()
scale = json.loads(scale_json)


train_gen = DataGenerator(trainIDs, path, batch_size, data_df, scale)
val_gen = DataGenerator(valIDs,  path, batch_size, data_df, scale)


transformer = dota_energies(
    num_tokens=param['num_tokens'],
    input_shape=param['data_shape'],
    projection_dim=param['projection_dim'],
    num_heads=param['num_heads'],
    num_transformers=param['num_transformers'], 
    kernel_size=param['kernel_size'],
    causal=True
)
transformer.summary()

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

#Learning rate scheduler. Manually reduce the learning rate.
sel_epochs = [4,8,12,16,20,24,28]
lr_scheduler = LearningRateScheduler(
    lambda epoch, lr: lr*0.5 if epoch in sel_epochs else lr,
    verbose=1)

optimizer.learning_rate.assign(learning_rate)
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
