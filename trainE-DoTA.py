import h5py
import json
import random
import pandas as pd
import sys
import os

# WandB was used for logging the training process.

#import wandb
#from wandb.keras import WandbCallback
#from wandb.keras import WandbMetricsLogger

sys.path.append('./src')
import numpy as np
#from generators import DataGenerator
from generator_ray import DataGenerator
from model import EDoTA

from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.config import list_physical_devices
import json
import tensorflow as tf
print(list_physical_devices('GPU'))



batch_size =  16
num_epochs = 30
learning_rate = 0.001
weight_decay = 0.0001

#Set which GPU to use

#################
gpu_index = 2
gpus = tf.config.experimental.list_physical_devices('GPU')
print('Available GPUs: ', gpus)
mem_growth = tf.config.experimental.get_memory_growth(gpus[gpu_index])
print('Memory growth: ', mem_growth)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
    mem_growth = tf.config.experimental.get_memory_growth(gpus[gpu_index])
    print('GPU set to be visible and memory growth set to: ', mem_growth)

################



# Load hyperparameters of the model
with open('ModelHyperParameters/hyperparamters3D.json', 'r') as hfile:
    param = json.load(hfile)


#Path to save the model weights
path_weights = r"/tudelft.net/staff-umbrella/simelectrons/LowNoiseFolder"

#Path to the training data
path=r"/scratch/hamdielsayed/FinalModel"

#Load the training data, sometimes investigating any outliers in the data is necessary as they can affect the normalization process.
data_df=pd.read_pickle(os.path.join(path,'training_data_complete.pkl'))
data_df=data_df[data_df['max_dose']<0.002]

#Where to save the model ckpt
path_ckpt = os.path.join(os.path.dirname(path_weights),'FinalModelWeights/ckpt/Ablation.ckpt')

#Train-Validation split
train_split = 0.80
val_split = 0.20

#The cropped geometry names are used as the unique identifiers for the different training examples.
listIDs=data_df['cropped_geometry_name'].tolist()

random.seed(333)
random.shuffle(listIDs)


train_size = int(round(train_split * len(listIDs)))

# Split the list into training and validation sets
trainIDs = listIDs[:train_size]
valIDs = listIDs[train_size:]

# Load the scaling parameters
with open('RescalingParameters/scale_last_model_new.json', 'r') as file:
    scale_json = file.read()
scale = json.loads(scale_json)


# Create the data generators for training and validation
train_gen=DataGenerator(trainIDs, path, batch_size, data_df, scale)
val_gen=DataGenerator(valIDs, path, batch_size, data_df, scale)

# Create the model
transformer = EDoTA(
    inshape=param['inshape'],
    steps=param['steps'],
    enc_feats=param['enc_feats'],
    num_heads=param['num_heads'],
    num_transformers=param['num_transformers'],
    kernel_size=param['kernel_size']
)
#Print the model summary
transformer.summary()

#Set the optimizer
optimizer = LAMB(learning_rate=learning_rate, weight_decay_rate=weight_decay)
transformer.compile(optimizer=optimizer, loss='mse', metrics=[])



# Save best model at the end of the epoch.
checkpoint = ModelCheckpoint(
    filepath=path_ckpt,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min')

#Learning rate scheduler. Manually reduce the learning rate.
sel_epochs = [5,10,15,20,25]

#Reduce the learning rate by a factor of 0.5 at the selected epochs.
lr_scheduler = LearningRateScheduler(
    lambda epoch, lr: lr*0.5 if epoch in sel_epochs else lr,
    verbose=1)

# Fit the model
optimizer.learning_rate.assign(learning_rate)
history = transformer.fit(
    x=train_gen,
    validation_data=val_gen,
    epochs=num_epochs,
    verbose=1,
    callbacks=[checkpoint, lr_scheduler]
    )

# Save last weights and hyperparameters.
path_last = os.path.join(os.path.dirname(path_weights),'./FinalModelWeights/FinalModelWeights.ckpt')
transformer.save_weights(path_last)
