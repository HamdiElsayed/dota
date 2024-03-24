
import h5py
import json
import random
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

import pydicom
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
from models import dota_energies
from preprocessing import DataRescaler
from evaluation import gamma_analysis, error_analysis
from tqdm import tqdm
from scipy.linalg import norm
from pymedphys import gamma







def gamma_analysis(ground_truth, prediction,scale,cutoff=0.1, dose_threshold=1, distance_threshold=3, resolution=[2,2,2]):
    """
    Performs a gamma analysis for a single instance of ground_truth and prediction.
    Optionally calculates in which part of the beam (quadrant) the failed voxels are.
    """


    # Cut off MC noise
    ground_truth[ground_truth < (cutoff/100) * scale['y_max']] = 0
    prediction[prediction < (cutoff/100) * scale['y_max']] = 0

    # Calculate gamma values.
    axes = (np.arange(ground_truth.shape[0]) * resolution[0],
            np.arange(ground_truth.shape[1]) * resolution[1],
            np.arange(ground_truth.shape[2]) * resolution[2])
    gamma_values = gamma(axes, ground_truth, axes, prediction, dose_threshold,
                         distance_threshold, lower_percent_dose_cutoff=0,max_gamma=1.1)
    gamma_values = np.nan_to_num(gamma_values, 0)

    # Calculate gamma pass rate.
    gamma_pass_rate = np.sum(gamma_values <= 1) / np.prod(gamma_values.shape)

    return gamma_pass_rate*100


def calc_relative_error(ground_truth,prediction):
    n_v=np.prod(ground_truth.shape)
    diff=ground_truth-prediction
    return (1/n_v)* (np.sqrt(np.sum(np.abs(diff)))/np.max(ground_truth)) *100

def calc_RMSE(ground_truth,prediction):
    n_v=np.prod(ground_truth.shape)

    return np.sqrt((1/n_v)* (np.sum((ground_truth-prediction)**2)))





path = r"/tudelft.net/staff-umbrella/simelectrons/OneGeometryMultipleEnergy"
test_df=pd.read_pickle(os.path.join(path,'test_picklefile.pkl'))

testIDs=test_df['cropped_geometry_name'].tolist()
cutoff=10
relative_errors=[]
RMSEs=[]
gamma_pass_rates=[]

with open('scale_mpe.json', 'r') as file:
    scale_json = file.read()
scale = json.loads(scale_json)


for filename in tqdm(testIDs):
    prediction=np.load(os.path.join(path,'prediction',filename+'.npy'))
    ground_truth=np.load(os.path.join(path,'ground_truth',filename+'.npy'))
    
    relative_error=calc_relative_error(ground_truth,prediction)
    gamma_pass_rate=gamma_analysis(ground_truth, prediction , scale)
    RMSE=calc_RMSE(ground_truth,prediction)

    
    relative_errors.append(relative_error)
    RMSEs.append(RMSE)
    gamma_pass_rates.append(gamma_pass_rate)
    
df=pd.DataFrame({'filename':os.listdir(os.path.join(path,"prediction")),'relative_error':relative_errors,'RMSE':RMSEs,'gamma_pass_rate':gamma_pass_rates})

df.to_csv('results.csv',index=False)