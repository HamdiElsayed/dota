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
import json
import tensorflow as tf
import seaborn as sns
from pymedphys import gamma



def gamma_analysis(ground_truth, prediction ,lower_percent_dose_cutoff, dose_percent_threshold=1, distance_mm_threshold =3, resolution=[2,2,2]):
   

    axes = (np.arange(ground_truth.shape[0]) * resolution[0],
            np.arange(ground_truth.shape[1]) * resolution[1],
            np.arange(ground_truth.shape[2]) * resolution[2])
    
    gamma_values = gamma(axes, ground_truth, axes, prediction, dose_percent_threshold,
                         distance_mm_threshold, lower_percent_dose_cutoff,random_subset=int(np.prod(ground_truth.shape)/10),max_gamma=1.1)
    valid_gamma = gamma_values[~np.isnan(gamma_values)]

    # Calculate gamma pass rate.

    gamma_pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma)

    return round(gamma_pass_rate*100,3)


def calc_relative_error(ground_truth,prediction):

    ground_truth=np.array(ground_truth,dtype=float)
    prediction=np.array(prediction,dtype=float)


    n_v=np.prod(ground_truth.shape)
    diff=ground_truth-prediction
    return (1/n_v)* (np.sqrt(np.sum(np.abs(diff)))/np.max(ground_truth)) *100

def calc_RMSE(ground_truth,prediction):

    ground_truth=np.array(ground_truth,dtype=float)
    prediction=np.array(prediction,dtype=float)

    
    n_v=np.prod(ground_truth.shape)

    return np.sqrt((1/n_v)* (np.sum((ground_truth-prediction)**2)))



def masked_relative_error(ground_truth,prediction,percentage=0.1):

    ground_truth=np.array(ground_truth,dtype=float)
    prediction=np.array(prediction,dtype=float)
    max_gt=ground_truth.max()
    mask = ((ground_truth > (percentage / 100) * max_gt) | (prediction > (percentage / 100) * max_gt))
    
    ground_truth[~mask] = np.nan
    prediction[~mask] = np.nan
    
    n_v=np.sum(~np.isnan(ground_truth))  
    abs_diff = np.nansum(np.abs(ground_truth - prediction))

    relative_error = (1 / n_v) * (np.sqrt(abs_diff) / max_gt) * 100
    return round(relative_error,3)
    
def masked_RMSE(ground_truth,prediction,percentage=0.1):

    ground_truth=np.array(ground_truth,dtype=float)* (1e9 / 1e5)
    prediction=np.array(prediction,dtype=float)* (1e9 / 1e5)


    max_gt=ground_truth.max()
    mask = ((ground_truth > (percentage / 100) * max_gt) | (prediction > (percentage / 100) * max_gt))
    
    sum_part=np.nansum((ground_truth-prediction)**2)
                       
    n_v=np.sum(~np.isnan(ground_truth))  
    
    rmse=np.sqrt(sum_part/n_v)

    return rmse