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
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
from scipy.stats import entropy


def gamma_analysis(ground_truth, prediction ,lower_percent_dose_cutoff, dose_percent_threshold=1, distance_mm_threshold =3, resolution=[2,2,2]):
    """Function to calculate the gamma pass rate between ground truth and prediction dose distributions.
    
    Args:
        ground_truth (np.ndarray): The ground truth dose distribution.
        prediction (np.ndarray): The model's predicted dose distribution.
        lower_percent_dose_cutoff (float): The lower dose cutoff percentage.
        dose_percent_threshold (float, optional): The dose percent threshold. Defaults to 1.
        distance_mm_threshold (float, optional): The distance threshold in mm. Defaults to 3.
        resolution (list, optional): The resolution of the dose distribution in mm. Defaults to [2,2,2]."""

    axes = (np.arange(ground_truth.shape[0]) * resolution[0],
            np.arange(ground_truth.shape[1]) * resolution[1],
            np.arange(ground_truth.shape[2]) * resolution[2])
    
    gamma_values = gamma(axes, ground_truth, axes, prediction, dose_percent_threshold,
                         distance_mm_threshold, lower_percent_dose_cutoff,max_gamma=1.1)
    valid_gamma = gamma_values[~np.isnan(gamma_values)]

    gamma_pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma)

    return round(gamma_pass_rate*100,3)


def calc_relative_error(ground_truth,prediction,spread):

    """Calculate the relative error between two dose distributions.
    
    Args:
        ground_truth (np.ndarray): The ground truth dose distribution.
        prediction (np.ndarray): The model's predicted dose distribution.
        spread (float): The gaussian positional spread of the beam in cm. This is later converted to mm.
        
    Returns:
        float: The relative error between the two dose distributions.
        
    """
    

    ground_truth=np.array(ground_truth,dtype=float)
    prediction=np.array(prediction,dtype=float)


    n_v=np.prod(ground_truth.shape)
    diff=ground_truth-prediction
    diff=diff/(spread*10)
    return (1/n_v)* (np.sqrt(np.sum(np.abs(diff)))/np.max(ground_truth)) *100

def calc_RMSE(ground_truth,prediction):
    """Calculate the root mean squared error between two dose distributions.
    
    Args:
        ground_truth (np.ndarray): The ground truth dose distribution.
        prediction (np.ndarray): The model's predicted dose distribution.
    
    Returns:
        float: The root mean squared error between the two dose distributions in Gy.
        
    """
    ground_truth=np.array(ground_truth,dtype=float)
    prediction=np.array(prediction,dtype=float)

    
    n_v=np.prod(ground_truth.shape)

    return np.sqrt((1/n_v)* (np.sum((ground_truth-prediction)**2)))



def masked_relative_error(ground_truth,prediction,spread,percentage=0.1):

    """Calculate the relative error between two dose distributions for areas where the dose is above a certain threshold.
    
    Args:
        ground_truth (np.ndarray): The ground truth dose distribution.
        prediction (np.ndarray): The model's predicted dose distribution.
        spread (float): The gaussian positional spread of the beam in cm. This is later converted to mm.
        percentage (float, optional): The percentage of the maximum dose to consider. Defaults to 0.1.
        
    Returns:
        float: The relative error between the two dose distributions.
    """

    ground_truth=np.array(ground_truth,dtype=float)
    prediction=np.array(prediction,dtype=float)
    max_gt=ground_truth.max()
    mask = ((ground_truth > (percentage / 100) * max_gt) | (prediction > (percentage / 100) * max_gt))
    
    ground_truth[~mask] = np.nan
    prediction[~mask] = np.nan
    
    n_v=np.sum(~np.isnan(ground_truth))  
    abs_diff = np.nansum(np.abs(ground_truth - prediction))/(spread*10)

    relative_error = (1 / n_v) * (np.sqrt(abs_diff) / max_gt) * 100
    return round(relative_error,3)
    
def masked_RMSE(ground_truth,prediction,percentage=0.1):
    """Calculate the root mean squared error between two dose distributions for areas where the dose is above a certain threshold.
    
    Args:
        ground_truth (np.ndarray): The ground truth dose distribution.
        prediction (np.ndarray): The model's predicted dose distribution.
        percentage (float, optional): The percentage of the maximum dose to consider. Defaults to 0.1.
        
    Returns:
        float: The root mean squared error between the two dose distributions in Gy.
    """
    ground_truth=np.array(ground_truth,dtype=float)* (1e9 / 1e6)
    prediction=np.array(prediction,dtype=float)* (1e9 / 1e6)


    max_gt=ground_truth.max()
    mask = ((ground_truth > (percentage / 100) * max_gt) | (prediction > (percentage / 100) * max_gt))
    
    sum_part=np.nansum((ground_truth-prediction)**2)
                       
    n_v=np.sum(~np.isnan(ground_truth))  
    
    rmse=np.sqrt(sum_part/n_v)

    return rmse



#The three functions below are used together to calculate the local entropy of a 3D CT scan.
#The input to the function should not be the complete CT scan but the area which actually interacts with the VHEE beam.

def pad_array_to_multiple(arr, block_size):
    """Pad an array to be a multiple of a block size. The array is padded with -1000."""
    
    pad_width = [(0, (block_size - (dim % block_size)) % block_size) for dim in arr.shape]
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=-1000)
    return padded_arr

def normalize_intensity(ct_scan):

    """Normalize the intensity of a CT scan to the range [0, 255]"""
    ct_scan = np.clip(ct_scan, -1000, 2996)  # Clipping to typical HU range
    ct_scan = (ct_scan - np.min(ct_scan)) / (np.max(ct_scan) - np.min(ct_scan)) * 255
    return ct_scan.astype(np.uint8)

def calculate_local_entropy_3d(ct_scan, patch_size):

    """Calculate the local entropy of a 3D CT scan.
    
    Args:
        ct_scan (np.ndarray): The 3D CT scan.
        patch_size (int): The size of the patches used to calculate the entropy.
    
    Returns:
        np.ndarray: The local entropy of the CT scan.
    """
    normalized_ct_scan = normalize_intensity(ct_scan)
    padded_ct_scan = pad_array_to_multiple(normalized_ct_scan, patch_size)
    patches = view_as_blocks(padded_ct_scan, (patch_size, patch_size, patch_size))
    entropy_map = np.zeros_like(padded_ct_scan, dtype=float)
    
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                patch = patches[i, j, k].ravel()
                if np.any(patch):
                    hist, _ = np.histogram(patch, bins=256, range=(0, 255), density=True)
                    hist = hist[hist > 0]
                    ent = entropy(hist, base=2)
                    entropy_map[i*patch_size:(i+1)*patch_size, 
                                j*patch_size:(j+1)*patch_size, 
                                k*patch_size:(k+1)*patch_size] = ent
    
    return entropy_map[:ct_scan.shape[0], :ct_scan.shape[1], :ct_scan.shape[2]]




