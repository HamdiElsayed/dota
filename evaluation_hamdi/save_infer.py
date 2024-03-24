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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from evaluation_hamdi.metrics import calc_relative_error, calc_RMSE ,masked_relative_error, gamma_analysis, masked_RMSE


def infer_spread(model,path, filename, scale,test_df, ikey='GeometryAll', okey='DoseAll'):
    """
    Get model prediction from test sample ID.
    """
    


    geometry_path=os.path.join(path,ikey,filename+'.dcm')
    tmp_geometry=np.swapaxes(pydicom.dcmread(geometry_path).pixel_array,0,2)
    geometry = np.expand_dims(tmp_geometry, axis=(0,-1))
    
    inputs = (geometry - scale['x_min']) / (scale['x_max'] - scale['x_min'])
    ground_truth_filename=test_df[test_df['cropped_geometry_name']==filename]['cropped_dose_name'].iloc[0]+'.dcm'
    ground_truth_metadata=pydicom.dcmread(os.path.join(path,okey,ground_truth_filename))
    ground_truth_array=ground_truth_metadata.pixel_array*ground_truth_metadata.DoseGridScaling
    ground_truth= np.swapaxes(ground_truth_array,0,2)
    
    energy_temp = test_df[test_df['cropped_geometry_name']==filename]['energy'].iloc[0]
    energy=(energy_temp - scale['e_min']) / (scale['e_max'] - scale['e_min'])
    spread_temp=test_df[test_df['cropped_geometry_name']==filename]['beam_spread'].iloc[0]
    spread=(spread_temp - scale['s_min']) / (scale['s_max'] - scale['s_min'])

    
    energies_spread_combined = np.array([[energy, spread]])

    
    # Predict dose distribution
    prediction = model.predict([inputs, energies_spread_combined],verbose=None)
    prediction = prediction * (scale['y_max']-scale['y_min']) + scale['y_min']

    return np.squeeze(geometry), np.squeeze(prediction), np.squeeze(ground_truth)



def save_files(infer_model,transformer,testIDs,base_path,scale):

    geometry_path = os.path.join(base_path, "Geometry")
    prediction_path = os.path.join(base_path, "Prediction")
    ground_truth_path = os.path.join(base_path, "GroundTruth")


    for directory in [geometry_path, prediction_path, ground_truth_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for filename in tqdm(testIDs):
        geometry, prediction, ground_truth = infer_model(transformer, base_path, filename, scale)

    # Construct file paths
    geometry_file_path = os.path.join(geometry_path, f"{filename}.npy")
    prediction_file_path = os.path.join(prediction_path, f"{filename}.npy")
    ground_truth_file_path = os.path.join(ground_truth_path, f"{filename}.npy")

    # Save the numpy arrays
    np.save(geometry_file_path, geometry)
    np.save(prediction_file_path, prediction)
    np.save(ground_truth_file_path, ground_truth)



