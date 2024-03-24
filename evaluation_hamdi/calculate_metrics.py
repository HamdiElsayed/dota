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



def process_file(filename, path):
    """
    Load the prediction, geometry, and ground_truth files, then calculate the relative error and RMSE.
    """
    prediction = np.load(os.path.join(path, 'Prediction', filename + '.npy'))
    geometry = np.load(os.path.join(path, 'Geometry', filename + '.npy'))
    ground_truth = np.load(os.path.join(path, 'GroundTruth', filename + '.npy'))
    
    relative_error = calc_relative_error(ground_truth, prediction)
    RMSE = calc_RMSE(ground_truth, prediction)
    masked_RMSE_val = masked_RMSE(ground_truth, prediction)
    relative_error_masked = masked_relative_error(ground_truth, prediction)
    #gpr=gamma_analysis(ground_truth, prediction, 10)
    gpr=1
    max_val = np.max(prediction)
    max_val_gt = np.max(ground_truth)
    mean_geometry = np.mean(geometry)
    spread_geometry = np.std(geometry)
    max_geometry = np.max(geometry)
    
    return {
        'Relative Error [%]': relative_error, 
        'RMSE [Gy]': RMSE, 
        'max_pred [Gy]': max_val,
        'max_gt [Gy]': max_val_gt,
        'cropped_geometry_name': filename,
        'mean HU Geometry [HU]': mean_geometry,
        'Spread HU Geometry [HU]': spread_geometry,
        'Max HU Geometry [HU]': max_geometry,
        'RMSE Masked (0.1%) [Gy]' : masked_RMSE_val,
        'Relative Error Masked (0.1%)': relative_error_masked,
        'GPR (1%,3mm) [%]': gpr
    }

def main(testIDs, path):
    # Initialize lists to store results
    results = []

    # Use ThreadPoolExecutor to parallelize the file processing
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_file, filename, path) for filename in testIDs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    # Convert results list to DataFrame
    df = pd.DataFrame(results)
    return df
