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

def plot_slice(inputs, ground_truth, outputs, dose_threshold=1,
    distance_threshold=3, cutoff=0, figsize=(10,12), fontsize=10,
    resolution=[2,2,2], gamma_slice=False, savefig=False):
    """
    Plots slices of the full beam along the Z axis.
    *inputs..........3D array [Y,X,Z] from function infer
    """
    # Initialize figure and axes.
    fig, axs = plt.subplots(5, 1, figsize=figsize)
    axs[0].set_title("CT scan", fontsize=fontsize, fontweight='bold')
    axs[1].set_title("Target (MC)", fontsize=fontsize, fontweight='bold')
    axs[2].set_title("Predicted (model)", fontsize=fontsize, fontweight='bold')
    
    plt.subplots_adjust(hspace=0.675, wspace=0.0675)


    ground_truth[ground_truth<(cutoff/100)*scale['y_max']] = 0
    outputs[outputs<(cutoff/100)*scale['y_max']] = 0

    ground_truth=ground_truth* (1e9 / 1e5)
    outputs=outputs* (1e9 / 1e5)
    
    # Cut off MC noise

    

    # Calculate maximum and minimum per column.
    min_input, max_input = np.min(inputs), np.max(inputs)
    min_output, max_output = np.min(outputs), np.max(outputs)
    min_ground_truth, max_ground_truth = np.min(ground_truth), np.max(ground_truth)
    slice_number = int(np.floor(ground_truth.shape[-1]/2))
    cb_ticks=np.linspace(0, max_output, num=4)
    # 1st row: input values
    cbh0 = axs[0].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', vmin=min_input, vmax=max_input)
    plt.sca(axs[0])
    plt.yticks([68,34,1], ['2','34','68'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    axs[0].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[0].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb0 = fig.colorbar(cbh0, ax=axs[0], aspect=fontsize)
    cb0.ax.set_ylabel("HU", size=fontsize)
    cb0.ax.tick_params(labelsize=fontsize)

    # 2nd row: ground truth
    axs[1].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
    cbh1 = axs[1].imshow(np.transpose(ground_truth[:,:,slice_number]), aspect='auto',
        cmap='turbo', alpha=0.6, vmin=min_output, vmax=max_output)
    plt.sca(axs[1])
    plt.yticks([68,34,1], ['2','34','68'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    axs[1].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[1].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb1 = fig.colorbar(cbh1, ax=axs[1], aspect=fontsize,ticks=cb_ticks)
    cb1.ax.set_ylabel(r"Gy/$10^9$ particles", size=fontsize)
    cb1.ax.tick_params(labelsize=fontsize)

    # 3rd row: model prediction
    axs[2].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
    cbh2 = axs[2].imshow(np.transpose(outputs[:,:,slice_number]), aspect='auto', 
        cmap='turbo', alpha=0.6, vmin=min_output, vmax=max_output)
    plt.sca(axs[2])
    plt.yticks([68,34,1], ['2','34','68'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    axs[2].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[2].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb2 = fig.colorbar(cbh2, ax=axs[2], aspect=fontsize, ticks=cb_ticks)
    cb2.ax.set_ylabel(r"Gy/$10^9$ particles", size=fontsize)
    cb2.ax.tick_params(labelsize=fontsize)

    # 4th row: difference or gamma analysis results
    if gamma_slice:
        axes = (np.arange(ground_truth.shape[0])*resolution[0],
            np.arange(ground_truth.shape[1])*resolution[1],
            np.arange(ground_truth.shape[2])*resolution[2])
        gamma_values = np.nan_to_num(
            gamma(axes, ground_truth, axes, outputs, dose_threshold,
            distance_threshold, lower_percent_dose_cutoff=0.1, quiet=True), 0)
        axs[3].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
            cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
        cbh3 = axs[3].imshow(np.transpose(np.absolute(gamma_values[:,:,slice_number])),
            aspect='auto', alpha=0.6, vmin=0, vmax=2, cmap='RdBu')

    else:
        axs[3].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
            cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
        difference=np.transpose(np.absolute(ground_truth[:,:,slice_number]-outputs[:,:,slice_number]))
        cbh3 = axs[3].imshow(difference,
            aspect='auto', cmap='turbo', alpha=0.6, vmin=min_output, vmax=max_output)
        
    plt.sca(axs[3])
    plt.yticks([68,34,1], ['2','34','68'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    axs[3].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[3].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb3 = fig.colorbar(cbh3, ax=axs[3], aspect=fontsize,ticks=cb_ticks)

    if gamma_slice:
        axs[3].set_title("Gamma analysis", fontsize=fontsize, fontweight='bold')
    else:
        axs[3].set_title("Dose difference max={}".format(np.max(np.round(difference,2))), fontsize=fontsize, fontweight='bold')
    if gamma_slice:
        cb3.ax.set_ylabel(r"$\gamma$ value", size=fontsize)
    else:
        cb3.ax.set_ylabel(r"Gy/$10^9$ particles", size=fontsize)
    cb3.ax.tick_params(labelsize=fontsize)

    axs[4].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
            cmap='gray', alpha=0.4)

    
    
    relative_error = difference*100/max_ground_truth
    ticks_relative_error=np.linspace(0, np.max(relative_error), num=4)
   
    cbh4 = axs[4].imshow(relative_error, aspect='auto', cmap='turbo', alpha=0.6, vmin=0, vmax=np.max(relative_error))
    axs[4].set_title(f"Relative Error: {np.round(np.mean(relative_error[relative_error>0]),2)} %", fontsize=fontsize, fontweight='bold')
    cb4 = fig.colorbar(cbh4, ax=axs[4], aspect=fontsize,ticks=ticks_relative_error)
    cb4.ax.set_ylabel(r"%", size=fontsize)
    cb4.ax.tick_params(labelsize=fontsize)
    plt.sca(axs[4])
    plt.yticks([68,34,1], ['2','34','68'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    if savefig:
        plt.savefig(time.strftime('%Y%m%d-%H%M'), dpi=300, bbox_inches='tight') 

    plt.show()




