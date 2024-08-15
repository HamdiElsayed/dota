import glob
import h5py
import random
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from concurrent.futures import ThreadPoolExecutor

import pydicom
import pandas as pd
import os
import traceback
from typing import List, Tuple, Dict, Any


class DataGenerator(Sequence):
    def __init__(self, 
                 list_IDs: List[str], 
                 path: str, 
                 batch_size: int, 
                 df: pd.DataFrame, 
                 scale: Dict[str, float], 
                 ikey: str = 'GeometryAll', 
                 okey: str = 'DoseAll', 
                 rkey: str = 'Rays', 
                 dim: Tuple[int, int, int] = (150, 64, 64), 
                 multiply: bool = True, 
                 shuffle: bool = True) -> None:
        """
        Data Generator class for loading and preprocessing data into the model.

        Args:
            list_IDs (List[str]): List of IDs to load, each representing a specific geometry.
            path (str): Path to the data directory.
            batch_size (int): Number of samples per batch.
            df (pd.DataFrame): DataFrame containing metadata related to the data.
            scale (Dict[str, float]): Dictionary with scaling factors for normalizing the data.
            ikey (str, optional): Folder name containing the geometry data. Defaults to 'GeometryAll'.
            okey (str, optional): Folder name containing the dose data. Defaults to 'DoseAll'.
            rkey (str, optional): Folder name containing the ray data. Defaults to 'RaysNew'.
            dim (Tuple[int, int, int], optional): Dimensions of the data to load. Defaults to (150, 64, 64).
            multiply (bool, optional): If True, multiplies the ray data with the energy. Defaults to True.
            shuffle (bool, optional): If True, shuffles the data after each epoch. Defaults to True.

        Outputs:
            X: Input data including geometry array, dose shape array, and energy.
            y: Output data containing the ground truth dose arrays.
        """
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.df = df
        self.path = path
        self.ikey = ikey
        self.okey = okey
        self.rkey = rkey
        self.shuffle = shuffle
        self.on_epoch_end()

        self.rotk = np.arange(4)

        self.X_min = scale['x_min']
        self.X_max = scale['x_max']
        self.y_min = scale['y_min']
        self.y_max = scale['y_max']
        self.min_energy = scale['e_min']
        self.max_energy = scale['e_max']
        self.r_max = scale['r_max']
        self.r_min = scale['r_min']

        self.multiply = multiply

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        """Calculates the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __data_generation(self, list_IDs_temp: List[str]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generates data containing batch_size samples.

        Args:
            list_IDs_temp (List[str]): List of IDs to load for the current batch.

        Returns:
            Tuple: A tuple (X, y) where:
                - X (List[np.ndarray]): A list of input arrays containing the geometry, ray, and energy data.
                - y (np.ndarray): Ground truth dose data.
        """
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))
        r = np.empty((self.batch_size, *self.dim))
        energies = np.empty((self.batch_size))

        def load_process_data(ID: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            # Load geometry
            tmp_geometry = pydicom.dcmread(os.path.join(self.path, self.ikey, ID + '.dcm')).pixel_array

            # Load dose
            corresponding_dose_name = self.df[self.df['cropped_geometry_name'] == ID]['cropped_dose_name'].iloc[0]
            dose_metadata = pydicom.dcmread(os.path.join(self.path, self.okey, corresponding_dose_name + '.dcm'))
            tmp_dose = dose_metadata.pixel_array * dose_metadata.DoseGridScaling

            # Load ray
            corresponding_ray_name = self.df[self.df['cropped_geometry_name'] == ID]['RayName'].iloc[0]
            tmp_ray = np.load(os.path.join(self.path, self.rkey, corresponding_ray_name))

            # Load energy
            energy = self.df[self.df['cropped_geometry_name'] == ID]['energy'].iloc[0]

            #multiply ray with energy
            if self.multiply:
                tmp_ray = tmp_ray * energy

            # Reduce the dimensions, this was done because the generated data was in the shape of (150,68,68) which was not divisible by 4.
            tmp_geometry = tmp_geometry[2:-2, 2:-2, :]
            tmp_dose = tmp_dose[2:-2, 2:-2, :]
            tmp_ray = tmp_ray[2:-2, 2:-2, :]

            # Random rotation
            rot = np.random.choice(self.rotk)
            tmp_geometry = np.rot90(np.swapaxes(tmp_geometry, 0, 2), rot, axes=(1, 2))
            tmp_dose = np.rot90(np.swapaxes(tmp_dose, 0, 2), rot, axes=(1, 2))
            tmp_ray = np.rot90(np.swapaxes(tmp_ray, 0, 2), rot, axes=(1, 2))

            return tmp_geometry, tmp_dose, tmp_ray, energy

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(load_process_data, list_IDs_temp))

        for i, (geom, dose, ray, energy) in enumerate(results):
            X[i, ] = geom
            y[i, ] = dose
            r[i, ] = ray
            energies[i] = energy

        # Normalize
        X = (X - self.X_min) / (self.X_max - self.X_min)
        y = (y - self.y_min) / (self.y_max - self.y_min)
        r = (r - self.r_min) / (self.r_max - self.r_min)
        energies = (energies - self.min_energy) / (self.max_energy - self.min_energy)

        return [np.expand_dims(X, -1), np.expand_dims(r, -1), np.expand_dims(energies, -1)], np.expand_dims(y, -1)

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate one batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            Tuple: A tuple (X, y) where:
                - X (List[np.ndarray]): A list of model inputs containing the geometry array, ray array, and energy data.
                - y (np.ndarray): Ground truth dose distribution data.

        Raises:
            Exception: If data generation fails after a maximum number of attempts.
        """
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            try:
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

                # Find list of IDs
                list_IDs_temp = [self.list_IDs[k] for k in indexes]

                # Generate data
                X, y = self.__data_generation(list_IDs_temp)

                return X, y
            except Exception as e:
                print(traceback.format_exc())
                attempts += 1
                index += 1
                if index * self.batch_size >= len(self.indexes):
                    index = 0
        raise Exception('Data generation failed after {} attempts'.format(max_attempts))
