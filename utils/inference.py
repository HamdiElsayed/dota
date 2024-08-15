import os
import numpy as np
import pydicom
from typing import Tuple, Any

def inference(model: Any, 
              path: str, 
              filename: str, 
              scale: dict, 
              test_df: 'pd.DataFrame', 
              okey: str = 'DoseAll', 
              ikey: str = "GeometryAll", 
              rkey: str = "Rays") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a model prediction for a single test case.

    This function loads the input geometry, ray data, and energy, scales them appropriately,
    and uses the provided model to predict the dose distribution. 

    Args:
        model (Any): The trained model used for prediction (E-DoTA)
        path (str): Filepath to the directory containing the test data.
        filename (str): The filename of the test case, typically the cropped geometry name.
        scale (dict): Dictionary containing the scaling parameters used for the model.
        test_df ('pd.DataFrame'): DataFrame containing metadata about the test data.
        okey (str, optional): Folder containing the ground truth dose distributions. Defaults to 'DoseAll'.
        ikey (str, optional): Folder containing the geometries. Defaults to 'GeometryAll'.
        rkey (str, optional): Folder containing the ray data. Defaults to 'Rays'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - geometry (np.ndarray): The input geometry used in the prediction.
            - prediction (np.ndarray): The model's predicted dose distribution.
            - ground_truth (np.ndarray): The ground truth dose distribution for comparison.
            - ray (np.ndarray): The input ray data used for the prediction.
    """

    # Load and preprocess the geometry data
    geometry_path = os.path.join(path, ikey, filename + '.dcm')
    tmp_geometry = np.swapaxes(pydicom.dcmread(geometry_path).pixel_array, 0, 2)
    geometry = np.expand_dims(tmp_geometry, axis=(0, -1))
    inputs = (geometry - scale['x_min']) / (scale['x_max'] - scale['x_min'])

    # Load and preprocess the ground truth dose data
    ground_truth_filename = test_df[test_df['cropped_geometry_name'] == filename]['cropped_dose_name'].iloc[0] + '.dcm'
    ground_truth_metadata = pydicom.dcmread(os.path.join(path, okey, ground_truth_filename))
    ground_truth_array = ground_truth_metadata.pixel_array * ground_truth_metadata.DoseGridScaling
    ground_truth = np.swapaxes(ground_truth_array, 0, 2)

    # Load and preprocess the energy data
    energy_temp = test_df[test_df['cropped_geometry_name'] == filename]['energy'].iloc[0]
    energy = (energy_temp - scale['e_min']) / (scale['e_max'] - scale['e_min'])
    energy = np.expand_dims(energy, axis=-1)

    # Load and preprocess the ray data
    ray_filename = test_df[test_df['cropped_geometry_name'] == filename]['RayName'].iloc[0]
    ray_path = os.path.join(path, rkey, ray_filename)
    ray = np.load(ray_path)
    ray = ray * energy_temp
    ray = np.swapaxes(ray, 0, 2)
    ray = np.expand_dims(ray, axis=(0, -1))
    ray = (ray - scale['r_min']) / (scale['r_max'] - scale['r_min'])

    # Crop the inputs and ground truth to reduce dimensions from (150,68,68) to (150,64,64)
    inputs = inputs[:, :, 2:-2, 2:-2, :]
    ray = ray[:, :, 2:-2, 2:-2, :]
    ground_truth = ground_truth[:, 2:-2, 2:-2]

    # Prepare model input and make the prediction
    model_input = [inputs, ray, energy]
    prediction = model.predict(model_input, verbose=0)
    
    # Rescale the prediction to the original dose range
    prediction = prediction * (scale['y_max'] - scale['y_min']) + scale['y_min']

    return np.squeeze(geometry), np.squeeze(prediction), np.squeeze(ground_truth), np.swapaxes(np.squeeze(ray), 0, 2)
