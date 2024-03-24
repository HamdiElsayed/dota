from pymedphys import gamma
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Define the part number and gamma pass rate percentage for this file
n = 1  # Example: Change this to 1, 2, or 3 depending on the file
percentage = 1  # Example: Change this to 1, 5, or 10 depending on the gamma analysis being performed

def gamma_analysis(ground_truth, prediction, lower_percent_dose_cutoff, dose_percent_threshold=1, distance_mm_threshold=3, resolution=[2,2,2]):
    # Calculate gamma values.
    axes = (np.arange(ground_truth.shape[0]) * resolution[0],
            np.arange(ground_truth.shape[1]) * resolution[1],
            np.arange(ground_truth.shape[2]) * resolution[2])
    gamma_values = gamma(axes, ground_truth, axes, prediction, dose_percent_threshold,
                         distance_mm_threshold, lower_percent_dose_cutoff, max_gamma=1.1)
    valid_gamma = gamma_values[~np.isnan(gamma_values)]

    # Calculate gamma pass rate.
    gamma_pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma)
    return round(gamma_pass_rate*100, 3)

path = "/tudelft.net/staff-umbrella/simelectrons/OneGeometryMultipleEnergyMultipleSpread"
test_df = pd.read_pickle(os.path.join(path, 'test_picklefile.pkl'))

# Assuming you've already loaded test_df and defined path
testIDs = test_df['cropped_geometry_name'].tolist()

# Dynamic calculation of start and end indices based on part number
total_ids = len(testIDs)
part_size = total_ids // 3
start_index = (n - 1) * part_size
end_index = start_index + part_size if n < 3 else total_ids

# Slicing testIDs based on calculated indices
testIDs_part = testIDs[start_index:end_index]

names_list = []
gpr_list = []
for filename in tqdm(testIDs_part):
    prediction = np.load(os.path.join(path, 'Prediction', filename + '.npy'))
    ground_truth = np.load(os.path.join(path, 'GroundTruth', filename + '.npy'))
    gpr = gamma_analysis(ground_truth, prediction,percentage)
    names_list.append(filename)
    gpr_list.append(gpr)
    # Your existing loop code, adjusting for the dynamic slicing

# Save results in a uniquely named DataFrame and CSV
dataframe = pd.DataFrame({'names': names_list, f'GPR {percentage} Percent': gpr_list})
dataframe.to_csv(os.path.join(path, f'gamma_results_{percentage}_part_{n}.csv'), index=False)
