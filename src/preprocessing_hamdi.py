import glob
import h5py
import numpy as np
import pydicom
from tqdm import tqdm
import os
import traceback
def get_scaling_factors(path,ikey,okey,df):

    e_min=df['energy'].min()
    e_max=df['energy'].max()
    if 'beam_spread' in df.columns:
        s_min=df['beam_spread'].min()
        s_max=df['beam_spread'].max()
    else:
        s_min,s_max=0,0
    min_value, max_value = -np.float32('inf'), np.float32('inf')
    error_IDs=[]
    geometry_list=df['cropped_geometry_name'].tolist()
    
    for i in tqdm(range(len(geometry_list))):

        try:

            geometry=pydicom.dcmread(os.path.join(path,ikey,df['cropped_geometry_name'].iloc[i]+'.dcm')).pixel_array
            metadata_dose=pydicom.dcmread(os.path.join(path,okey,df['cropped_dose_name'].iloc[i]+'.dcm'))
            dose=metadata_dose.pixel_array*metadata_dose.DoseGridScaling
            

            x_max, x_min = min(max_value, np.max(geometry)),max(min_value, np.min(geometry))
            y_max, y_min = min(max_value, np.max(dose)),max(min_value, np.min(dose))
        except Exception as e:
            error_IDs.append(df['cropped_geometry_name'].iloc[i])
            
            continue

    
    scaling_dic={'x_min': float(x_min),'x_max':float(x_max),
                'y_min':float(y_min),'y_max':float(y_max),
                'e_min':float(e_min),'e_max':float(e_max),
                 's_min':float(s_min),'s_max':float(s_max)}
    
    return scaling_dic,error_IDs
        

        
            




            



    