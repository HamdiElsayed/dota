import glob
import h5py
import random
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
import pydicom
import pandas as pd
import os


class DataGenerator(Sequence):
    def __init__(self, list_IDs, path, batch_size,df,scale, ikey='GeometryAll',okey='DoseAll',dim=(150,68,68) ,shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.df=df
        self.path=path
        self.ikey = ikey
        self.okey = okey
        self.shuffle = shuffle
        self.on_epoch_end()

        self.rotk=np.arange(4)


        self.X_min = scale['x_min']
        self.X_max = scale['x_max']
        self.y_min = scale['y_min']
        self.y_max = scale['y_max']
        self.min_energy = scale['e_min']
        self.max_energy = scale['e_max']
        self.min_spread = scale['s_min']
        self.max_spread = scale['s_max']


    def on_epoch_end(self):
        # Updates indexes after each epoch.
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # Calculates the number of batches per epoch.
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))
        energies=np.empty((self.batch_size))
        spread=np.empty((self.batch_size))

        for i,ID in enumerate(list_IDs_temp):
        
            tmp_geometry=pydicom.dcmread(os.path.join(self.path,self.ikey,ID+'.dcm')).pixel_array
            corresponding_dose_name=self.df[self.df['cropped_geometry_name']==ID]['cropped_dose_name'].iloc[0]
            dose_metadata=pydicom.dcmread(os.path.join(self.path,self.okey,corresponding_dose_name+'.dcm'))
            tmp_dose=dose_metadata.pixel_array*dose_metadata.DoseGridScaling


            energies[i]=self.df[self.df['cropped_geometry_name']==ID]['energy'].iloc[0]
            spread[i]=self.df[self.df['cropped_geometry_name']==ID]['beam_spread'].iloc[0]


            rot=np.random.choice(self.rotk)

            X[i,]=np.rot90(np.swapaxes(tmp_geometry,0,2),rot,axes=(1,2))  #swap axes due to my date being in the shape of (z,x,y) instead of (x,y,z)
            y[i,]=np.rot90(np.swapaxes(tmp_dose,0,2),rot,axes=(1,2))
            #X[i,]=np.rot90(tmp_geometry,rot,axes=(0,1))
            #y[i,]=np.rot90(tmp_dose,rot,axes=(0,1))

        

        
        X=(X-self.X_min)/(self.X_max-self.X_min)
        y=(y-self.y_min)/(self.y_max-self.y_min)
        if self.min_energy!=self.max_energy:
            energies=(energies-self.min_energy)/(self.max_energy-self.min_energy)
        else:
            energies=energies/self.max_energy
        spread=(spread-self.min_spread)/(self.max_spread-self.min_spread)
            

        
   
        return [np.expand_dims(X, -1), np.stack([energies, spread], axis=-1)], np.expand_dims(y,-1)
    
 

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
    

                
