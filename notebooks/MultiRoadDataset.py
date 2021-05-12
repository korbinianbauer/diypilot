from tensorflow.keras.utils import Sequence

import pandas as pd
import random
import math
import numpy as np

from RoadDataset import RoadDataset

from diypilot_augmentation import lateral_shift_augmentation, horizontal_rotation_augmentation

class MultiRoadDataset(Sequence):
    
    def __init__(self, column_names, batch_size=16):
        self.datasets = []
        self.column_names = column_names
        self.batch_size = batch_size
        
        
    def addRoadDataset(self, csv_path, frames_path):
        
        dataset = RoadDataset(csv_path, frames_path, self.column_names, batch_size = 1)
        self.datasets.append(dataset)
        
        #swas = dataset.get_csv()["steering_wheel_angle"]
        #for swa in swas:
        #    print(swa)
        
    def __len__(self):
        #return 500
        total_len = 0
        for dataset in self.datasets:
            total_len += len(dataset)
        return int(math.floor(total_len / float(self.batch_size)))
        
    def __getitem__(self, batch_index):
        
        batch_frames = []
        batch_velocities = []
        batch_swas = []
        
        while len(batch_frames) < self.batch_size:
            
            dataset = random.choice(self.datasets)
            #dataset = self.datasets[0]
            if len(dataset) == 0:
                continue
            
            sample_idx = random.randint(0, len(dataset)-1)
            sample = dataset[sample_idx]
            #sample = dataset[0]
            
            frame = sample[0][0][0]
            v_vehicle = sample[0][1][0]
            swa = sample[1][0]
            
            #frame = np.zeros_like(frame)
            #v_vehicle = 0
            #swa = 0.5
            
            #print("MultiRoadDataset yielding v_vehicle: {}, swa {}".format(v_vehicle, swa))
            
            batch_frames.append(frame)
            batch_velocities.append(v_vehicle)
            batch_swas.append(swa)
            
        #print("MultiRoadDataset yielding frames {}".format(batch_frames))
        #print("MultiRoadDataset yielding swas {}".format(batch_swas))
        return ([np.array(batch_frames), np.array(batch_velocities)], np.array(batch_swas))
    
    def get_batch_size(self):
        return self.batch_size
    
    
    def get_sample(self, sample_index, augment=False, crop=False, normalize=False):
        
        for dataset in self.datasets:
            if len(dataset) >= sample_index:
                return dataset.get_sample(sample_index, augment, crop, normalize)
            sample_index -= len(dataset)
            
        print("MultiRoadDataset: get_sample() - sample_index Out of Range")
        print("Requested: {}".format(sample_index))
        print("Individual dataset lengths:")
        for dataset in self.datasets:
            print(len(dataset))
        
        
        
    def augment_sample(self, frame, orig_swa, v_vehicle, lateral_shift=None, hor_rotation=None):
        
        return self.datasets[0].augment_sample(frame, orig_swa, v_vehicle, lateral_shift, hor_rotation)
    
    def set_lateral_shift_range(self, lateral_shift_range):
        for dataset in self.datasets:
            dataset.set_lateral_shift_range(lateral_shift_range)
    
    
    def crop_to_roi(self, frame):
        
        return self.datasets[0].crop_to_roi(frame)
    
    
    def clean(self):
        for dataset in self.datasets:
            dataset.clean()
            
        # only keep datasets with length > 0
        set_count_before = len(self.datasets)
        self.datasets = [dataset for dataset in self.datasets if len(dataset) > 0]
        set_count_after = len(self.datasets)
        
        print("\n Dropped {} datasets with length==0, {} remaining".format(set_count_before-set_count_after, set_count_after))
        
    def balance(self):
        for dataset in self.datasets:
            dataset.balance()
            
        # only keep datasets with length > 0
        set_count_before = len(self.datasets)
        self.datasets = [dataset for dataset in self.datasets if len(dataset) > 0]
        set_count_after = len(self.datasets)
        
        print("\n Dropped {} datasets with length==0, {} remaining".format(set_count_before-set_count_after, set_count_after))
        
    def get_csv(self):
        return pd.concat([dataset.get_csv()  for dataset in self.datasets])
    
    def get_mean_absolute_normalized_swa(self):
        ma_norm_swas = []
        for dataset in self.datasets:
            dataset_ma_norm_swa = dataset.get_mean_absolute_normalized_swa()
            print("Dataset dataset_ma_norm_swa: {}".format(dataset_ma_norm_swa))
            if np.isnan(dataset_ma_norm_swa):
                print(dataset.get_csv().describe())
            ma_norm_swas.append(dataset_ma_norm_swa)
        return np.mean(ma_norm_swas)
    
    def denormalize_swa(self, swa_norm):
        return self.datasets[0].denormalize_swa(swa_norm)