from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import math
from matplotlib import pyplot as plt
import numpy as np
import cv2
from IPython.display import Image, display, clear_output
import seaborn as sns
import random

from diypilot_augmentation import lateral_shift_augmentation, horizontal_rotation_augmentation

class RoadDataset(Sequence):
    
    def __init__(self, csv_path, frames_path, column_names, batch_size=16, mode='train'):
        self.csv_path = csv_path
        self.frames_path = frames_path
        self.column_names = column_names
        try:
            print("Reading csv file: {}".format(csv_path))
            self.csv = pd.read_csv(csv_path, names=column_names)
        except:
            print("Failed to read csv file!")
            self.csv = pd.DataFrame(columns = column_names)
        self.batch_size = batch_size
        self.mode = mode
        self.indices = np.arange(len(self.csv))
        
        # Vehicle
        self.vehicle_turn_circle = 11.5
        self.vehicle_width = 1.8
        self.vehicle_turn_radius = (self.vehicle_turn_circle - self.vehicle_width)/2
        self.vehicle_max_swa = 500
        
        # Camera
        self.frame_hor_fov_deg = 69 # degree
        self.latency_compensation_frames = 3 # frames
        
        # ROI
        self.roi_y = 230
        self.roi_h = 170
        self.roi_x = 104
        self.roi_w = 640
        self.horizon_height = 215
        
        # Augmentation
        self.lateral_shift_range = 0.5 # meter
        self.hor_rotation_range = 5 # degree
        
        # Normalization
        #self.swa_range = 90.0 # degrees
        self.velocity_range = 250.0 # km/h
        
        print('Loaded dataset with ' + str(len(self.csv)) + ' samples')
        
    def __len__(self):
        return int(math.floor(len(self.csv) / float(self.batch_size)))
        
    def __getitem__(self, batch_index):
        return self.get_random_batch(batch_index)
    
    def get_random_batch(self, batch_index):
        batch_frames = []
        batch_velocities = []
        batch_swas = []
        
        for i in range(self.batch_size):
            
            sample_index = random.randint(0, len(self.csv)-1)
            #sample_index = 100
            sample = self.get_sample(sample_index, augment=True, crop=True, normalize=True)
            frame = sample['frame']
            v_vehicle = sample['v_vehicle']
            swa = sample['swa']

            batch_frames.append(frame)
            batch_velocities.append(v_vehicle)
            batch_swas.append(swa)
            
        batch_features = [np.array(batch_frames), np.array(batch_velocities)]
        batch_labels = np.array(batch_swas)
            
        #print(batch_labels)
            
        return (batch_features, batch_labels)
    
    def get_batch_size(self):
        return self.batch_size
    
    
    def get_sample(self, sample_index, augment=False, crop=False, normalize=False):
        csv = self.get_csv()
        csv = csv.iloc[sample_index]
        frame_path = self.frames_path + csv['filename']
        img = load_img(frame_path, color_mode='rgb')
        frame = img_to_array(img, dtype=np.uint8)
        v_vehicle = csv['speed']
        swa = csv['steering_wheel_angle']
        
        # augment here
        if augment:
            frame, swa = self.augment_sample(frame, swa, v_vehicle)

        # crop here
        if crop:
            frame = self.crop_to_roi(frame)
            
        # normalize here
        if normalize:
            #print("swa: {}, normed swa: {}".format(swa, self.normalize_swa(swa)))
            swa = self.normalize_swa(swa)
            v_vehicle = v_vehicle/self.velocity_range
        
        sample = {}
        sample['filename'] = csv['filename']
        sample['frame'] = frame
        sample['swa'] = swa
        sample['v_vehicle'] = v_vehicle
        sample['blink_l'] = csv["blink_l"]
        sample['blink_r'] = csv["blink_r"]
        
        
        return sample
        
        
    def augment_sample(self, frame, orig_swa, v_vehicle, lateral_shift=None, hor_rotation=None):
        
        time_to_center = 2 # seconds
        frame_physical_width = 1450.0/1088.0*5.0 # meter
        
        swa = orig_swa
        
        if lateral_shift is None:
            lateral_shift = random.uniform(-self.lateral_shift_range, self.lateral_shift_range)
        
        frame, swa = lateral_shift_augmentation(frame, swa, frame_physical_width, v_vehicle, time_to_center, self.vehicle_turn_radius, self.vehicle_max_swa, lateral_shift, self.horizon_height, verbose=False)
        
        time_to_recover = 0.5 # seconds
        
        if hor_rotation is None:
            hor_rotation = random.uniform(-self.hor_rotation_range, self.hor_rotation_range)
            
        frame, swa = horizontal_rotation_augmentation(frame, swa, self.frame_hor_fov_deg, v_vehicle, time_to_recover, self.vehicle_turn_radius, self.vehicle_max_swa, hor_rotation, verbose=False)
        
        return frame, swa
    
    def set_lateral_shift_range(self, lateral_shift_range):
        self.lateral_shift_range = lateral_shift_range
    
    
    def crop_to_roi(self, frame):
        
        crop_img = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w].copy()
        return crop_img
    
    
    def clean(self):
        csv = self.get_csv()
        print("Cleaning dataset. Starting with " + str(len(csv)) + " samples.")
        
        # Check if image files exist and are readable
        #missing_idxs = []
        #for idx in range(len(self)):
        #    try:
        #        self.get_frame(idx)
        #    except:
        #        missing_idxs.append(idx)
        #        print(self.get_csv(idx))
        #print("Removing " + str(len(missing_idxs)) + " rows for reason: Failed reading image file")
        #csv = csv.drop(missing_idxs)
        
        ## Remove samples whre swa is NaN
        #try:
        #    swa_nan_indices = csv[np.isnan(csv["steering_wheel_angle"])].index
        #    print("Removing " + str(len(swa_nan_indices)) + " rows for reason: SWA is NaN")
        #    csv = csv.drop(swa_nan_indices)
        #except:
        #    print("Failed dropping swa NaN")
        #    print(csv["steering_wheel_angle"])
        #    #for swa in csv["steering_wheel_angle"]:
        #    #    print(swa)
        
        
        # shift frames down to compensate for camera lag
        
        csv['filename'] = csv['filename'].shift(self.latency_compensation_frames, fill_value = "No frame")
        no_frame_indices = csv[csv["filename"] == "No frame"].index
        print("Removing " + str(len(no_frame_indices)) + " rows for reason: No frame after cam latency compensation")
        csv = csv.drop(no_frame_indices)
        
        # Remove Low speed samples
        low_speed_indices = csv[csv["speed"] < 25].index
        print("Removing " + str(len(low_speed_indices)) + " rows for reason: Low speed (< 25 km/h)")
        csv = csv.drop(low_speed_indices)
        
        # Remove high Steering wheel angle samples
        high_neg_swa_indices = csv[csv["steering_wheel_angle"] < -45].index
        high_pos_swa_indices = csv[csv["steering_wheel_angle"] > 45].index
        print("Removing " + str(len(high_neg_swa_indices) + len(high_pos_swa_indices)) + " rows for reason: High SWA (> +45/ < -45 deg)")
        csv = csv.drop(high_neg_swa_indices)
        csv = csv.drop(high_pos_swa_indices)
        
        # Remove samples with blinker on
        blink_left_indices = csv[csv["blink_l"] == 1].index
        print("Removing " + str(len(list(blink_left_indices))) + " rows for reason: Left blinker on")
        csv = csv.drop(blink_left_indices)
        
        blink_right_indices = csv[csv["blink_r"] == 1].index
        print("Removing " + str(len(blink_right_indices)) + " rows for reason: Right blinker on")
        csv = csv.drop(blink_right_indices)
        
        print(str(len(csv)) + " samples remaining.")
        
        self.csv = csv
        
    def balance(self):
        csv = self.get_csv()
        print("Balancing dataset. Starting with " + str(len(csv)) + " samples.")
        
        # Create 18 bins a 5 deg
        range_bins = [[5*i, 5*(i+1)] for i in range(-9, 9)]
        #print(range_bins)
        
        idx_bins = [[] for i in range(18)]
        #print(idx_bins)
        
        bin_counts = []
        
        for i, range_bin in enumerate(range_bins):
            range_indices = csv[(csv["steering_wheel_angle"] >= range_bin[0]) & (csv["steering_wheel_angle"] < range_bin[1])].index.to_list()
            idx_bins[i] = range_indices
            bin_counts.append(len(range_indices))
            
        #median_bin_count = int(np.median(bin_counts))
        #median_bin_count = int(min(bin_counts))
        median_bin_count = int(np.sum(sorted(bin_counts)[:5]))
            
        # Keep min_bin_sample_count samples in each bin, random sampling
        for i in range(len(idx_bins)):
            idx_bins[i] = random.sample(idx_bins[i], min(median_bin_count, len(idx_bins[i])))
        
        print(bin_counts)
        print(median_bin_count)
        
        balanced_indices = []
        for indices in idx_bins:
            balanced_indices = [*balanced_indices, *indices]
        #print(balanced_indices)
        
        balanced_csv = csv[csv.index.isin(balanced_indices)]
        
        print(str(len(balanced_csv)) + " samples remaining.")
        self.csv = balanced_csv
        
        
    def get_csv(self):
        return self.csv
    
    def get_mean_absolute_swa(self):
        swas = self.get_csv()['steering_wheel_angle']
        return np.mean(abs(swas))
    
    
    def get_mean_absolute_normalized_swa(self):
        swas = self.get_csv()['steering_wheel_angle']
        normed_swas = np.array([self.normalize_swa(swa) for swa in swas])
        return np.mean(abs(normed_swas))
    
    def normalize_swa(self, swa_deg):
        swa = swa_deg
        signed_log_swa = self.signed_log([swa], 2)[0]
        return signed_log_swa/4
    
    def denormalize_swa(self, swa_norm):
        swa = self.revert_signed_log([swa_norm*4], 2)[0]
        swa_deg = swa
        return swa_deg
    
    def signed_log(self, arr, zero_gap):
        arr2 = []
        for x in arr:
            if x < 0:
                x = -np.log(abs(x-zero_gap))
                x += np.log(zero_gap)
            else:
                x = np.log(abs(x+zero_gap))
                x -= np.log(zero_gap)
            arr2.append(x)
        return np.array(arr2)

    def revert_signed_log(self, arr, zero_gap):
        arr2 = []
        for x in arr:
            if x < 0:
                x -= np.log(zero_gap)
                x = -np.exp(abs(x))+zero_gap
            else:
                x += np.log(zero_gap)
                x = np.exp(abs(x))-zero_gap
            arr2.append(x)
        return np.array(arr2)
    
    ##########################################################################################
    
    
    
    
    
    
    ''' 
    
    
    
    def set_offset_range(self, offset_range):
        self.offset_range = offset_range
        
    def get_offset_multiplier(self):
        return self.offset_multiplier
    
    def get_pseudorandom_offset(self, sample_index):
        offset = ((self.some_offset_prime*sample_index) % (2*self.offset_range+1)) - self.offset_range
        return offset
        
    
    
    def on_epoch_end(self):
        # Shuffles indices after each epoch if in training mode
        self.rebuild_indices()
        #if self.mode == 'train':
        #    self.indices = np.arange(len(self.csv))
        #    np.random.shuffle(self.indices)
            
    def rebuild_indices(self):
        self.indices = np.arange(len(self.csv))
        if self.mode == 'train':
            #print("shuffling indices. Before:")
            #print(self.indices)
            np.random.shuffle(self.indices)
            #print("After:")
            #print(self.indices)
            
            
    def get_batch_labels(self, batch_index):
        # Fetch a batch of labels
        sample_indices = self.indices[(batch_index * self.batch_size):((batch_index + 1) * self.batch_size)]
        return np.array([self.get_label(sample_index) for sample_index in sample_indices])

    def get_batch_features(self, batch_index):
        # Fetch a batch of inputs
        #sample_indices = range(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)
        sample_indices = self.indices[(batch_index * self.batch_size):((batch_index + 1) * self.batch_size)]
        return np.array([self.get_cropped_frame(sample_index, lateral_shift = "random", offset = "random")[0] for sample_index in sample_indices])

    
        
    
        
    def pairplot(self):
        sns.pairplot(self.get_csv(), diag_kind="kde")
        
    
        
    
        
    
        
    def get_label(self, sample_index):
        original = self.get_csv(sample_index)['steering_wheel_angle']
        lateral_shift = self.get_pseudorandom_lateral_shift(sample_index)
        #print("Label: Random shift: {}px".format(shift))
        lateral_shift *= self.lateral_shift_multiplier
        #print("Label: Random shift: {}deg".format(shift))
        lateral_shift /= 90
        #print("Label: Random shift: {}units".format(shift))
        
        
        offset = self.get_pseudorandom_offset(sample_index)
        #print("Label: Random shift: {}px".format(shift))
        offset *= self.offset_multiplier
        #print("Label: Random shift: {}deg".format(shift))
        offset /= 90
        
        return original + lateral_shift + offset
    
    def get_swa(self, sample_index):
        return self.get_label(sample_index)
    
    def get_cropped_frame(self, index, lateral_shift, offset):
        img_arr, csv = self.get_frame(index)
        img_crop = self.crop_to_roi(img_arr)
        
        if lateral_shift == None:
            lateral_shift = 0
            
        if offset == None:
            offset = 0
        
        # If no explicit shift is given, apply "random" shift
        
        if lateral_shift == "random":
            lateral_shift = self.get_pseudorandom_lateral_shift(index)
            #print("Image: Random shift: {}px".format(shift))
        
        if offset == "random":
            offset = self.get_pseudorandom_offset(index)
        
        # Locate points of the documents or object which you want to transform
        # source:
        frame_width = self.roi_w
        frame_height = self.roi_h
        src_pt1 = [0 + offset, 0]
        src_pt2 = [frame_width + offset, 0]
        src_pt3 = [0 + lateral_shift + offset, frame_height]
        src_pt4 = [frame_width + lateral_shift + offset, frame_height]
        
        pts1 = np.float32([src_pt1, src_pt2, src_pt3, src_pt4]) 
        pts2 = np.float32([[0, 0], [frame_width, 0], [0, frame_height], [frame_width, frame_height]])

        # Apply Perspective Transform Algorithm 
        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        result = cv2.warpPerspective(img_crop/255., matrix, (frame_width, frame_height), borderMode=cv2.BORDER_WRAP)
        
        cropped_frame = (result*255).astype(np.uint8)
        
        # Add white areas to hide transform artefacts when adding shift and offset
        contours1 = np.array( [[0,0], [0,170], [100, 170], [100, 0]] )
        cropped_frame = cv2.fillPoly(cropped_frame, pts =[contours1], color=(255,255,255))
        contours1 = np.array( [[847,0], [847,170], [747, 170], [747, 0]] )
        cropped_frame = cv2.fillPoly(cropped_frame, pts =[contours1], color=(255,255,255))
        
        return cropped_frame, csv
        
    def get_frame(self, index):
        csv = self.get_csv(index)
        frame_path = self.frames_path + csv['filename']
        img = load_img(frame_path)
        img_arr = img_to_array(img, dtype=np.uint8)
        
        return img_arr, csv
    
    def get_velocity(self, sample_index):
        return self.get_csv(sample_index)['speed']
    
    def get_indicator_left(self, sample_index):
        blink_l = self.get_csv(sample_index)['blink_l']
        if blink_l == 1:
            return True
        return False
    
    def get_indicator_right(self, sample_index):
        blink_r = self.get_csv(sample_index)['blink_r']
        if blink_r == 1:
            return True
        return False
    
    def get_timestamp(self, sample_index):
        return self.get_csv(sample_index)['filename']
    
    def swa_to_pathradius(self, swa):
        # steering wheel angle (swa) in degrees
        # path radius in meters
        
        wendekreis = 11.5
        fahrzeugbreite = 1.8
        max_swa = 500
        
        min_radius = (wendekreis - fahrzeugbreite)/2
        
        if (swa == 0):
            return 9999
        else:
            radius = max_swa/swa * min_radius
            return radius
            
    '''
        
   
        
