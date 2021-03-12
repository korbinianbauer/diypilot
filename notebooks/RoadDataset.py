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
        
        # ROI
        self.roi_y = 230
        self.roi_h = 170
        self.roi_x = 0
        self.roi_w = 848
        
        self.some_prime = 131
        self.shift_range = 100
        self.shift_multiplier = 0.2 # degrees per pixel shift
        
        print('Loaded dataset with ' + str(len(self.csv)) + ' samples')
        
    def set_shift_range(self, shift_range):
        self.shift_range = shift_range
        
    def __len__(self):
        return int(math.floor(len(self.csv) / float(self.batch_size)))
    
    def get_batch_size(self):
        return self.batch_size
    
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
        return np.array([self.get_cropped_frame(sample_index, shift = "random")[0] for sample_index in sample_indices])

    def __getitem__(self, batch_index):
        batch_x = self.get_batch_features(batch_index)
        batch_y = self.get_batch_labels(batch_index)
        return batch_x, batch_y
        
    def get_csv(self, index=None):
        if index == None:
            return self.csv
        else:
            #print('Accessing csv index ' + str(index))
            return self.csv.iloc[index]
        
    def pairplot(self):
        sns.pairplot(self.get_csv(), diag_kind="kde")
        
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
        
        # Remove samples with left blinker on
        blink_left_indices = csv[csv["blink_l"] == 1].index
        print("Removing " + str(len(list(blink_left_indices))) + " rows for reason: Left blinker on")
        csv = csv.drop(blink_left_indices)
        
        blink_right_indices = csv[csv["blink_r"] == 1].index
        print("Removing " + str(len(blink_right_indices)) + " rows for reason: Right blinker on")
        csv = csv.drop(blink_right_indices)
        
        print(str(len(csv)) + " samples remaining.")
        
        self.csv = csv
        self.rebuild_indices()
        
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
        self.rebuild_indices()
        
    def normalize(self):
        
        # map steering wheel angles of range [-90 90] to [-1, 1]
        self.csv["steering_wheel_angle"] /= 90.0
        
    def get_label(self, sample_index):
        original = self.get_csv(sample_index)['steering_wheel_angle']
        shift = ((self.some_prime*sample_index) % (2*self.shift_range+1)) - self.shift_range
        #print("Label: Random shift: {}px".format(shift))
        shift *= self.shift_multiplier
        #print("Label: Random shift: {}deg".format(shift))
        shift /= 90
        #print("Label: Random shift: {}units".format(shift))
        
        return original + shift
    
    def get_swa(self, sample_index):
        return self.get_label(sample_index)
    
    def get_cropped_frame(self, index, shift):
        img_arr, csv = self.get_frame(index)
        img_crop = self.crop_to_roi(img_arr)
        
        if shift == None:
            return img_crop, csv
        
        # If no explicit shift is given, apply "random" shift
        
        if shift == "random":
            shift = ((self.some_prime*index) % (2*self.shift_range+1)) - self.shift_range
            #print("Image: Random shift: {}px".format(shift))
            
        # Locate points of the documents or object which you want to transform
        # source:
        frame_width = self.roi_w
        frame_height = self.roi_h
        src_pt1 = [0, 0]
        src_pt2 = [frame_width, 0]
        src_pt3 = [0 + shift, frame_height]
        src_pt4 = [frame_width + shift, frame_height]
        
        pts1 = np.float32([src_pt1, src_pt2, src_pt3, src_pt4]) 
        pts2 = np.float32([[0, 0], [frame_width, 0], [0, frame_height], [frame_width, frame_height]])

        # Apply Perspective Transform Algorithm 
        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        result = cv2.warpPerspective(img_crop/255., matrix, (frame_width, frame_height), borderMode=cv2.BORDER_WRAP)
        
        cropped_frame = (result*255).astype(np.uint8)
        
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
        
    @staticmethod
    def crop_to_roi(frame):
        roi_y = 230
        roi_h = 170
        roi_x = 0
        roi_w = 848
        crop_img = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        return crop_img
        
