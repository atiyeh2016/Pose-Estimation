#%% Importing Librarires
from __future__ import print_function, division
from PoseDataset import PoseLandmarksDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from ToTensor import ToTensor
from Rotation import Rotation
from Scaling import Scaling
from Shearing import Shearing
from Translation import Translation
from Brighness import Brightness
from Contrast import Contrast
import csv

#%% Dataset Preparing
def joint():    
    flag = 1
    # Original Dataset
    original = PoseLandmarksDataset('joints_train.csv',
                                            r'_images')
    
    # Rotation
    rotated = PoseLandmarksDataset('joints_train.csv',
                                           r'_images',
                                           transform=transforms.Compose([Rotation(show = flag)]))
    
    # Scaling
    scaled = PoseLandmarksDataset('joints_train.csv',
                                           r'_images',
                                           transform=transforms.Compose([Scaling(show = flag)]))
    
    # Shearing
    sheared = PoseLandmarksDataset('joints_train.csv',
                                               r'_images',
                                               transform=transforms.Compose([Shearing(show = flag)]))
    
    # Brightness
    brightness_changed = PoseLandmarksDataset('joints_train.csv',
                                                    r'_images',
                                                    transform=transforms.Compose([Brightness(show = flag)]))
    
    # Contrast
    contrast_changed = PoseLandmarksDataset('joints_train.csv',
                                                    r'_images',
                                                    transform=transforms.Compose([Contrast(show = flag)]))
    
    # Concating datasets
    final_dataset = torch.utils.data.ConcatDataset((original, rotated, scaled,
                                                    sheared, brightness_changed, contrast_changed))

    # annotation preparing
    annotations = []
    for idx in range(8394):
#        print(idx)
        active_img = final_dataset[idx]
        img_name = f'img{idx:04d}'
        annotations.append((img_name, active_img['landmarks']))
        
    return annotations

#%% Making CSV
annotations = joint()
with open('joints_aug.csv', 'w', newline='') as csvfile:
    fieldnames = ['img_name',
                  'joint_1_x', 'joint_1_y',
                  'joint_2_x', 'joint_2_y',
                  'joint_3_x', 'joint_3_y',
                  'joint_4_x', 'joint_4_y',
                  'joint_5_x', 'joint_5_y',
                  'joint_6_x', 'joint_6_y',
                  'joint_7_x', 'joint_7_y',
                  'joint_8_x', 'joint_8_y',
                  'joint_9_x', 'joint_9_y',
                  'joint_10_x', 'joint_10_y',
                  'joint_11_x', 'joint_11_y',
                  'joint_12_x', 'joint_12_y',
                  'joint_13_x', 'joint_13_y',
                  'joint_14_x', 'joint_14_y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    # data assignment
    for jointt in annotations:
        new_img_joints = {'img_name': jointt[0],
                         'joint_1_x': jointt[1][0][0],'joint_1_y': jointt[1][0][1],
                         'joint_2_x': jointt[1][1][0],'joint_2_y': jointt[1][1][1],
                         'joint_3_x': jointt[1][2][0],'joint_3_y': jointt[1][2][1],
                         'joint_4_x': jointt[1][3][0],'joint_4_y': jointt[1][3][1],
                         'joint_5_x': jointt[1][4][0],'joint_5_y': jointt[1][4][1],
                         'joint_6_x': jointt[1][5][0],'joint_6_y': jointt[1][5][1],
                         'joint_7_x': jointt[1][6][0],'joint_7_y': jointt[1][6][1],
                         'joint_8_x': jointt[1][7][0],'joint_8_y': jointt[1][7][1],
                         'joint_9_x': jointt[1][8][0],'joint_9_y': jointt[1][8][1],
                         'joint_10_x': jointt[1][9][0],'joint_10_y': jointt[1][9][1],
                         'joint_11_x': jointt[1][10][0],'joint_11_y': jointt[1][10][1],
                         'joint_12_x': jointt[1][11][0],'joint_12_y': jointt[1][11][1],
                         'joint_13_x': jointt[1][12][0],'joint_13_y': jointt[1][12][1],
                         'joint_14_x': jointt[1][13][0],'joint_14_y': jointt[1][13][1]}
        writer.writerow(new_img_joints)
