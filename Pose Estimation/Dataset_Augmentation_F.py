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

#%% Dataset Preparing
def joint():    
    flag = 1
    # Original Dataset
    original = PoseLandmarksDataset('joints_train.csv',
                                            r'_images',
                                            transform=transforms.Compose([ToTensor()]))
    
    # Rotation
    rotated = PoseLandmarksDataset('joints_train.csv',
                                           r'_images',
                                           transform=transforms.Compose([Rotation(show = flag), ToTensor()]))
    
    # Scaling
    scaled = PoseLandmarksDataset('joints_train.csv',
                                           r'_images',
                                           transform=transforms.Compose([Scaling(show = flag), ToTensor()]))
    
    # Shearing
    sheared = PoseLandmarksDataset('joints_train.csv',
                                               r'_images',
                                               transform=transforms.Compose([Shearing(show = flag), ToTensor()]))
    
    # Brightness
    brightness_changed = PoseLandmarksDataset('joints_train.csv',
                                                    r'_images',
                                                    transform=transforms.Compose([Brightness(show = flag), ToTensor()]))
    
    # Contrast
    contrast_changed = PoseLandmarksDataset('joints_train.csv',
                                                    r'_images',
                                                    transform=transforms.Compose([Contrast(show = flag), ToTensor()]))
    
    # Concating datasets
    final_dataset = torch.utils.data.ConcatDataset((original, rotated, scaled,
                                                    sheared, brightness_changed, contrast_changed))

    # annotation preparing
    annotations = []
    for idx in range(10): #8394
        print(idx)
        active_img = final_dataset[idx]
        img_name = f'{idx:04d}'
        annotations.append((img_name, active_img['landmarks']))
        
    return annotations

#%% Making CSV
annotations = joint()