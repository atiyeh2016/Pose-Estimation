#%% Importing Librarires
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PoseDataset import PoseLandmarksDataset
from ToTensor import ToTensor
from Rotation import Rotation
from Scaling import Scaling

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#%% Displaying Information and Showing image with landmarks

# Displaying Information
#joints = pd.read_csv('joints_train.csv')
#n = 25
#img_name = joints.iloc[n, 0]
#landmarks = joints.iloc[n, 1:]
#landmarks = np.asarray(landmarks)
#landmarks = landmarks.astype('float').reshape(-1, 2)
#
#print('Image name: {}'.format(img_name))
#print('Landmarks shape: {}'.format(landmarks.shape))
#print('First 4 Landmarks: {}'.format(landmarks[:4]))
#
## Showing image with landmarks
def show_landmarks(image, landmarks=[]):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
#
#plt.figure()
#show_landmarks(io.imread(os.path.join('_images', img_name)), landmarks)
#plt.show()

#%% Making an instant of dataset
#pose_dataset = PoseLandmarksDataset('joints_train.csv',
#                                   'images')
#
#fig = plt.figure()
#
#for i in range(len(pose_dataset)):
#    sample = pose_dataset[i]
#
#    print(i, sample['image'].shape, sample['landmarks'].shape)
#
#    ax = plt.subplot(2, 2, i + 1)
#    plt.tight_layout()
#    ax.set_title('Sample #{}'.format(i))
#    ax.axis('off')
#    show_landmarks(**sample)
#
#    if i == 3:
#        plt.show()
#        break

#%% Dataloader

# Helper function to show a batch
#def show_landmarks_batch(sample_batched):
#    """Show image with landmarks for a batch of samples."""
#    images_batch, landmarks_batch = \
#            sample_batched['image'], sample_batched['landmarks']
#    batch_size = len(images_batch)
#    im_size = images_batch.size(2)
#    grid_border_size = 2
#
#    grid = utils.make_grid(images_batch)
#    plt.imshow(grid.numpy().transpose((1, 2, 0)))
#
#    for i in range(batch_size):
#        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
#                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
#                    s=10, marker='.', c='r')
#
#        plt.title('Batch from dataloader')


def main():
    transformed_dataset = PoseLandmarksDataset('joints_train.csv',
                                    r'_images', transform=transforms.Compose([Rotation(show=1)]))
#    transformed_dataset = PoseLandmarksDataset('joints_train.csv',
#                                                r'_images',
#                                                transform=transforms.Compose([Scaling(show=1)]))
#    transformed_dataset = PoseLandmarksDataset('joints.csv',
#                                                r'images')
    
    test_image = transformed_dataset[351]
    fig = plt.figure()
#    sample = pose_dataset[5]
    show_landmarks(**test_image)
    

#    dataloader = DataLoader(transformed_dataset, batch_size=4,
#                        shuffle=True, num_workers=0)
#
#
#    for i_batch, sample_batched in enumerate(dataloader):
#        print(i_batch, sample_batched['image'].size(),
#              sample_batched['landmarks'].size())
#    
#        # observe 4th batch and stop.
#        if i_batch == 3:
#            plt.figure()
#            show_landmarks_batch(sample_batched)
#            plt.axis('off')
#            plt.ioff()
#            plt.show()
#            break

#%% main
if __name__ == "__main__":
    main()

