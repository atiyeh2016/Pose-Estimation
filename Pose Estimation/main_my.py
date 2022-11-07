#%%
from Data_preprocessing_my import Preprocessing
import scipy.io as io
import numpy as np
import csv

#%% # data preprocessing
P = Preprocessing((220,220,3))
mat = io.loadmat('joints.mat')
mat = np.array(mat['joints'])


for i in range(2000):
    name = 'im%04d.jpg' % (i+1,)
    joints = mat[0:2,:,i]
    P.process_img(name,joints)
    
#%% saving as csv
#naking headers
with open('joints_rotate.csv', 'w', newline='') as csvfile:
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
    for joint in P.annotations:
        new_img_joints = {'img_name': joint[0],
                         'joint_1_x': joint[1][0][0],'joint_1_y': joint[1][1][0],
                         'joint_2_x': joint[1][0][1],'joint_2_y': joint[1][1][1],
                         'joint_3_x': joint[1][0][2],'joint_3_y': joint[1][1][2],
                         'joint_4_x': joint[1][0][3],'joint_4_y': joint[1][1][3],
                         'joint_5_x': joint[1][0][4],'joint_5_y': joint[1][1][4],
                         'joint_6_x': joint[1][0][5],'joint_6_y': joint[1][1][5],
                         'joint_7_x': joint[1][0][6],'joint_7_y': joint[1][1][6],
                         'joint_8_x': joint[1][0][7],'joint_8_y': joint[1][1][7],
                         'joint_9_x': joint[1][0][8],'joint_9_y': joint[1][1][8],
                         'joint_10_x': joint[1][0][9],'joint_10_y': joint[1][1][9],
                         'joint_11_x': joint[1][0][10],'joint_11_y': joint[1][1][10],
                         'joint_12_x': joint[1][0][11],'joint_12_y': joint[1][1][11],
                         'joint_13_x': joint[1][0][12],'joint_13_y': joint[1][1][12],
                         'joint_14_x': joint[1][0][13],'joint_14_y': joint[1][1][13],}
        writer.writerow(new_img_joints)
