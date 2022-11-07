from Data_preprocessing import Preprocessing
import scipy.io as io
import numpy as np
import pickle

P = Preprocessing((220,220,3))
mat = io.loadmat('joints.mat')
mat = np.array(mat['joints'])

annotations = []

for i in range(25):
    name = 'im%04d.jpg' % (i+1,)
    joints = mat[0:2,:,i]
    P.process_img(name,joints)
    annotations.append()


#with open('data.pickle', 'wb') as f:
#    pickle.dump(P.data, f)