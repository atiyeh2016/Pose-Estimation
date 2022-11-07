import matplotlib
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import scipy.io as io
import scipy.misc

class Preprocessing:
    
    
    def __init__(self, im_size):
        self.im_size = im_size
        self.data = []
        self.annotations = []
        
    def active_img(self,img):
        self.h = len(img)
        self.w = len(img[0])
        self.h1 = (self.im_size[0]-self.h)//2
        self.h2 = self.h1 + self.h
        self.w1 = (self.im_size[1]-self.w)//2
        self.w2 = self.w1 + self.w
    
    def crop_img(self, img):
        img_ = np.zeros(self.im_size)
        img_[self.h1:self.h2,self.w1:self.w2,:] = img[:][:][:]
        return img_
    
    def normalize_joints(self, joints):
        joints_ = joints + [[self.w1],[self.h1]]
        return joints_
    
    def read_img(self,img_name):
        img = mpimg.imread(img_name) 
        return img
    
    def save_img(self, img_, img_name):
        matplotlib.image.imsave('_' + img_name, img_/255)        

    def process_img(self,img_name,joints):
        img = self.read_img(img_name)
        self.active_img(img)
        img_ = self.crop_img(img)
        joints_ = self.normalize_joints(joints)
        self.save_img(img_, img_name)
        self.data.append((img_,joints_))
        self.annotations.append((img_name,joints_))

    
if __name__ == "__main__" :
    # Read Images 
    img = mpimg.imread('im0001.jpg') 
      
    # Output Images 
    plt.imshow(img)
    plt.show()
    
    mat = io.loadmat('joints.mat')
    mat = np.array(mat['joints'])
    joints = mat[0:2,:,0]
            
    P = Preprocessing((220,220,3))
    img_name = 'im0001.jpg'
    img_, joints_ = P.process_img(img_name,joints)
    
    plt.imshow(img_/255)
    plt.plot(joints_[0][:],joints_[1][:],'o')
    
    img2 = mpimg.imread('_im0001.jpg') 