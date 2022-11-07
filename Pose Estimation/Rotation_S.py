import imgaug.augmenters as iaa
from random import randrange
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage

class Rotation(object):
    
    def __init__(self, show = []):
        self.show = show
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        landmarks /= 220
        
        if self.show:
            Keypoints = [Keypoint(x*220,y*200) for (x,y) in landmarks]
        else:
            Keypoints = [Keypoint(x,y) for (x,y) in landmarks]           
        
        kps = KeypointsOnImage(Keypoints, shape=image.shape)
        
        StochasticParameter = randrange(-360,360)
        seq = iaa.Sequential([iaa.geometric.Rotate(rotate = StochasticParameter)])
        image_aug, landmarks_aug = seq(image=image, keypoints=kps)
        
        landmarks_aug_np = np.zeros(landmarks.shape)
        for i, k in enumerate(landmarks_aug):
            landmarks_aug_np[i,0] = np.array(k.x)
            landmarks_aug_np[i,1] = np.array(k.y)
            
        w1 = (220-len(image_aug[0]))//2
        h1 = (220-len(image_aug))//2
        
        landmarks_aug_np_p = landmarks_aug_np.copy()
        
        landmarks_aug_np_p[:,0] += w1
        landmarks_aug_np_p[:,1] += h1
        
        h2 = h1 + len(image_aug)
        w2 = w1 + len(image_aug[0])
    
        img_ = np.zeros([220,220,3]).astype(np.uint8)
        img_[h1:h2,w1:w2,:] = image_aug[:][:][:]

        return {'image': img_, 'landmarks': landmarks_aug_np_p}
