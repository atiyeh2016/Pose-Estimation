import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage

class Translation(object):
    
     def __init__(self, show = []):
        self.show = show

     def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if self.show:
            Keypoints = [Keypoint(x*220,y*200) for (x,y) in landmarks]
        else:
            Keypoints = [Keypoint(x,y) for (x,y) in landmarks]         
        
        kps = KeypointsOnImage(Keypoints, shape=image.shape)
        
        seq = iaa.Sequential([iaa.geometric.Affine(
                            translate_px = {"x": (0, len(image)), "y": (0, len(image))})])
        image_aug, landmarks_aug = seq(image=image, keypoints=kps)
        
        landmarks_aug_np = np.zeros(landmarks.shape)
        for i, k in enumerate(landmarks_aug):
            landmarks_aug_np[i,0] = np.array(k.x)
            landmarks_aug_np[i,1] = np.array(k.y)
            
        return {'image': image_aug, 'landmarks': landmarks_aug_np}

    