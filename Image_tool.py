# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:15:18 2023

@author: Dong
"""

import cv2
import skimage
import numpy as np

class image_tool(object):
    
    def __init__(self):
        self.edge_low_param = 100
        self.edge_up_param = 200
        self.size = (8, 8)
    
    def edge_detection(self, image):
        image = image.reshape(64, 64, 3)  # the obs image is (1, 64, 64, 3), need to be reshape to a handleable RGB format matrix
        return cv2.Canny(image, self.edge_low_param, self.edge_up_param)
    
    def image_unified(self, image):
        return image / 255.0  # converage from (0, 255) to (0, 1)
    
    def image_average_pooling(self, image):
        return skimage.measure.block_reduce(image, self.size, np.mean)
    
    def image_reshape_unify(self, image):
        return image.reshape(64,)
