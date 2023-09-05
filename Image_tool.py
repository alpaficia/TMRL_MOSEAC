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
        self.len_of_image_history = 4
    
    def edge_detection(self, image):
        img_edge = np.zeros([self.len_of_image_history, 64, 64])
        for i in range(self.len_of_image_history):
            img_edge[i] = cv2.Canny(image[i], self.edge_low_param, self.edge_up_param)
        return img_edge
    
    def image_unified(self, image):
        return image / 255.0  # converage from (0, 255) to (0, 1)
    
    def image_average_pooling(self, image):
        img_pooling = np.zeros([self.len_of_image_history, 8, 8])
        for i in range(self.len_of_image_history):
            img_pooling[i] = skimage.measure.block_reduce(image[i], self.size, np.mean)
        return img_pooling
    
    def image_reshape_unify(self, image):
        return image.reshape(256,)
