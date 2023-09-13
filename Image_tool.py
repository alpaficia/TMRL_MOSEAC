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
        self.size = (4, 4)
        self.len_of_image_history = 4
        self.img_height = 64
        self.polygons_line = np.array([[(1,self.img_height),(1,int(0.6 * self.img_height)),(25,20),(35,20),(64,int(0.6 * self.img_height)),(64, self.img_height)]])
        self.polygons_car = np.array([[(10,int(0.9 * self.img_height)),(15,int(0.6 * self.img_height)),(39,int(0.6 * self.img_height)),(54,int(0.9 * self.img_height))]])
    
    def edge_detection(self, image):
        img_edge = np.zeros([self.len_of_image_history, 64, 64])
        for i in range(self.len_of_image_history):
            edge = cv2.Canny(image[i], self.edge_low_param, self.edge_up_param)
            edge_segment_line = self.do_segment(self.polygons_line, edge)
            edge_segment_car = self.do_segment(self.polygons_car, edge)
            hough = cv2.HoughLinesP(edge_segment_line, 0.8, np.pi/180, 15)
            lines = self.calculate_lines(image[i], hough)
            lines_visualize = self.visualize_lines(image[i], lines)
            if lines_visualize is not None:
                img_edge[i] = cv2.addWeighted(edge_segment_car,1,lines_visualize,1,0)
            else:
                img_edge[i] = edge_segment_car
        return img_edge
    
    def do_segment(self, polygons, frame):
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask,polygons,255) 
        segment = cv2.bitwise_and(frame,mask) 
        return segment
    
    def calculate_lines(self, frame, lines):
        left = []
        right = []
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line.reshape(4)
                parameters = np.polyfit((x1,x2), (y1,y2), 1)
                slope = parameters[0] 
                y_intercept = parameters[1]
                if slope < 0:
                    left.append((slope,y_intercept))
                else:
                    right.append((slope,y_intercept))
            left_avg = np.average(left,axis=0)
            right_avg = np.average(right,axis=0)
            left_line = self.calculate_coordinate(frame,parameters=left_avg)
            right_line = self.calculate_coordinate(frame, parameters=right_avg)
            return np.array([left_line,right_line])
        else:
            return None
    
    def calculate_coordinate(self, frame, parameters):
        slope, y_intercept = parameters
        y1 = frame.shape[0]
        y2 = int(0.6 * y1)
        x1 = int((y1-y_intercept)/slope)
        x2 = int((y2-y_intercept)/slope)
        return np.array([x1,y1,x2,y2])
    
    def visualize_lines(self, frame,lines):
        lines_visualize = np.zeros_like(frame)
        if lines is not None:
            for x1,y1,x2,y2 in lines:
                cv2.line(lines_visualize,(x1,y1),(x2,y2),(255,255,255),2)
            line_gray = cv2.cvtColor(lines_visualize, cv2.COLOR_BGR2GRAY)
            return line_gray
        else:
            return None
    
    def image_unified(self, image):
        return image / 255.0  # converage from (0, 255) to (0, 1)
    
    def image_average_pooling(self, image):
        img_pooling = np.zeros([self.len_of_image_history, 16, 16])
        for i in range(self.len_of_image_history):
            img_pooling[i] = skimage.measure.block_reduce(image[i], self.size, np.mean)
        return img_pooling
    
    def image_reshape_unify(self, image):
        return image.reshape(1024,)
