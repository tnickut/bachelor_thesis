import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} to control the verbosity
import cv2
import numpy as np
import time
import math
from ultralytics import *
from collections import deque

from multibin_pipeline import *

from libs.bbox3d_utils import *
from libs.Plotting import *


image = 'a2'
p2 = np.array([
    [1920/(2*math.tan(120/2)), 0.0, 960, 0.0], 
    [0.0, 1920/(2*math.tan(120/2)), 540, 0.0], 
    [0.0, 0.0, 1.0, 0.0]])
frame = cv2.imread('./gta_images/' + image + '.png')

meanpool_layer = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")

img = frame.copy() 
img2 = frame.copy()
img3 = frame.copy()

## process 2D and 3D boxes
bboxes2d = process2D(img2)
for bbox2d in bboxes2d:
    bbox_coords, scores, classes = bbox2d
    bbox_ = bbox_coords.cpu().detach().numpy()
    result_3d = process3D(img, p2, bbox2d, poolingLayer=meanpool_layer, poolingMinWidth=500)
    if result_3d is None:
        continue
    rotation_y, theta_ray, alpha, dimensions = result_3d
    plot3d(img3, p2, bbox_, dimensions, alpha, theta_ray)


# Display the frame
cv2.imshow("2D", img2)
cv2.imshow("3D", img3)

# wait for key press
cv2.waitKey(0)