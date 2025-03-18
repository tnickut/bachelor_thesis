import os
import tensorflow as tf
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from utils import compute_theta_ray, preprocess_image, parse_label, decode_output, get_P2

# Load 2D model
yolo_classes = ['Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck']
bbox2d_model = YOLO('yolov8n.pt')  # load yolov8 smallest model (n) for object detection
bbox2d_model.overrides['conf'] = 0.9  # NMS confidence threshold
bbox2d_model.overrides['iou'] = 0.45  # NMS IoU threshold
bbox2d_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
bbox2d_model.overrides['max_det'] = 1000  # maximum number of detections per image
bbox2d_model.overrides['classes'] = 0,1,2,3,5,7 ## define classes (these are the indexes of the classes of the variable yolo_classes)

# Load 3D model
DIMS_AVG = {
    'Car': np.array([1.52131309, 1.64441358, 3.85728004]),
    'Van': np.array([2.18560847, 1.91077601, 5.08042328]),
    'Truck': np.array([3.07044968,  2.62877944, 11.17126338]),
    'Pedestrian': np.array([1.75562272, 0.67027992, 0.87397566]),
    'Person_sitting': np.array([1.28627907, 0.53976744, 0.96906977]),
    'Cyclist': np.array([1.73456498, 0.58174006, 1.77485499]),
    'Tram': np.array([3.56020305,  2.40172589, 18.60659898])
}
yolo_class_names_matched = [
    cls.replace('Person', 'Pedestrian').replace('Bicycle', 'Cyclist').replace('Motorcycle', 'Cyclist')
    for cls in yolo_classes
]
n_bins = 6
input_shape = (224, 224, 3)
trained_classes_3d = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']

model = tf.keras.models.load_model('./orientation_model.h5', compile=False)



def process2D(image, track = True):
    bboxes = []
    results = bbox2d_model.predict(image, verbose=False)  # predict on an image
    for predictions in results:
        if predictions is None:
            continue  # Skip this image if YOLO fails to detect any objects
        if predictions.boxes is None:
            continue  # Skip this image if there are no boxes or masks

        for bbox in predictions.boxes: 
            for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                xmin    = bbox_coords[0]
                ymin    = bbox_coords[1]
                xmax    = bbox_coords[2]
                ymax    = bbox_coords[3]
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                bboxes.append([bbox_coords, scores, classes])

    return bboxes



def process3D(img, p2, bbox2d, poolingLayer=None, poolingMinWidth=0):
    bbox_coords, scores, yolo_class, *id_ = bbox2d if len(bbox2d) == 4 else (*bbox2d, None)
    if 0 <= int(yolo_class) < len(yolo_class_names_matched) and (yolo_class_names_matched[int(yolo_class)] in trained_classes_3d):
        xmin = bbox_coords[0]
        ymin = bbox_coords[1]
        xmax = bbox_coords[2]
        ymax = bbox_coords[3]
        bbox_ = [int(xmin), int(ymin), int(xmax), int(ymax)]

        crop = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
        crop = cv2.resize(crop, (224, 224))
        crop = crop / 255.0

        tensor = np.expand_dims(crop, axis=0).astype(np.float32)

        if poolingLayer is not None and poolingMinWidth < (xmax - xmin):
            tensor = poolingLayer(tensor)
            pooled_image = np.squeeze(tensor)
            #cv2.imshow("Original Crop", (crop * 255).astype(np.uint8))
            #cv2.imshow("Max Pooled Crop", (pooled_image * 255).astype(np.uint8))
            #cv2.waitKey(0)

        prediction = model.predict(tensor)

        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        alpha, dimensions = decode_output(prediction, n_bins, DIMS_AVG, yolo_class_names_matched[int(yolo_class)])
        theta_ray = compute_theta_ray(center_x, center_y, p2)
        rotation_y = alpha + theta_ray
        return rotation_y, theta_ray, alpha, dimensions


def get_yolo_classes_matched():
    return yolo_class_names_matched