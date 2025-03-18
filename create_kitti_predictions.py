import os
import cv2
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} to control the verbosity
import numpy as np

from utils import compute_theta_ray, preprocess_image, parse_label, decode_output, get_P2
from multibin_pipeline import *

image_dir = './test_images'
label_dir = './test_labels'

def compute_xyz_from_bbox(P2, bbox_coords, dimensions):
    """ Calculation of (x, y, z) from the 2D bounding box and the KITTI camera matrix P2 accoding to "Vision meets Robotics" by Geiger et al. """
    left, top, right, bottom = bbox_coords
    obj_height_real, _, _ = dimensions

    # Calculate the center of the bounding box:
    u = (left + right) / 2
    v = (bottom + top) / 2
    x = np.array([u, v, 1]) # Here it is assumed that the bottom edge stands on the ground.

    # Extract the intrinsic matrix P from the projection matrix P2
    P = P2[:, :3]

    # Calculate the normalised direction vector y
    y = np.linalg.inv(P).dot(x)

    # Estimate the depth Z using the aspect ratio (f_y corresponds to the focal length in the y direction)
    Z = (P2[1, 1] * obj_height_real) / (bottom - top)

    return y * Z.item()


image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
for image_path in image_files:
    image_name = os.path.basename(image_path)
    image_id = os.path.splitext(image_name)[0]
    print("Current id: ", image_id)
    
    frame = cv2.imread(image_path)
    img = frame.copy() 
    img2 = frame.copy() 

    txt_filename = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join('kitti_predictions', txt_filename)

    bboxes2d = process2D(img2)
    with open(txt_path, 'w') as f:
        for bbox2d in bboxes2d:
            p2 = get_P2(image_id)
            prediction_3d = process3D(img, p2, bbox2d)
            if prediction_3d is None:
                continue
            bbox_coords, scores, classes = bbox2d
            left, top, right, bottom = bbox_coords
            rotation_y, theta_ray, alpha, dimensions = prediction_3d
            height, width, length = dimensions
            x, y, z = compute_xyz_from_bbox(p2, bbox_coords, dimensions)

            # Write the data in KITTI format. The last value is the score for the average precision calculation, which is not standard in KITTI.
            line = f"{get_yolo_classes_matched()[int(classes)]} {0} {0} {alpha:.3f} {left:.3f} {top:.3f} {right:.3f} {bottom:.3f} {height:.3f} {width:.3f} {length:.3f} {x:.3f} {y:.3f} {z:.3f} {rotation_y:.3f} {scores.item():.3f}\n"
            f.write(line)