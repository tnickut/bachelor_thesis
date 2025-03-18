import numpy as np
import cv2
import tensorflow as tf
import os

def compute_theta_ray(center_x, center_y, P2):
    """Compute the ray angle (theta_ray) from the 2D bounding box center."""
    K = P2[:3, :3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (center_x - cx) / fx
    return np.arctan2(x, 1.0)

def preprocess_image(img_path, box, target_size=(224, 224)):
    """Crop and resize image based on 2D bounding box."""
    img = cv2.imread(img_path)
    left, top, right, bottom = map(int, box)
    crop = img[top:bottom, left:right]
    crop = cv2.resize(crop, target_size)
    return crop / 255.0  # Normalize to [0,1]

def parse_label(label_path):
    """Parse KITTI label file and filter valid objects."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    objects = []
    for line in lines:
        parts = line.strip().split()
        obj_type = parts[0]
        trunc, occ, alpha = float(parts[1]), int(parts[2]), float(parts[3])
        box = list(map(float, parts[4:8]))
        dims = list(map(float, parts[8:11]))
        position = list(map(float, parts[11:14]))
        rot_y = float(parts[14])
        score = float(parts[15]) if len(parts) == 16 else 1.0
        objects.append({
            'type': obj_type,
            'truncation': trunc,
            'occlusion': occ,
            'alpha': alpha,
            'box': box,
            'dims': dims,
            'pos': position,
            'rot_y': rot_y,
            'score': score # Nur für AP Berechnung notwendig, anderweitig irrelevant. Kein KITTI Standard
        })
    return objects

def decode_output(prediction, n_bins, dims_avg, obj_type):
    """Convert model output to orientation angle and dimensions"""
    
    # 1.) Calculate bin center points
    bin_centers = np.linspace(-np.pi, np.pi, num=n_bins, endpoint=False)

    # 2.) Extract the output of the three branches
    conf_logits = prediction[0, :n_bins] # 1st branch -> Confidences of the individual bins
    cos_deltas = prediction[0, n_bins:2*n_bins] # 2nd branch -> cosine deltas of the individual bins
    sin_deltas = prediction[0, 2*n_bins:3*n_bins] # 2nd branch -> sine deltas of the individual bins
    dims_residual = prediction[0, 3*n_bins:] # 3rd branch -> residuals for the object dimensions

    # 2) L2 normalisation for residual angles of each bin
    norm_factor = np.sqrt(cos_deltas**2 + sin_deltas**2)
    cos_deltas = cos_deltas / norm_factor
    sin_deltas = sin_deltas / norm_factor

    # 3) Determine bin with the highest confidence
    selected_bin = np.argmax(conf_logits)

    # 4) Calculate residual angle in the selected bin by using arctan2 function
    delta_angle = np.arctan2(sin_deltas[selected_bin], cos_deltas[selected_bin])

    # 5) Total orientation
    theta_l = bin_centers[selected_bin] + delta_angle
    
    # 6) Object dimensions
    dimensions = dims_avg[obj_type] + dims_residual
    
    return theta_l, dimensions

def get_P2(id):
    """Read camera calibration file and extract P2 matrix based on image id."""

    base_dir = './calib'
    file_path = os.path.join(base_dir, f"{id}.txt")
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("P2:"):
                parts = line.strip().split()
                data = list(map(float, parts[1:]))
                if len(data) != 12:
                    raise ValueError("P2 line does not contain 12 elements")
                P2 = np.array(data).reshape(3, 4)
                return P2
    raise ValueError("P2 data not found in the file.")


# Function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0