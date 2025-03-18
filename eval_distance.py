import numpy as np
import pandas as pd
import os
import math
import glob

from utils import parse_label, calculate_iou

label_dir = './test_labels'
prediction_dir = './kitti_predictions'
prediction_files = sorted(glob.glob(os.path.join(prediction_dir, "*.txt")))

# Lists to collect distance errors for each ground truth distance range
distance_error_bin_0_10 = []
distance_error_bin_10_20 = []
distance_error_bin_20_30 = []
distance_error_bin_30_40 = []
distance_error_bin_above_40 = []

iou_threshold = 0.7

for prediction_path in prediction_files:
    prediction_name = os.path.basename(prediction_path)
    id = os.path.splitext(prediction_name)[0]

    # Load labels and predictions
    labels = parse_label(os.path.join(label_dir, f"{id}.txt"))
    predictions = parse_label(prediction_path)

    for label in labels:
        if label['type'] != 'Car':
            continue

        best_iou = 0
        best_match = -1

        for pred_idx, pred in enumerate(predictions):
            iou = calculate_iou(label['box'], pred['box'])
            if iou > best_iou:
                best_iou = iou
                best_match = pred_idx

        if best_match != -1 and best_iou >= iou_threshold:
            # Compute ground truth and predicted distances (using Euclidean distance from the camera)
            d_label = math.sqrt(label['pos'][0]**2 + label['pos'][1]**2 + label['pos'][2]**2)
            d_pred  = math.sqrt(predictions[best_match]['pos'][0]**2 + predictions[best_match]['pos'][1]**2 + predictions[best_match]['pos'][2]**2)
            d_err = abs(d_label - d_pred)

            # Sort the distance error based on the ground truth distance in the correct list
            if d_label < 10:
                distance_error_bin_0_10.append(d_err)
            elif d_label < 20:
                distance_error_bin_10_20.append(d_err)
            elif d_label < 30:
                distance_error_bin_20_30.append(d_err)
            elif d_label < 40:
                distance_error_bin_30_40.append(d_err)
            else:
                distance_error_bin_above_40.append(d_err)

# Print the results
print("\nDistance Error by Ground Truth Distance Bins:")
print(f"0-10m:    Avg Error: {sum(distance_error_bin_0_10)/len(distance_error_bin_0_10):.3f}, Count: {len(distance_error_bin_0_10)}")
print(f"10-20m:   Avg Error: {sum(distance_error_bin_10_20)/len(distance_error_bin_10_20):.3f}, Count: {len(distance_error_bin_10_20)}")
print(f"20-30m:   Avg Error: {sum(distance_error_bin_20_30)/len(distance_error_bin_20_30):.3f}, Count: {len(distance_error_bin_20_30)}")
print(f"30-40m:   Avg Error: {sum(distance_error_bin_30_40)/len(distance_error_bin_30_40):.3f}, Count: {len(distance_error_bin_30_40)}")
print(f"> 40m:    Avg Error: {sum(distance_error_bin_above_40)/len(distance_error_bin_above_40):.3f}, Count: {len(distance_error_bin_above_40)}")