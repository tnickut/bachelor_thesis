import numpy as np
import pandas as pd
import os
import math
import glob

from utils import parse_label, calculate_iou

label_dir = './test_labels'
prediction_dir = './kitti_predictions'

orientation_scores_easy = []
orientation_scores_moderate = []
orientation_scores_hard = []

iou_threshold = 0.7

prediction_files = sorted(glob.glob(os.path.join(prediction_dir, "*.txt")))
for prediction_path in prediction_files:
    prediction_name = os.path.basename(prediction_path)
    id = os.path.splitext(prediction_name)[0]

    # Load labels and predictions
    labels = parse_label(os.path.join(label_dir, f"{id}.txt"))
    predictions = parse_label(prediction_path)

    for label_idx, label in enumerate(labels):

        if (label['type'] != 'Car'):
            continue

        best_iou = 0
        best_match = -1
        
        for pred_idx, pred in enumerate(predictions):
            iou = calculate_iou(label['box'], pred['box'])
            if iou > best_iou:
                best_iou = iou
                best_match = pred_idx

        if best_match != -1 and best_iou >= iou_threshold:
            orientation_score = (1 + math.cos(label['rot_y'] - predictions[best_match]['rot_y'])) / 2

            bbox_height = label['box'][3] - label['box'][1]
            occlusion = label['occlusion']
            truncation = label['truncation']

            # Sort based on the difficulty level
            if (bbox_height >= 40 and occlusion <= 0 and truncation <= 15):
                orientation_scores_easy.append(orientation_score)
            elif (bbox_height >= 25 and occlusion <= 1 and truncation <= 30):
                orientation_scores_moderate.append(orientation_score)
            elif (bbox_height >= 25 and occlusion <= 2 and truncation <= 50):
                orientation_scores_hard.append(orientation_score)

# Print results
print("\nOrientation Scores Summary:")
print(f"Easy:     Avg: {sum(orientation_scores_easy)/len(orientation_scores_easy):.3f},  N: {len(orientation_scores_easy)}")
print(f"Moderate: Avg: {sum(orientation_scores_moderate)/len(orientation_scores_moderate):.3f},  N: {len(orientation_scores_moderate)}")
print(f"Hard:     Avg: {sum(orientation_scores_hard)/len(orientation_scores_hard):.3f},  N: {len(orientation_scores_hard)}")