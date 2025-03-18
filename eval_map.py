import os
import numpy as np
import matplotlib.pyplot as plt
from utils import parse_label, calculate_iou

label_dir = './test_labels'
prediction_dir = './kitti_predictions'

# --- Step 1: Load and filter ground truth objects (only "Car") ---
gt_data = {}  # key: filename, value: list of ground truth objects (for "Car")
label_files = os.listdir(label_dir)
for file in label_files:
    label_path = os.path.join(label_dir, file)
    objects = parse_label(label_path)
    car_objects = [obj for obj in objects if obj['type'] == 'Car'] # Filter only for "Car"
    gt_data[file] = car_objects

# --- Step 2: Load and filter prediction objects (only "Car") ---
pred_data = []  # list of dicts with keys: file, score, box
prediction_files = os.listdir(prediction_dir)
for file in prediction_files:
    pred_path = os.path.join(prediction_dir, file)
    objects = parse_label(pred_path)
    car_objects = [obj for obj in objects if obj['type'] == 'Car'] # Filter for "Car"
    for obj in car_objects:
        pred_data.append({
            'file': file,
            'score': obj['score'],
            'box': obj['box']
        })

# --- Step 3: Compute AP for IoU thresholds from 0.5 to 0.95 ---
iou_thresholds = [round(th, 2) for th in np.arange(0.5, 0.96, 0.05)] # The list of IoU thresholds (step size 0.05)
ap_list = [] # List of AP values for each IoU threshold
pr_curves = {} # For visualization of each precision-recall curve

for iou_thr in iou_thresholds:
    # Sort all predictions by descending score
    sorted_preds = sorted(pred_data, key=lambda x: x['score'], reverse=True)
    
    # Create arrays to mark true positives (TP) and false positives (FP)
    TP = np.zeros(len(sorted_preds))
    FP = np.zeros(len(sorted_preds))
    
    # For each image, keep track of which ground truth boxes have been matched
    gt_detected = {}
    for file, objs in gt_data.items():
        gt_detected[file] = np.zeros(len(objs))
    
    # Total number of ground truth "Car" boxes across all images
    total_gts = sum(len(objs) for objs in gt_data.values())
    
    # Loop over each prediction and determine if it is a TP or FP at the current IoU threshold.
    for idx, pred in enumerate(sorted_preds):
        file = pred['file']
        pred_box = pred['box']
        gt_objs = gt_data.get(file, [])
        
        best_iou = 0.0
        best_gt_idx = -1
        # Compare the prediction with each ground truth box in the same file
        for j, gt in enumerate(gt_objs):
            iou = calculate_iou(pred_box, gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_thr:
            # Check if this ground truth box was already detected
            if gt_detected[file][best_gt_idx] == 0:
                TP[idx] = 1
                gt_detected[file][best_gt_idx] = 1
            else:
                FP[idx] = 1
        else:
            FP[idx] = 1

    # Compute cumulative true positives and false positives
    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)
    
    # Compute precision and recall at each prediction threshold
    recalls = cum_TP / (total_gts + 1e-6) # +1e-6 to avoid division by zero
    precisions = cum_TP / (cum_TP + cum_FP + 1e-6) # +1e-6 to avoid division by zero
    pr_curves[iou_thr] = (recalls, precisions)
    
    # Compute Average Precision (AP) using the method of integrating the precision-recall curve
    # First, add boundary points
    modified_recall = np.concatenate(([0.0], recalls, [1.0]))
    modified_precision = np.concatenate(([0.0], precisions, [0.0]))
    
    # Make the precision monotonically decreasing
    for i in range(len(modified_precision) - 1, 0, -1):
        modified_precision[i - 1] = max(modified_precision[i - 1], modified_precision[i])
    
    # Compute the area under the curve by summing over recall step changes
    ap = 0.0
    for i in range(1, len(modified_recall)):
        if modified_recall[i] != modified_recall[i - 1]:
            ap += (modified_recall[i] - modified_recall[i - 1]) * modified_precision[i]
    ap_list.append(ap)

# Finally, compute the mean AP over all IoU thresholds
mean_ap = np.mean(ap_list)
print("AP[0.5-0.95] for class 'Car':", mean_ap)

# --- Step 4: Draw the precision recall curves for each IoU threshold ---
plt.figure(figsize=(10, 6))
for iou_thr, (recalls, precisions) in pr_curves.items():
    plt.plot(recalls, precisions, marker='.', label=f'IoU = {iou_thr}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall-Kurven for the class 'Car'")
plt.legend()
plt.grid(True)
plt.show()