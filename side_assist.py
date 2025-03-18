import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Control verbosity

import cv2
import numpy as np
import time
import math
import threading
from collections import deque
import tensorflow as tf
from flask import Flask, request, jsonify
from ultralytics import *
from multibin_pipeline import *
from libs.bbox3d_utils import *
from libs.Plotting import *

# Camera projection matrix (P2) where f_x = f_y = w/(2*tan(fov/2))
p2 = np.array([
    [1920/(2*math.tan(120/2)), 0.0, 960, 0.0], 
    [0.0, 1920/(2*math.tan(120/2)), 540, 0.0], 
    [0.0, 0.0, 1.0, 0.0]])

# mean pooling layer for 3D processing
meanpool_layer = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")

# Global variables for storing the latest images and occupancy flags
latest_left_image = None
latest_right_image = None

occupied_left = False
occupied_right = False

# Locks to protect shared resources
left_lock = threading.Lock()
right_lock = threading.Lock()
occupancy_lock = threading.Lock()

# Flags to indicate if a camera image is currently under processing
processing_left = False
processing_right = False


def process_image(image, cam_position):
    """
    Process the image using the multibin pipeline.
    If any object is more than 1m away and in the same travel direction,
    flag the lane as occupied.
    """
    occupied = False
    img2 = image.copy()
    img3 = image.copy()
    bboxes2d = process2D(img2)
    for bbox2d in bboxes2d:
        bbox_coords, scores, classes = bbox2d
        bbox_np = bbox_coords.cpu().detach().numpy()
        left, top, right, bottom = bbox_np

        result_3d = process3D(image, p2, bbox2d, poolingLayer=meanpool_layer, poolingMinWidth=500)
        if result_3d is None:
            continue
        rotation_y, theta_ray, alpha, dimensions = result_3d

        # Estimate depth using formula in 2.6 Representation of 3D Bounding Boxes
        depth = (p2[1, 1] * dimensions[0]) / (bottom - top)
        
        # Convert rotation from radians to degrees for simplicity
        rotation_deg = rotation_y * 180 / math.pi

        # Optional visualization of 3D bounding box
        plot3d(img3, p2, bbox_np, dimensions, alpha, theta_ray)
        
        # Check condition: object less than one lane width (3.75m) away and in same travel direction
        # Tracel direction is set to 0° - 100° for left camera and 80° - 180° for right camera
        print(cam_position, depth, rotation_deg)
        if depth < 3.75: # Lane width according to ADFC
            if (cam_position == "left" and 0 <= rotation_deg <= 100) or (cam_position == "right" and 80 <= rotation_deg <= 180):
                occupied = True
                break  # One qualifying object is enough to flag occupancy

    # Display the frame (Optional)
    cv2.imshow("3D", img3)
    cv2.waitKey(10)

    return occupied

def left_processing_thread():
    global latest_left_image, processing_left, occupied_left
    while True:
        if latest_left_image is not None and not processing_left:
            with left_lock:
                # Copy and clear the image so new uploads can override
                img = latest_left_image.copy()
                latest_left_image = None
                processing_left = True
            occupied = process_image(img, cam_position="left")
            with occupancy_lock:
                occupied_left = occupied
            processing_left = False
        else:
            time.sleep(0.01)  # Small delay to prevent busy waiting

def right_processing_thread():
    global latest_right_image, processing_right, occupied_right
    while True:
        if latest_right_image is not None and not processing_right:
            with right_lock:
                # Copy and clear the image so new uploads can override
                img = latest_right_image.copy()
                latest_right_image = None
                processing_right = True
            occupied = process_image(img, cam_position="right")
            with occupancy_lock:
                occupied_right = occupied
            processing_right = False
        else:
            time.sleep(0.01) # Small delay to prevent busy waiting

# -------------------------------
# Flask (Server) App for Left Camera (Port 8080)
# -------------------------------
app_left = Flask("left_camera")

@app_left.route('/upload', methods=['POST'])
def upload_left():
    global latest_left_image
    if 'image' not in request.files:
        return "No image file in request", 400
    file = request.files['image']
    # Read file bytes and decode as PNG image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "Could not decode image", 400
    with left_lock:
        latest_left_image = img
    return "Left image received", 200

# -------------------------------
# Flask (Server) App for Right Camera (Port 8081)
# -------------------------------
app_right = Flask("right_camera")

@app_right.route('/upload', methods=['POST'])
def upload_right():
    global latest_right_image
    if 'image' not in request.files:
        return "No image file in request", 400
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "Could not decode image", 400
    with right_lock:
        latest_right_image = img
    return "Right image received", 200

# -------------------------------
# Flask (Server) App for Occupancy Status (Port 8082)
# -------------------------------
app_status = Flask("status_server")

@app_status.route('/status', methods=['GET'])
def status():
    with occupancy_lock:
        status_data = {
            "occupied_left": occupied_left,
            "occupied_right": occupied_right
        }
    return jsonify(status_data)


if __name__ == '__main__':
    # Start processing threads for left and right cameras
    threading.Thread(target=left_processing_thread, daemon=True).start()
    threading.Thread(target=right_processing_thread, daemon=True).start()

    # Start Flask servers in separate threads.
    def run_app(app, port):
        app.run(port=port, host="0.0.0.0", threaded=True)

    threading.Thread(target=run_app, args=(app_left, 8080), daemon=True).start()
    threading.Thread(target=run_app, args=(app_right, 8081), daemon=True).start()
    threading.Thread(target=run_app, args=(app_status, 8082), daemon=True).start()

    print("Servers running on ports 8080 (left), 8081 (right), and 8082 (occupied status).")
    while True:
        time.sleep(1)
