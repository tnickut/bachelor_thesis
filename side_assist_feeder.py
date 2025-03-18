import cv2
import requests
import time
import numpy as np

VIDEO_PATH = "./traffic_videos_gta5/city_traffic.mp4"
VIDEO_SIDE = "left"
LEFT_CAM_URL = "http://localhost:8080/upload"
RIGHT_CAM_URL = "http://localhost:8081/upload"
STATUS_URL = "http://localhost:8082/status"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        _, encoded_image = cv2.imencode('.png', frame)
        files = {'image': ('frame.png', encoded_image.tobytes(), 'image/png')}
        
        # Send the frame to thhe correct camera server
        try:
            response = None
            if VIDEO_SIDE == "left":
                response = requests.post(LEFT_CAM_URL, files=files)
            else:
                response = requests.post(RIGHT_CAM_URL, files=files)
            print("Frame sent to camera. Response:", response.text)
            
        except Exception as e:
            print("Error sending frame:", e)
        
        # Query the occupancy status from the status server (port 8082)
        try:
            status_response = requests.get(STATUS_URL)
            status = status_response.json()
            print("Occupancy status:", status)
        except Exception as e:
            print("Error fetching occupancy status:", e)

    cap.release()

if __name__ == "__main__":
    main()
