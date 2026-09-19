import cv2
import json
import numpy as np

with open("camera_calibration.json", "r") as f:
    data = json.load(f)

mtx = np.array(data["camera_matrix"])
dist = np.array(data["distortion_coefficients"])

capture = cv2.VideoCapture(2)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

success, frame = capture.read()

if success:
    h, w = frame.shape[:2]
    
    # 1. Calculate an optimized camera matrix
    # alpha=0: Automatically crops out ALL warped edges (gives clean rectangle)
    # alpha=1: Keeps every pixel but leaves slight dark curved borders
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    
    # 2. Correct the image instantly
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
    cv2.imwrite("test1.png", frame)
    # FEED THIS DIRECTLY TO YOLO
    cv2.imwrite("test2.png", undistorted_frame)
    print("Saved clean pinhole-corrected frame sample to disk.")

capture.release()
