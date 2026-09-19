# takes care of both the robot's spacial awareness relative to itself, and where the object is located
# also takes care of finding obstacles and stuff, this is where the yolo thingies should go 
# need to have fixed constants class for the sizings of objects, to accurately tell distance? (ex. hands, doors)

import cv2
import json
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

with open("camera_calibration.json", "r") as f:
    data = json.load(f)

mtx = np.array(data["camera_matrix"])
dist = np.array(data["distortion_coefficients"])

capture = cv2.VideoCapture(2)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, frame = capture.read()

    if success:
        h, w = frame.shape[:2]
        
        # 1. Calculate an optimized camera matrix
        # alpha=0: Automatically crops out ALL warped edges (gives clean rectangle)
        # alpha=1: Keeps every pixel but leaves slight dark curved borders
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        
        # 2. Correct the image instantly
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
        results = model(undistorted_frame)

        for result in results:
            for box in result.boxes:
                xm, ym, xM, yM = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                cv2.rectangle( 
                    undistorted_frame, 
                    (xm, ym), 
                    (xM, yM), 
                    (255, 200, 0), 
                    2 
                )

                cv2.putText( 
                    undistorted_frame, 
                    f"{cls_name} {conf:.2f}", 
                    (xm, max(ym-10, 15)), 
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 
                    0.5, 
                    (255, 200, 0), 
                    2 
                )

        cv2.imshow("Josh Locator", undistorted_frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
