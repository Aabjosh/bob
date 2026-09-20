# takes care of both the robot's spacial awareness relative to itself, and where the object is located
# also takes care of finding obstacles and stuff, this is where the yolo thingies should go 
# need to have fixed constants class for the sizings of objects, to accurately tell distance? (ex. hands, doors)

import cv2
import json
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

with open("camera_calibration.json", "r") as f:
    data = json.load(f)

with open("objectWidths.json", "r") as f:
    widths = json.load(f)

mtx = np.array(data["camera_matrix"])
dist = np.array(data["distortion_coefficients"])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,  # Lite model (fastest performance for live video/Pi)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

capture = cv2.VideoCapture(2)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

success, frame = capture.read()

if not success:
    raise RuntimeError( "No valid frame reading!" )

h, w = frame.shape[:2]

# alpha=0: Automatically crops out ALL warped edges (gives clean rectangle)
# alpha=1: Keeps every pixel but leaves slight dark curved borders
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
focalLength = new_camera_mtx[0, 0] # the fx and fy are the same, pixel apparent width is calced using min x and max x

while True:
    success, frame = capture.read()

    if success:
        h, w = frame.shape[:2]
        
        # 2. Correct the image instantly
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
        results = model(undistorted_frame)

        for result in results:
            for box in result.boxes:
                xm, ym, xM, yM = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                pixelWidth = xM - xm
                widthPinholeRatio = pixelWidth / focalLength

                # ObjectHeightOrWidth / Distance = focalLength / pixelProjectionDistance 
                # for the pinhole thingies
                estimatedDist = widths[ cls_name ] / widthPinholeRatio # QUICK FIX, IT SEEMS TO BE DOUBLE OFF

                # =========================================================
                # MEDIAPIPE POSE (ISOLATED CROP ONLY) (Thanks gemini idk how mediapipe works :))
                # =========================================================
                if cls_name == "person" and conf >= 0.75:
                    crop_ym, crop_yM = max(0, ym), min(h, yM)
                    crop_xm, crop_xM = max(0, xm), min(w, xM)
                    person_roi = undistorted_frame[crop_ym:crop_yM, crop_xm:crop_xM]

                    if person_roi.size > 0:
                        roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(roi_rgb)

                        if pose_results.pose_landmarks:
                            lms = pose_results.pose_landmarks.landmark
                            roi_h, roi_w = person_roi.shape[:2]

                            l_sh = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
                            r_sh = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                            if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
                                # Map relative landmark coordinates back to global frame pixels
                                lx = crop_xm + int(l_sh.x * roi_w)
                                ly = crop_ym + int(l_sh.y * roi_h)
                                rx = crop_xm + int(r_sh.x * roi_w)
                                ry = crop_ym + int(r_sh.y * roi_h)

                                px_w = np.hypot(lx - rx, ly - ry)
                                if px_w > 0:
                                    shoulder_width = 0.40  # Standard shoulder width in meters
                                    estimatedDist = shoulder_width / (px_w / focalLength)
                                    
                                    # Draw a green line ONLY between the shoulders inside the ROI
                                    cv2.line(undistorted_frame, (lx, ly), (rx, ry), (0, 255, 0), 2)
                # =========================================================

                if conf >= 0.75:
                    cv2.rectangle( 
                        undistorted_frame, 
                        (xm, ym), 
                        (xM, yM), 
                        (255, 200, 0), 
                        2 
                    )

                    cv2.putText( 
                        undistorted_frame, 
                        f"{cls_name} at {estimatedDist:.2f}m\nConfidence:{conf:.2f}", 
                        (xm, max(ym-10, 15)), 
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 
                        0.8, 
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
