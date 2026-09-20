# takes care of both the robot's spacial awareness relative to itself, and where the object is located
# also takes care of finding obstacles and stuff, this is where the yolo thingies should go 
# need to have fixed constants class for the sizings of objects, to accurately tell distance? (ex. hands, doors)

import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except (ImportError, AttributeError):  # missing, or a new mediapipe without the legacy solutions API
    mp = None
    print("mediapipe pose unavailable, person distance will use bounding box width")

# resolve data files next to this file so it works regardless of the caller's cwd
HERE = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(HERE, "yolo26n.pt"))

with open(os.path.join(HERE, "camera_calibration.json"), "r") as f:
    data = json.load(f)

with open(os.path.join(HERE, "objectWidths.json"), "r") as f:
    widths = json.load(f)

CAMERA_DEVICE = os.environ.get("BOB_CAMERA", "/dev/video0")

mtx = np.array(data["camera_matrix"], dtype=np.float64)
dist = np.array(data["distortion_coefficients"])

pose = None
if mp is not None:
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# Create a quick helper to get the index from a string
def getItemIndex( instruction ):

    listOfWords = instruction.split(" ")
    print(listOfWords)
    # Invert the dictionary to map string -> integer index

    for word in listOfWords:
        name_to_index = {v: k for k, v in model.names.items()}
        index = name_to_index.get(word.lower(), None)
        if index != None:
            return index

    return None

class Click:
    def __init__(self):
        # On the Pi, OpenCV's numeric indexes don't match /dev/videoN, so open the Innomaker by path.
        # Override with BOB_CAMERA=<index or path> if it ever moves.
        source = int(CAMERA_DEVICE) if CAMERA_DEVICE.isdigit() else CAMERA_DEVICE
        self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2) if isinstance(source, str) else cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        success, frame = False, None
        for _ in range(30):  # let auto-exposure settle, the first frames are black
            success, frame = self.capture.read()
        self.lastKnownTarget = None

        if not success:
            raise RuntimeError( "No valid frame reading!" )

        h, w = frame.shape[:2]

        # calibration was made at data["resolution"]; rescale if the camera gave a different size
        calW, calH = data.get("resolution", [w, h])
        if (w, h) != (calW, calH):
            mtx[0, :] *= w / calW
            mtx[1, :] *= h / calH
            print(f"camera is {w}x{h}, calibration rescaled from {calW}x{calH}")

        # alpha=0: Automatically crops out ALL warped edges (gives clean rectangle)
        # alpha=1: Keeps every pixel but leaves slight dark curved borders
        self.new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        self.focalLength = self.new_camera_mtx[0, 0] # the fx and fy are the same, pixel apparent width is calced using min x and max x
        print("ready to capture...")

    def getImg(self, id):

        # dont do anything if u cant find 
        if id is None:
            print("no objects to locate... calling the regular model")
            
            return None, None
        
        self.capture.read()
        success, frame = self.capture.read()

        returns = []
        objectsFound = []
        obstaclesFound = []
        
        if success:
            h, w = frame.shape[:2]
            
            # 2. Correct the image instantly
            undistorted_frame = cv2.undistort(frame, mtx, dist, None, self.new_camera_mtx)
            results = model(undistorted_frame)

            for result in results:
                for box in result.boxes:

                    isObject = True

                    cls_id = int(box.cls[0])

                    if cls_id != id:
                        isObject = False

                    cls_name = model.names[cls_id]
                    xm, ym, xM, yM = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    pixelWidth = xM - xm
                    widthPinholeRatio = pixelWidth / self.focalLength

                    # ObjectHeightOrWidth / Distance = focalLength / pixelProjectionDistance 
                    # for the pinhole thingies
                    estimatedDist = widths[ cls_name ] / widthPinholeRatio # QUICK FIX, IT SEEMS TO BE DOUBLE OFF

                    # =========================================================
                    # MEDIAPIPE POSE (ISOLATED CROP ONLY) (Thanks gemini idk how mediapipe works :))
                    # =========================================================
                    if pose is not None and cls_name == "person" and conf >= 0.75:
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
                                        estimatedDist = shoulder_width / (px_w / self.focalLength)
                                        
                                        # Draw a green line ONLY between the shoulders inside the ROI
                                        # cv2.line(undistorted_frame, (lx, ly), (rx, ry), (0, 255, 0) if isObject else (0, 0, 255), 2) # to remove!
                    # =========================================================

                    if conf >= 0.70 and estimatedDist <= 4:

                        centerX = ( xm + xM ) / 2

                        # object/obstacle flag, distance in Z (not numpy), center of obj on x axis relative to sensor
                        
                        if isObject:
                            objectsFound.append([True, float(estimatedDist), centerX])
                        else:
                            obstaclesFound.append([False, float(estimatedDist), centerX])
                        
                        # cv2.rectangle( 
                        #     undistorted_frame, 
                        #     (xm, ym), 
                        #     (xM, yM), 
                        #     (0, 255, 0) if isObject else (0, 0, 255), 
                        #     2 
                        # ) # to remove!

                        # cv2.putText( 
                        #     undistorted_frame, 
                        #     f"{cls_name} at {estimatedDist:.2f}m\nConfidence:{conf:.2f}", 
                        #     (xm, max(ym-10, 15)), 
                        #     cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 
                        #     0.8, 
                        #     (0, 255, 255), 
                        #     2 
                        # ) # to remove!
            
            # cv2.imwrite("distances.png", undistorted_frame) # to remove!

            # only care about the closest instance of the object you are locating, and treat other one as an obstacle
            if len(objectsFound) > 0:
                closest = None
                closestDist = 999
                for obj in objectsFound:
                    if obj[1] < closestDist:
                        if closest is not None:
                            closest[0] = False
                            obstaclesFound.append(closest)
                        closestDist = obj[1]
                        closest = obj
                    else:
                        obj[0] = False
                        obstaclesFound.append(obj)

                self.lastKnownTarget = closest
                return closest, obstaclesFound
            else:
                if len(obstaclesFound) > 0:
                    return self.lastKnownTarget, obstaclesFound
                else:
                    return None, None
        else:
            return None, None

    def detectAll(self, minConf=0.5):
        """
        Everything currently in view, for "what do you see" style questions.
        Returns None if the camera read failed, else a list of
        {"name": str, "conf": float, "dist": float | None (meters), "centerX": float (px)}.
        Does not touch lastKnownTarget or the navigation logic.
        """
        self.capture.read()  # flush the stale buffered frame, same as getImg
        success, frame = self.capture.read()
        if not success:
            return None

        undistorted_frame = cv2.undistort(frame, mtx, dist, None, self.new_camera_mtx)
        found = []
        for result in model(undistorted_frame, verbose=False):
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < minConf:
                    continue
                cls_name = model.names[int(box.cls[0])]
                xm, ym, xM, yM = map(int, box.xyxy[0].tolist())
                pixelWidth = xM - xm
                estimatedDist = None
                if pixelWidth > 0 and cls_name in widths:
                    estimatedDist = float(widths[cls_name] / (pixelWidth / self.focalLength))
                found.append({
                    "name": cls_name,
                    "conf": conf,
                    "dist": estimatedDist,
                    "centerX": (xm + xM) / 2,
                })
        return found

    def __del__(self):
        self.capture.release()
        print("stopping captures...")

def main():
    cam = Click()
    tagID = getItemIndex("go to the chair")

    for _ in range(10):
        target, obstacles = cam.getImg(tagID)
        print("target:", target)
        print("obstacles:", obstacles)

if __name__ == "__main__":
    main()