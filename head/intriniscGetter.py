# gemini's tuner

import cv2
import numpy as np
import json

# Initialize camera port 2
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def nothing(x):
    pass

# Create an interactive adjustment window
cv2.namedWindow("Live Tuner")
cv2.createTrackbar("Focal Length", "Live Tuner", 900, 2000, nothing)
cv2.createTrackbar("k1 (Barrel)", "Live Tuner", 200, 1000, nothing) # Mapped to negative scale
cv2.createTrackbar("k2 (Corners)", "Live Tuner", 500, 1000, nothing)

print("🎛️ Live tuner started! Look at a straight line (doorframe, table edge).")
print("• Adjust sliders until the lines look completely flat and straight.")
print("• Press [SPACEBAR] to save your custom matrix configuration file.")
print("• Press [ESC] to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    h, w = frame.shape[:2]
    
    # Get values from sliders
    f_slider = cv2.getTrackbarPos("Focal Length", "Live Tuner")
    k1_slider = cv2.getTrackbarPos("k1 (Barrel)", "Live Tuner")
    k2_slider = cv2.getTrackbarPos("k2 (Corners)", "Live Tuner")
    
    # Convert sliders to proper mathematical float values
    fx = float(max(f_slider, 100))
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    
    # Map sliders to the correct positive/negative ranges
    k1 = -(float(k1_slider) / 1000.0)  # Barrel distortion is negative
    k2 = (float(k2_slider) - 500.0) / 1000.0
    
    # Build standard matrices
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    
    # Apply settings instantly to the live stream
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
    
    # Display the result
    cv2.imshow("Live Tuner", undistorted_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Press Space to Save
        calibration_data = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "is_fisheye": False,
            "resolution": [w, h]
        }
        with open("camera_calibration.json", "w") as f:
            json.dump(calibration_data, f, indent=4)
        print("\n💾 Saved custom profile successfully to 'camera_calibration.json'!")
        print(f"Final parameters -> Focal: {fx}, k1: {k1}, k2: {k2}")
        
    elif key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
