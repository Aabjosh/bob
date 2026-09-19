import numpy as np
import cv2
import glob
import json

# ==========================================
# 1. CONFIGURATION (Updated to 20mm)
# ==========================================
CHESSBOARD_SIZE = (7, 6)
SQUARE_SIZE = 20.0  # Defined in millimeters (mm)

# ==========================================
# 2. SETUP DATA ARRAYS
# ==========================================
# Prepare 3D real-world coordinates scaled to millimeters
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []  # 3D points in real-world space (now in mm)
img_points = []  # 2D points in image plane (in pixels)

# Load your calibration pictures
images = glob.glob('calibration_images/*.[jp][pe][g]*') + glob.glob('calibration_images/*.png')

if not images:
    print("❌ Error: No images found in the 'calibration_images' folder!")
    exit()

print(f"📸 Found {len(images)} images. Processing corners using {SQUARE_SIZE}mm squares...")

# ==========================================
# 3. DETECT CHESSBOARD CORNERS
# ==========================================
image_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_shape = gray.shape[::-1]

    # Find checkerboard corners
    success, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if success:
        obj_points.append(objp)
        
        # Sub-pixel refinement for ultra-precise matrices
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_refined)
        print(f"✅ Pattern detected in: {fname}")
    else:
        print(f"⚠️ Pattern NOT detected in: {fname} (Skipping)")

# ==========================================
# 4. COMPUTE INTRINSIC MATRIX
# ==========================================
if len(img_points) < 3:
    print("❌ Error: Not enough valid images detected. Need at least 3 successful frames.")
    exit()

print("\n🧮 Calculating camera intrinsic parameters...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_shape, None, None
)

# ==========================================
# 5. DISPLAY & SAVE RESULTS
# ==========================================
print("\n🏆 --- CALIBRATION SUCCESSFUL ---")
print(f"Reprojection Error: {ret:.4f} pixels (Lower is better, ideal is < 0.5)")

print("\n📋 CAMERA INTRINSIC MATRIX (mtx):")
print(mtx)
print("\n📋 DISTORTION COEFFICIENTS (dist):")
print(dist)

# Save calibration matrix to a reusable file
calibration_data = {
    "square_size_mm": SQUARE_SIZE,
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.tolist(),
    "resolution": image_shape
}

output_file = "camera_calibration.json"
with open(output_file, "w") as f:
    json.dump(calibration_data, f, indent=4)

print(f"\n💾 Saved intrinsic matrix data securely to '{output_file}'!")
