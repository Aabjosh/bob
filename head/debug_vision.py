# Diagnoses "detectAll() returns []": grabs a warmed-up frame, saves raw + undistorted
# copies, and prints what YOLO sees in each at a low confidence threshold.
import cv2
import ClickClass as C

cam = C.Click()
for _ in range(30):  # let auto-exposure settle
    cam.capture.read()
ok, frame = cam.capture.read()
print("read ok:", ok)
if not ok:
    raise SystemExit("no frame from camera")

print("frame shape:", frame.shape, "| mean brightness (0-255):", round(float(frame.mean()), 1))
cv2.imwrite("raw.jpg", frame)
undistorted = cv2.undistort(frame, C.mtx, C.dist, None, cam.new_camera_mtx)
cv2.imwrite("undistorted.jpg", undistorted)

for label, img in (("raw", frame), ("undistorted", undistorted)):
    for result in C.model(img, conf=0.1, verbose=False):
        print(label, [(C.model.names[int(b.cls[0])], round(float(b.conf[0]), 2)) for b in result.boxes])
