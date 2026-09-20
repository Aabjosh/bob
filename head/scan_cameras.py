# Opens each camera by its exact /dev/video path (V4L2), warms it up, and reports
# size + brightness. Saves cam_video<N>.jpg for each. Edit PATHS if yours differ.
import cv2

PATHS = ["/dev/video0", "/dev/video2"]  # Innomaker U20CAM, LifeCam HD-6000

for path in PATHS:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"{path}: could not open")
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = False, None
    for _ in range(60):  # warm up auto-exposure
        ok, frame = cap.read()
    if ok and frame is not None:
        name = path.split("/")[-1]
        print(f"{path}: {frame.shape[1]}x{frame.shape[0]}, brightness {frame.mean():.1f}")
        cv2.imwrite(f"cam_{name}.jpg", frame)
    else:
        print(f"{path}: opened but no frame")
    cap.release()
