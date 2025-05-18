import cv2
import numpy as np
import random
import torch
from ultralytics import YOLO


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Load YOLOv8 model for tracking
yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8n pretrained model
yolo_model.to(device)
yolo_model.fuse()

# Load calibration parameters
calibration_file = "stereo_calib_params.npz"
calib_data = np.load(calibration_file)

cameraMatrix1 = calib_data["cameraMatrix1"]
cameraMatrix2 = calib_data["cameraMatrix2"]
distCoeffs1 = calib_data["distCoeffs1"]
distCoeffs2 = calib_data["distCoeffs2"]
R = calib_data["R"]
T = calib_data["T"]

# Stereo rectification
baseline_m = 0.06  # Baseline in meters
focal_length_px = cameraMatrix1[0, 0]  # Focal length in pixels

# Scaling factor
actual_distance = 1.00  # True distance in meters
measured_distance = 0.74  # Measured distance in meters
scaling_factor = actual_distance / measured_distance
print(f"Scaling Factor: {scaling_factor:.3f}")

cap = cv2.VideoCapture(0)  # Replace with your stereo camera ID
if not cap.isOpened():
    print("Error: Unable to open the stereo camera.")
    exit()

# Stereo rectification setup
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read a frame from the stereo camera.")
    cap.release()
    exit()

height, width, _ = frame.shape
mid = width // 2
image_size = (mid, height)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R, T, alpha=0
)

map1_left, map2_left = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_16SC2
)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 8,
    blockSize=11,
    P1=8 * 3 * 11**2,
    P2=32 * 3 * 11**2
)

# Filtered classes
target_classes = {"person", "bottle", "pen", "cup", "cell phone", "scissors", "ruler"}


# Generate consistent dark colors for annotations based on object ID
def get_color(obj_id):
    if obj_id is None:  # Handle None case
        obj_id = 0
    random.seed(int(obj_id))  # Ensure obj_id is an integer
    base_color = [random.randint(0, 255) for _ in range(3)]  # Generate a consistent color
    dark_color = [int(c * 0.7) for c in base_color]  # Reduce brightness to make it dark
    return dark_color

def filter_disparity(disparity, bbox):
    x1, y1, x2, y2 = bbox
    object_disparity = disparity[y1:y2, x1:x2]
    object_disparity = cv2.medianBlur(object_disparity, 5)  # Apply median blur to smooth disparity
    return object_disparity

def draw_label_with_background(image, text, pos, color, text_color=(255, 255, 255), font_scale=1.4, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    x, y = pos
    cv2.rectangle(image, (x, y - text_height - 5), (x + text_width + 5, y + 5), color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)

# Get the FPS of the camera
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Use the actual FPS from the camera
if fps == 0:
    fps = 20  # Default to 20 FPS if the camera does not provide FPS

# Setup VideoWriter
output_file = "tracked_objects_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
frame_width = width
frame_height = height
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read a frame.")
        break

    frame_left = frame[:, :mid]
    frame_right = frame[:, mid:]

    rectified_left = cv2.remap(frame_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(frame_right, map1_right, map2_right, cv2.INTER_LINEAR)

    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Convert disparity map to color map (Jet)
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(disparity, alpha=255 / np.max(disparity)), cv2.COLORMAP_JET
    )

    # Run object tracking on the left image
    results = yolo_model.track(source=rectified_left, persist=True, conf=0.5)

    for result in results:
        if result.boxes is not None:
            for box, conf, cls, obj_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls, result.boxes.id):
                if obj_id is None:
                    obj_id = 0  # Default ID for None case

                x1, y1, x2, y2 = map(int, box[:4])
                class_name = yolo_model.names[int(cls)]
                confidence = float(conf)

                if class_name not in target_classes:
                    continue

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Filter disparity map for the object
                filtered_disparity = filter_disparity(disparity, (x1, y1, x2, y2))
                center_disparity = np.median(filtered_disparity)

                if center_disparity > 0:
                    Z_meters = (focal_length_px * baseline_m) / center_disparity
                    Z_meters_calibrated = Z_meters * scaling_factor
                    distance_text = f"{Z_meters_calibrated:.2f} m"
                else:
                    distance_text = "N/A"

                # Draw bounding box, ID, class name, confidence, and distance
                color = get_color(obj_id)
                cv2.rectangle(rectified_left, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), color, 2)
                text = f"ID {int(obj_id)}, {class_name}, distance: {distance_text}"
                draw_label_with_background(rectified_left, text, (x1, y1 - 10), color)
                draw_label_with_background(depth_colormap, text, (x1, y1 - 10), color)

    # Concatenate and display the results
    combined_display = cv2.hconcat([rectified_left, depth_colormap])
    cv2.imshow("Tracked Objects with Distance and Depth Map", combined_display)

    # Write the frame to the output video
    out.write(combined_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and camera
out.release()
cap.release()
cv2.destroyAllWindows()
