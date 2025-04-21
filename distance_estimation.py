import cv2
import numpy as np

# Load calibration parameters
calibration_file = "stereo_calib_params.npz"
calib_data = np.load(calibration_file)

# Load intrinsic and extrinsic parameters
cameraMatrix1 = calib_data["cameraMatrix1"]
cameraMatrix2 = calib_data["cameraMatrix2"]
distCoeffs1 = calib_data["distCoeffs1"]
distCoeffs2 = calib_data["distCoeffs2"]
R = calib_data["R"]
T = calib_data["T"]

# Focal length in pixels
focal_length_px = cameraMatrix1[0, 0]  # Typically f_x

# Sensor size and resolution
sensor_width_mm = 6.4  # Adjust to your sensor's width in mm
sensor_resolution_px = None  # Placeholder for the sensor resolution

# Initialize video capture for the stereo camera
cap = cv2.VideoCapture(0)  # Replace with your stereo camera ID
if not cap.isOpened():
    print("Error: Unable to open the stereo camera.")
    exit()

# Read a frame to determine the frame size and resolution
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read a frame from the stereo camera.")
    cap.release()
    exit()

# Extract sensor resolution (width of the image in pixels)
height, width, _ = frame.shape
sensor_resolution_px = width // 2  # Divide by 2 for a stereo setup (left and right cameras)
print(f"Sensor Resolution (px): {sensor_resolution_px}")

# Calculate pixel size in mm/px
pixel_size_mm = sensor_width_mm / sensor_resolution_px
print(f"Pixel Size (mm/px): {pixel_size_mm:.6f}")

# Calculate focal length in mm
focal_length_mm = focal_length_px * pixel_size_mm
print(f"Focal Length (mm): {focal_length_mm:.2f} mm")

# Baseline (distance between cameras)
baseline_m = 0.06  # Baseline in meters (60 mm)

# Scaling factor (for distance calibration)
actual_distance = 1.00  # True distance in meters
measured_distance = 0.74  # Measured distance in meters (from the system)
scaling_factor = actual_distance / measured_distance
print(f"Scaling Factor: {scaling_factor:.3f}")

# Stereo rectification
mid = width // 2
image_size = (mid, height)
alpha = 0
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
    image_size, R, T, alpha=alpha
)

# Print Q matrix for verification
print("Q matrix:", Q)

# Compute rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_16SC2
)

# StereoBM or StereoSGBM setup
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,
    blockSize=11,
    P1=8 * 3 * 11**2,
    P2=32 * 3 * 11**2
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read a frame.")
        break

    # Split the frame into left and right images
    frame_left = frame[:, :mid]
    frame_right = frame[:, mid:]

    # Apply rectification maps
    rectified_left = cv2.remap(frame_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(frame_right, map1_right, map2_right, cv2.INTER_LINEAR)

    # Compute disparity map
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Normalize for visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

    # Get disparity at the center of the frame
    h, w = disparity.shape
    center_x, center_y = w // 2, h // 2
    center_disparity = disparity[center_y, center_x]

    if center_disparity > 0:  # Check for valid disparity
        Z_meters = (focal_length_px * baseline_m) / center_disparity  # Measured distance
        Z_meters_calibrated = Z_meters * scaling_factor  # Apply scaling factor
        calibrated_text = f"Distance: {Z_meters_calibrated:.2f} m"
    else:
        calibrated_text = "Distance: N/A"

    # Draw a rectangle with text on the top-right corner
    text_size, _ = cv2.getTextSize(calibrated_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_w, text_h = text_size
    rect_x1, rect_y1 = w - text_w - 20, 10
    rect_x2, rect_y2 = w - 10, text_h + 20
    cv2.rectangle(disp_vis, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)  # Black rectangle
    cv2.putText(disp_vis, calibrated_text, (rect_x1 + 5, rect_y2 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text

    # Draw a circle at the center point
    cv2.circle(disp_vis, (center_x, center_y), 5, (0, 255, 0), 2)

    # Show disparity map
    cv2.imshow("Disparity", disp_vis)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
