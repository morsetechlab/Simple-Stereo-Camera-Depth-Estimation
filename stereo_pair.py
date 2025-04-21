import cv2
import numpy as np

# Load calibration parameters
calibration_file = "stereo_calib_params.npz"
calib_data = np.load(calibration_file)

cameraMatrix1 = calib_data["cameraMatrix1"]
distCoeffs1 = calib_data["distCoeffs1"]
cameraMatrix2 = calib_data["cameraMatrix2"]
distCoeffs2 = calib_data["distCoeffs2"]
R = calib_data["R"]
T = calib_data["T"]

# Initialize stereo camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the stereo camera.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Unable to read a frame from the stereo camera.")
    cap.release()
    exit()

height, width, _ = frame.shape
mid = width // 2
image_size = (mid, height)

# Stereo rectification
alpha = 0
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
    image_size, R, T, alpha=alpha
)

map1_left, map2_left = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_16SC2
)

# Stereo block matching
stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read a frame.")
        break

    # Split stereo image into left and right
    frame_left = frame[:, :mid]
    frame_right = frame[:, mid:]

    # Apply rectification maps
    rectified_left = cv2.remap(frame_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(frame_right, map1_right, map2_right, cv2.INTER_LINEAR)

    # Convert left and right images to red and blue channels
    red_channel = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    blue_channel = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    # Create a blank green channel (since we're not using green here)
    green_channel = np.zeros_like(red_channel)

    # Merge channels: left (red), right (blue), and green as 0
    stereo_colored = cv2.merge([blue_channel, green_channel, red_channel])

    # Display the results
    cv2.imshow("Stereo Pair with Red-Blue Coloring", stereo_colored)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
