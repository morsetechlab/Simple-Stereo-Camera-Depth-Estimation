import cv2
import numpy as np

# Load calibration parameters
# Load previously saved stereo camera calibration parameters from a file
calibration_file = "stereo_calib_params.npz"
calib_data = np.load(calibration_file)

# Load intrinsic and extrinsic parameters
# Extract intrinsic parameters (camera matrices and distortion coefficients) for both cameras
cameraMatrix1 = calib_data["cameraMatrix1"]  # Camera matrix for the left camera
distCoeffs1 = calib_data["distCoeffs1"]      # Distortion coefficients for the left camera
cameraMatrix2 = calib_data["cameraMatrix2"]  # Camera matrix for the right camera
distCoeffs2 = calib_data["distCoeffs2"]      # Distortion coefficients for the right camera

# Extract extrinsic parameters (rotation and translation between cameras)
R = calib_data["R"]  # Rotation matrix between the two cameras
T = calib_data["T"]  # Translation vector between the two cameras

# Initialize video capture for the stereo camera
# Open the stereo camera feed (e.g., ID 0 is typically the default camera)
cap = cv2.VideoCapture(0)  # Replace with your stereo camera ID

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open the stereo camera.")
    exit()

# Read a frame to determine the frame size
ret, frame = cap.read()  # Capture one frame from the stereo camera
if not ret:
    print("Error: Unable to read a frame from the stereo camera.")
    cap.release()  # Release the camera resource
    exit()

# Split the frame into left and right images
# Assuming the stereo image is a single frame with left and right images side by side
height, width, _ = frame.shape  # Get the height and width of the frame
mid = width // 2                # Calculate the middle of the frame
image_size = (mid, height)      # Define the size of each individual image

# Stereo rectification
# Rectify the images to align epipolar lines (correct perspective)
alpha = 0  # Alpha parameter controls the cropping level (0 = crop maximally, 1 = no crop)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
    image_size, R, T, alpha=alpha
)

# Compute rectification maps
# Generate mapping matrices for remapping the images to their rectified forms
map1_left, map2_left = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_16SC2  # Left camera
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_16SC2  # Right camera
)

# Real-time processing loop
# Process video frames from the stereo camera in real-time
while True:
    ret, frame = cap.read()  # Read a frame from the stereo camera
    if not ret:
        print("Error: Unable to read a frame.")  # Handle frame read failure
        break

    # Split the frame into left and right images
    frame_left = frame[:, :mid]  # Extract the left half of the frame
    frame_right = frame[:, mid:]  # Extract the right half of the frame

    # Apply rectification maps
    # Remap the left and right images to their rectified versions
    rectified_left = cv2.remap(frame_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(frame_right, map1_right, map2_right, cv2.INTER_LINEAR)

    # Concatenate rectified images for visualization
    # Combine the left and right rectified images into a single side-by-side image
    rectified_concat = cv2.hconcat([rectified_left, rectified_right])

    # Draw horizontal lines for alignment check
    # Draw green horizontal lines on the concatenated image at intervals to check rectification
    for i in range(0, rectified_concat.shape[0], 50):
        cv2.line(rectified_concat, (0, i), (rectified_concat.shape[1], i), (0, 255, 0), 1)

    # Show the rectified stereo feed
    # Display the concatenated rectified images in a window
    cv2.imshow("Rectified Stereo Feed", rectified_concat)

    # Exit on pressing 'q'
    # Stop the video feed if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
