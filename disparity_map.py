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
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# TrackBar update function
def update_parameters(x):
    numDisparities = cv2.getTrackbarPos("numDisparities", "Disparity Map") * 16
    blockSize = cv2.getTrackbarPos("blockSize", "Disparity Map") * 2 + 5
    preFilterCap = cv2.getTrackbarPos("preFilterCap", "Disparity Map")
    uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "Disparity Map")
    speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "Disparity Map")
    speckleRange = cv2.getTrackbarPos("speckleRange", "Disparity Map")

    stereo.setNumDisparities(max(16, numDisparities))  # numDisparities ต้องหารด้วย 16
    stereo.setBlockSize(max(5, blockSize))  # blockSize ต้องเป็นเลขคี่
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setSpeckleRange(speckleRange)

# Create TrackBars
cv2.namedWindow("Disparity Map")
cv2.createTrackbar("numDisparities", "Disparity Map", 1, 10, update_parameters)
cv2.createTrackbar("blockSize", "Disparity Map", 5, 20, update_parameters)
cv2.createTrackbar("preFilterCap", "Disparity Map", 15, 31, update_parameters)
cv2.createTrackbar("uniquenessRatio", "Disparity Map", 10, 50, update_parameters)
cv2.createTrackbar("speckleWindowSize", "Disparity Map", 100, 200, update_parameters)
cv2.createTrackbar("speckleRange", "Disparity Map", 32, 100, update_parameters)

# Preprocessing function to improve matching
def preprocess_image(image):
    return cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  # ใช้การเพิ่ม histogram ให้สมดุล

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

    # Preprocess images
    gray_left = preprocess_image(rectified_left)
    gray_right = preprocess_image(rectified_right)

    # Compute disparity map
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32)

    # Normalize disparity map for visualization
    disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_visual = np.uint8(disparity_visual)

    # Apply Color Map
    disparity_visual = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)

    # Display the results
    cv2.imshow("Disparity Map", disparity_visual)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
