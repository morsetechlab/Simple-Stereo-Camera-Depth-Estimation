# Import necessary libraries
import cv2  # OpenCV for image processing
import os  # File system operations
import numpy as np  # Numerical operations
import glob  # File pattern matching
import matplotlib.pyplot as plt  # Plotting for visualization

# -------------------
# Settings for capturing images
# -------------------

# Define output folders for left and right camera images
output_left_folder = "captured_images/left"
output_right_folder = "captured_images/right"

# Create directories if they do not exist
os.makedirs(output_left_folder, exist_ok=True)
os.makedirs(output_right_folder, exist_ok=True)

# Get the list of sorted image file paths in each folder
images_left = sorted(glob.glob(os.path.join(output_left_folder, '*.png')))
images_right = sorted(glob.glob(os.path.join(output_right_folder, '*.png')))

# Ensure at least 30 images per camera are available for calibration
if len(images_left) < 30 or len(images_right) < 30:
    print("Not enough images found. Please capture 30 images for each camera.")
    exit()

# -------------------
# Stereo camera calibration
# -------------------

print("Starting stereo camera calibration...")

# Define the checkerboard grid dimensions
CHECKERBOARD = (9, 6)  # Number of inner corners in the grid
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare a 3D array representing the real-world positions of the checkerboard corners
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Lists to store 3D and 2D points for calibration
objpoints = []  # 3D real-world points
imgpoints_left = []  # 2D image points from the left camera
imgpoints_right = []  # 2D image points from the right camera

# Initialize real-time plot for reprojection error
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
errors = []
ax.set_title("Real-time Reprojection Error")
ax.set_xlabel("Image Index")
ax.set_ylabel("Reprojection Error")
line, = ax.plot([], [], 'r-', label="Mean Error")  # Real-time line plot
ax.legend()

# Iterate through pairs of left and right images
for i, (img_left, img_right) in enumerate(zip(images_left, images_right), start=1):
    # Load images in color
    color_left = cv2.imread(img_left)
    color_right = cv2.imread(img_right)

    # Convert images to grayscale for checkerboard detection
    gray_left = cv2.cvtColor(color_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(color_right, cv2.COLOR_BGR2GRAY)

    # Detect checkerboard corners in both images
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

    # Draw detected checkerboard corners for visualization
    if ret_left:
        cv2.drawChessboardCorners(color_left, CHECKERBOARD, corners_left, ret_left)
    else:
        print(f"Checkerboard not detected in left image of pair {i}")

    if ret_right:
        cv2.drawChessboardCorners(color_right, CHECKERBOARD, corners_right, ret_right)
    else:
        print(f"Checkerboard not detected in right image of pair {i}")

    # Combine left and right images side by side for display
    stereo_concat = cv2.hconcat([color_left, color_right])

    # Display progress on the combined image
    text = f"Calibrating... {i}/{len(images_left)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(stereo_concat, text, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the combined image with progress
    cv2.imshow("Stereo Calibration", stereo_concat)

    # Wait for 500 ms for visualization
    cv2.waitKey(500)

    # Skip the current pair if checkerboard is not detected in both images
    if not (ret_left and ret_right):
        print(f"Skipping pair {i} as checkerboard was not detected in one or both images.")
        continue

    # Store the object points and refined image points
    objpoints.append(objp)
    corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
    imgpoints_left.append(corners2_left)
    imgpoints_right.append(corners2_right)

    # Perform individual camera calibration
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_left.shape[::-1], None, None
    )
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_right.shape[::-1], None, None
    )

    # Perform stereo calibration
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1],
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
    )

    # Calculate reprojection error for both cameras
    total_error = 0
    for j in range(len(objpoints)):
        imgpoints2_left, _ = cv2.projectPoints(objpoints[j], rvecs_left[j], tvecs_left[j], mtx_left, dist_left)
        error_left = cv2.norm(imgpoints_left[j], imgpoints2_left, cv2.NORM_L2) / len(imgpoints2_left)

        imgpoints2_right, _ = cv2.projectPoints(objpoints[j], rvecs_right[j], tvecs_right[j], mtx_right, dist_right)
        error_right = cv2.norm(imgpoints_right[j], imgpoints2_right, cv2.NORM_L2) / len(imgpoints2_right)

        total_error += (error_left + error_right)

    mean_error = total_error / (2 * len(objpoints))
    errors.append(mean_error)

    # Update the real-time error plot
    line.set_xdata(range(1, i + 1))
    line.set_ydata(errors)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

# Finalize the plot
plt.ioff()
plt.show()

# Display final calibration error
print(f"Calibration completed with Mean Reprojection Error: {mean_error}")

# Save calibration parameters to a file
np.savez('stereo_calib_params.npz',
         cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
         cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
         R=R, T=T, E=E, F=F)

print("Stereo camera calibration completed! Results saved to 'stereo_calib_params.npz'")

# Close all OpenCV windows
cv2.destroyAllWindows()