import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# เปิดกล้อง
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the stereo camera.")
    exit()

# Stereo Calibration Parameters
calibration_file = "stereo_calib_params.npz"
calib_data = np.load(calibration_file)
cameraMatrix1 = calib_data["cameraMatrix1"]
distCoeffs1 = calib_data["distCoeffs1"]
cameraMatrix2 = calib_data["cameraMatrix2"]
distCoeffs2 = calib_data["distCoeffs2"]
R = calib_data["R"]
T = calib_data["T"]

# อ่านเฟรมแรกเพื่อดูขนาดภาพ
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read a frame from the stereo camera.")
    cap.release()
    exit()

height, width, _ = frame.shape
mid = width // 2
image_size = (mid, height)

# Stereo Rectification
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

# StereoBM or StereoSGBM setup
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*5,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2
)

# ตั้งค่ากราฟ 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([], [], [], s=0.5, c=[], cmap='jet', marker='o')

ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")
ax.set_title("Realtime 3D Point Cloud")

plt.ion()  # เปิดโหมด interactive สำหรับการแสดงผลแบบเรียลไทม์
plt.show()

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

    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    mask = disparity > 0  # Mask out invalid points
    points_3D = points_3D[mask]

    # Extract x, y, z coordinates
    X = points_3D[:, 0]
    Y = points_3D[:, 1]
    Z = points_3D[:, 2]

    # อัปเดตข้อมูลในกราฟ 3D
    scatter._offsets3d = (X, Y, Z)
    scatter.set_array(Z)  # ตั้งสีตามแกน Z
    ax.set_xlim([np.min(X), np.max(X)])
    ax.set_ylim([np.min(Y), np.max(Y)])
    ax.set_zlim([np.min(Z), np.max(Z)])

    plt.pause(0.001)  # หน่วงเวลาเล็กน้อยเพื่ออัปเดตกราฟ

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # ปิดโหมด interactive
plt.show()
