# 📘 [เวอร์ชั่นภาษาไทยที่นี่](README.md)

![Model: Stereo Depth + YOLOv8](https://img.shields.io/badge/Model-YOLOv8%7C%20Ultralytics-purple?style=for-the-badge)
![Task: Depth Estimation](https://img.shields.io/badge/Task-3D%20Object%20Tracking-blue?style=for-the-badge)
![Framework: OpenCV + Ultralytics](https://img.shields.io/badge/Framework-OpenCV-red?style=for-the-badge)
![Real-time Ready](https://img.shields.io/badge/Real--time-Yes-green?style=for-the-badge)

# Simple Stereo Depth Estimation with YOLOv8 Tracking

> Real-time depth-aware object detection and tracking using stereo vision and YOLOv8. Built for robotics, computer vision research, and embedded systems.

This project presents a workflow that combines stereo vision depth estimation with object detection and tracking using YOLOv8. It is designed for real-time operation through a stereo camera system.

---

## Project Overview

### Steps:

1. Capture at least 30 images of a chessboard calibration pattern using a stereo camera to compute the camera intrinsics and the relative positioning of the two cameras (using the script [`stereo_capture.py`](stereo_capture.py)).

2. Perform Rectification to align the left and right images and reduce distortion.

3. Use YOLOv8 to detect objects in the left image of the stereo camera.

4. Create a Disparity Map from the rectified left and right images.

5. Use the disparity values to compute the object's distance (Z-axis in meters) based on the stereo camera geometry.

6. Display the results: The left image with bounding boxes around detected objects, including class info, distance, and object IDs.

---

## Core Components

### Stereo Calibration
- Load parameters from `stereo_calib_params.npz`
- Parameters used:
  - `cameraMatrix1`, `cameraMatrix2`
  - `distCoeffs1`, `distCoeffs2`
  - `R`, `T`

### Stereo Rectification
- Use `cv2.stereoRectify` and `initUndistortRectifyMap`
- Obtain rectified left and right images

### Depth Estimation
- Use `cv2.StereoSGBM_create` to create the disparity map
- Compute the object distance from the disparity using the equation:

```math
Z = (focal\_length \times baseline) / disparity
```

- Adjust the Z value with a scaling factor based on real-world distance provided by the user for improved accuracy in the specific use case

### YOLOv8 Detection
- Use Ultralytics YOLOv8 (`yolov8n.pt`)
- Detect and track objects of interest such as `person`, `bottle`, `cup`, `cell phone`, etc.

### Visualization
- Display bounding boxes with ID, class name, and distance
- Show the depth map in a colormap (Jet)
- Save the output video as `tracked_objects_output.mp4`

---

## Dependencies
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- Python 3.8 or higher

---

## Distance Estimation and Depth Map
<img src="distance-estimation.gif" alt="Sample Results" style="width: 100%;" />

---

## Usage Notes
- A real stereo camera setup is required to capture both the left and right images in a single frame.
- The `baseline` and `focal_length` values should be referenced from calibration results to ensure accurate distance computation.
- Supports real-time processing and video saving.

---

- This code is adapted and extended from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [OpenCV Stereo Vision API](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html).


## Attribution

- **Open source computer vision library** [OpenCV](https://github.com/opencv/opencv)
- **YOLOv8** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Developed by** [MorseTech Lab](https://www.morsetechlab.com)

## 🛡️ License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](LICENSE), in order to comply with the licenses of core dependencies used in this project

## Citation
```bibtex
@misc{morsetechlab2025stereoyolo,
  title = {Stereo Depth Estimation with YOLOv8: Real-time 3D Object Tracking using OpenCV and Ultralytics},
  author = {MorseTech Lab},
  year = {2025},
  url = {https://github.com/morsetechlab/stereo-depth-yolov8-tracking}
}
```

<meta property="og:image" content="https://raw.githubusercontent.com/morsetechlab/stereo-depth-yolov8-tracking/main/distance-estimation.gif" />

<!--
tags: Stereo Vision, Depth Estimation, YOLOv8, OpenCV, Real-time Tracking, Robotics, ADAS, Python Computer Vision, 3D Perception, Ultranalytics, Z-axis measurement
-->