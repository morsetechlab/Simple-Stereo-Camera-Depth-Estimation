# 📘 [English version available here](README.en.md)

![Task: Object Tracking + Depth Estimation](https://img.shields.io/badge/Task-Object%20Tracking%20%2B%20Depth%20Estimation-blue?style=for-the-badge)
![Model: YOLOv8 | Ultralytics](https://img.shields.io/badge/Model-YOLOv8%20%7C%20Ultralytics-purple?style=for-the-badge)
![Framework: OpenCV](https://img.shields.io/badge/Framework-OpenCV-red?style=for-the-badge)
![Real-time Ready](https://img.shields.io/badge/Real--time-Yes-green?style=for-the-badge)

# Simple Stereo Depth Estimation with YOLOv8 Tracking

> การตรวจจับและติดตามวัตถุพร้อมข้อมูลความลึกแบบเรียลไทม์ ด้วยกล้องสเตอริโอและ YOLOv8 พัฒนาเพื่อใช้งานในระบบหุ่นยนต์ งานวิจัยด้านคอมพิวเตอร์วิทัศน์ และอุปกรณ์ฝังตัว (Embedded Systems)

โปรเจกต์นี้นำเสนอขั้นตอนการทำงานที่ผสานการประเมินความลึกจากภาพด้วยกล้องสเตอริโอ (stereo vision) เข้ากับการตรวจจับและติดตามวัตถุด้วย YOLOv8 โดยออกแบบมาเพื่อรองรับการทำงานแบบเรียลไทม์ผ่านระบบกล้องสเตอริโอ

---

## Project Overview

### ขั้นตอนการใช้งาน:

1. ถ่ายภาพ Calibration ของกระดานหมากรุก (chessboard) อย่างน้อย 30 ภาพ โดยใช้กล้อง stereo เพื่อคำนวณพารามิเตอร์ภายในกล้อง และระยะสัมพันธ์ระหว่างกล้องทั้งสอง (ใช้สคริปต์ [`stereo_capture.py`](stereo_capture.py)) 

2. ทำการ Rectification (ปรับแนวภาพ) เพื่อให้ภาพซ้าย-ขวาอยู่ในแนวเดียวกัน และลดความผิดเพี้ยน (undistortion)

3. ใช้ YOLOv8 เพื่อตรวจจับวัตถุในภาพฝั่งซ้าย (Left Image) ของกล้อง stereo

4. สร้างแผนที่ความลึก (Disparity Map) จากภาพ rectified ซ้ายและขวา

5. นำค่า disparity มาคำนวณระยะของวัตถุ (Z-axis in meters) โดยอิงจาก geometry ของกล้อง stereo

6. แสดงผลลัพธ์: ภาพฝั่งซ้ายพร้อมกล่องครอบวัตถุ (bounding box) ที่ประกอบด้วยข้อมูลคลาส, ระยะห่าง และ ID ของวัตถุ

---

## Core Components

### Stereo Calibration
- โหลดพารามิเตอร์จาก `stereo_calib_params.npz`
- พารามิเตอร์ที่ใช้:
  - `cameraMatrix1`, `cameraMatrix2`
  - `distCoeffs1`, `distCoeffs2`
  - `R`, `T`

### Stereo Rectification
- ใช้ `cv2.stereoRectify` และ `initUndistortRectifyMap`
- ได้ภาพ rectified ทั้งซ้ายและขวา

### Depth Estimation
- ใช้ `cv2.StereoSGBM_create` เพื่อสร้าง disparity map
- คำนวณระยะวัตถุจาก disparity โดยใช้สมการ:

```math
Z = (focal\_length \times baseline) / disparity
```

- ปรับค่า Z ด้วย scaling factor โดยอ้างอิงจากระยะจริงที่ผู้ใช้กำหนดเองเพื่อความแม่นยำตามสถานการณ์ใช้งาน

### YOLOv8 Detection
- ใช้ Ultralytics YOLOv8 (`yolov8n.pt`)
- ตรวจจับและติดตามวัตถุที่น่าสนใจ เช่น `person`, `bottle`, `cup`, `cell phone` เป็นต้น

### Visualization
- แสดงผล bounding box พร้อม ID, ชื่อคลาส และระยะห่าง
- แสดงแผนที่ความลึกในรูปแบบ colormap (Jet)
- บันทึกวิดีโอผลลัพธ์เป็น `tracked_objects_output.mp4`

---

## Dependencies
- NumPy
- OpenCV
- Torch
- Ultralytics (YOLOv8)
- Python 3.8 ขึ้นไป

---

## Distance Estimation and Depth Map

<p align="center">
  <img src="distance-estimation.gif" alt="ตัวอย่างผลลัพธ์" width="100%" />
</p>

---

## หมายเหตุการใช้งาน
- ควรใช้กล้อง stereo จริงที่สามารถแสดงภาพซ้ายและขวาในเฟรมเดียวกัน
- ค่า `baseline` และ `focal_length` ควรอ้างอิงจากค่าที่ได้จากการคาลิเบรต เพื่อให้การคำนวณระยะมีความแม่นยำ
- รองรับการประมวลผลและบันทึกวิดีโอแบบเรียลไทม์

---

- โค้ดนี้ดัดแปลงและพัฒนาต่อยอดจาก [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) และ [OpenCV Stereo Vision API](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)


## Attribution

- **Open source computer vision library** [OpenCV](https://github.com/opencv/opencv)
- **YOLOv8** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Developed by** [MorseTech Lab](https://www.morsetechlab.com)

## 🛡️ License

Project นี้เผยแพร่ภายใต้ [GNU Affero General Public License v3.0 (AGPLv3)](LICENSE) เพื่อให้สอดคล้องกับเงื่อนไขการใช้งานของไลบรารีที่เกี่ยวข้อง

<!--
tags: Stereo Vision, Depth Estimation, YOLOv8, OpenCV, Real-time Tracking, Robotics, ADAS, Python Computer Vision, 3D Perception, Ultranalytics, Z-axis measurement
-->

<!-- Open Graph / Twitter Meta -->
<!--
<meta property="og:title" content="Stereo Depth Estimation with YOLOv8 Tracking" />
<meta property="og:description" content="Real-time object detection and depth estimation using stereo vision and YOLOv8. Ideal for robotics, ADAS, and 3D computer vision applications." />
<meta property="og:image" content="https://raw.githubusercontent.com/morsetechlab/stereo-depth-yolov8-tracking/main/distance-estimation.gif" />
<meta property="og:url" content="https://github.com/morsetechlab/stereo-depth-yolov8-tracking" />
<meta property="og:type" content="website" />

<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="Stereo Depth Estimation with YOLOv8 Tracking" />
<meta name="twitter:description" content="Track and estimate object distance in real-time using stereo cameras and YOLOv8. Built for robotics and embedded systems." />
<meta name="twitter:image" content="https://raw.githubusercontent.com/morsetechlab/stereo-depth-yolov8-tracking/main/distance-estimation.gif" />
-->
