# KLT Algorithm – Kanade–Lucas–Tomasi Feature Tracking

An implementation of the KLT feature tracking method using image pyramids and optical flow.

### Overview

The Kanade–Lucas–Tomasi (KLT) algorithm is a classical computer vision method used to detect and track feature points across video frames.
It combines Tomasi–Shi corner detection for selecting good features and Lucas–Kanade optical flow for tracking them over time.

KLT is widely used in:

- Motion tracking
- Video stabilization
- Structure-from-motion
- Object tracking
- Human pose and gesture tracking

### Key Concepts
1. Good Features to Track (Tomasi–Kanade)

   Identifies strong corner points using the structure tensor and keeps points where both eigenvalues are large, meaning the pixel has intensity variation in all directions.

2. Lucas–Kanade Optical Flow

   Tracks the selected features in the next frame by assuming:
  - Small motion
  - Brightness constancy
  - Locally constant motion within a small neighborhood

3. Image Pyramid

   Used to handle large motions by tracking from coarse (low-resolution) to fine (high-resolution) levels.

### Workflow of KLT Algorithm

1. Convert input frame to grayscale.
2. Detect good feature points using Shi–Tomasi corner detector.
3. Build an image pyramid for multi-scale processing.
4. Apply Lucas–Kanade optical flow on each pyramid level.
5. Estimate motion and compute new feature positions.
6. Reject unreliable or lost feature points.
7. Visualize motion using lines/arrows between old and new positions.

### Features of This Implementation

- Selects high-quality corner features
- Tracks features frame-by-frame
- Uses pyramidal optical flow (good for large movements)
- Draws motion vectors (arrows)
- Works with webcam or video file

### Input Options

- Webcam mode: Tracks features in live video.
- Video file mode: Tracks features in a pre-recorded file.

---

### Requirements

- Python 3.x
- OpenCV
- NumPy

*Install dependencies:*
`pip install opencv-python numpy`

---
