# Vector Orient To Target 

This script enables an Anki Vector robot to detect a another Vector using YOLO object detection and turn toward it smoothly using a PID controller.

It runs in real-time (~10 Hz), performing detection and motion control with a live video feed from Vector's onboard camera. 

# How It Works
## 1. Object Detection (YOLO):
Every 100 ms (~10 Hz), the robot captures a frame from its camera. YOLO is run on the image to detect objects — specifically a vector class.
## 2. Calculate Target Angle:

- Find the center x of the bounding box of the detected Vector.

- Subtract the camera's center x to get the pixel offset (delta_x).

- Convert delta_x to degrees using:

```
degrees_per_pixel = FOV_HORIZONTAL / CAMERA_WIDTH
angle_to_target = delta_x * degrees_per_pixel

```


## 3. PID Control for Turning:
A PID (Proportional-Integral-Derivative) controller smooths the turn toward the target:

- Proportional (P): Corrects based on how far the object is from center.

- Integral (I): Corrects based on how long the error persists over time.

- Derivative (D): Corrects based on how fast the error is changing.

This allows Vector to turn smoothly and accurately toward the detected object.

## Expected Behavior
![VectorAngularTurn - Made with Clipchamp](https://github.com/user-attachments/assets/381a2d7b-b201-4770-abd7-972aa9c9e8bd)
![VectorAngularTurnIRL - Made with Clipchamp](https://github.com/user-attachments/assets/cefda71d-95fc-43fc-9eb2-b27a30bb0aed)
