import anki_vector 
import cv2 
import numpy as np
from PIL import Image 
from ultralytics import YOLO
import time
from anki_vector.util import degrees
import matplotlib.pyplot as plt
import math


## CONSTANTS

CAMERA_WIDTH = 640 # pixels
CAMERA_HEIGHT = 384  # pixels
CAMERA_CENTER_X = CAMERA_WIDTH //  2
CAMERA_CENTER_Y = CAMERA_HEIGHT // 2
FOV_HORIZONTAL = 90
FOV_VERTIVAL = 50
NUM_STEPS = 5000

# Try to do 8 degrees per step 

# PID controller constants for tuning 
K_P = 0.15
K_I = 0.002
K_D = 0.025
BIAS = 0





#### MODEL

model = YOLO("current.pt")


def move_deg_to_speed(move_deg, scale=7):
    # Scale move_deg (degrees) to wheel speed [-100, 100]
    return max(min(move_deg * scale, 100), -100)



def main():
    

    error_prev = 0
    integral = 0

    last_time = time.time()





    with anki_vector.Robot("00806b78") as robot:

        robot.behavior.set_head_angle(degrees(7.0))
        robot.behavior.set_lift_height(0.0)


        # Initialize Camera Feed 
        robot.camera.init_camera_feed()
        print("Camera Initialized")


        #LOOP
        while True:

            now = time.time()
            dt = (now - last_time)
            last_time = now


            frame_pil = robot.camera.latest_image.raw_image
            frame_np = np.array(frame_pil)
            frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            results = model(frame, conf=0.5)
            boxes = results[0].boxes.xywh
            classes = results[0].boxes.cls

            # img_h, img_w = frame.shape
            # cv2.line(frame, (img_w // 2, 0), (img_w // 2, img_h), (255, 255, 255), 1


            for i, box in enumerate(boxes):
                cx, cy, w, h = box
                cls = int(classes[i])
                label = model.names[cls]

                if label == 'vector':

                    delta_x = cx - CAMERA_CENTER_X

                    deg_per_pixel_h = FOV_HORIZONTAL / CAMERA_WIDTH
                    angle_to_target = delta_x * deg_per_pixel_h

    
                    print(f'Angle to target {angle_to_target}')

                    error = angle_to_target

                    integral = integral + (error * dt)
                    derivative = (error - error_prev) / dt
                    move_deg = (K_P * error) + (K_I * integral) + (K_D * derivative) + BIAS

                    move_deg = max(min(move_deg, 5), -5)

                    wheel_speed = move_deg_to_speed(move_deg)
                    robot.motors.set_wheel_motors(wheel_speed, -wheel_speed)

                    print(f"dt={dt:.3f} s | error={error:.2f} | integral={integral:.2f} | derivative={derivative:.2f} | move_deg={move_deg:.2f} | wheel_speed={wheel_speed:.2f}")

                    error_prev = error


                    


                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)




            cv2.imshow("Vector FOV", frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1) # ~ 10 Hz loop rate
            
        robot.motors.set_wheel_motors(0, 0) 
        cv2.destroyAllWindows()
        


if __name__ == '__main__':
    main()
