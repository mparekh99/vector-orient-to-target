import anki_vector 
import cv2 
import numpy as np
from PIL import Image 
from ultralytics import YOLO
import time
from anki_vector.util import degrees

# ### CONSTANTS

CAMERA_WIDTH = 640 # pixels
CAMERA_HEIGHT = 384  # pixels
CAMERA_CENTER_X = CAMERA_WIDTH //  2
CAMERA_CENTER_Y = CAMERA_HEIGHT // 2
FOV_HORIZONTAL = 90
FOV_VERTIVAL = 50





#### MODEL

model = YOLO("current.pt")


# def oreint_est(x, y):
#     # a^2 + b^2 = c^2  

#     a = x - CAMERA_CENTER_X



#     # a = CAMERA_CENTER_X 
#     # b = x
#     # c = a**2 + b**2
    



def main():
    
    with anki_vector.Robot("006068a2") as robot:

        robot.behavior.set_head_angle(degrees(7.0))
        robot.behavior.set_lift_height(0.0)


        # Initialize Camera Feed 
        robot.camera.init_camera_feed()
        print("Camera Initialized")

        while True:
            frame_pil = robot.camera.latest_image.raw_image
            frame_np = np.array(frame_pil)
            frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            results = model(frame, conf=0.5)
            boxes = results[0].boxes.xywh
            classes = results[0].boxes.cls

            # img_h, img_w = frame.shape
            # cv2.line(frame, (img_w // 2, 0), (img_w // 2, img_h), (255, 255, 255), 1)


            for i, box in enumerate(boxes):
                cx, cy, w, h = box
                cls = int(classes[i])
                label = model.names[cls]

                if label == 'vector':

                    delta_x = cx - CAMERA_CENTER_X

                    deg_per_pixel_h = FOV_HORIZONTAL / CAMERA_WIDTH
                    angle_to_turn_deg = delta_x * deg_per_pixel_h

                    print(f"Turning {angle_to_turn_deg:.2f} degrees to face detected object...")
                    


                    ### DO WORK HERE 

                    # orient_est(cx, cy)

                
                    # # Draw vertical (Y) center line
                    # cv2.line(frame, (CAMERA_CENTER_X, 0), (CAMERA_CENTER_X, CAMERA_HEIGHT), (0, 255, 255), 1)

                    # # Draw horizontal (X) center line
                    # cv2.line(frame, (0, CAMERA_CENTER_Y), (CAMERA_WIDTH, CAMERA_CENTER_Y), (0, 255, 255), 1)

                    


                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


                    robot.behavior.turn_in_place(degrees(-angle_to_turn_deg))



            cv2.imshow("Vector FOV", frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1) # ~ 10 Hz loop rate

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
