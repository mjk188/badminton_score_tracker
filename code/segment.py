from ultralytics import YOLO
import cv2
import cvzone
import math
from ultralytics.yolo.utils.ops import scale_image
import numpy as np
# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)



cap = cv2.VideoCapture("/Users/shubham.gupta1/Downloads/Object-Detection-101/Videos/badminton_full.mp4")  # For Video

model1 = YOLO("../Yolo-Weights/shuttle.pt")
model2 = YOLO("../Yolo-Weights/yolov8n.pt")
model_seg = YOLO("../Yolo-Weights/court_segmentation.pt")

classNames1 = ['shuttlecock']
classNames2 = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
classNames_seg= ['box3', 'netarea', 'box4', 'box2', 'box1']
myColor = (0, 0, 255)




#
#
# while True:
#     success, img = cap.read()
#     alpha = 0.3
#     if success == True:
#         results = model_seg(img, stream=True,device="mps")
#         for r in results:
#             masks = r.masks
#             print(r.boxes.cls.cpu().numpy())
#             for mask in masks:
#                 # Bounding Box
#
#                 masks = mask.masks.cpu().numpy()  # masks, (N, H, W)
#                 masks = np.moveaxis(masks, 0, -1)  # masks, (H, W, N)
#                 # rescale masks to original image
#                 masks = scale_image(masks, mask.orig_shape)
#                 masks = np.moveaxis(masks, -1, 0)  # masks, (N, H, W)
#                 color = (0, 255, 0)
#                 color = color[::-1]
#                 colored_mask = np.expand_dims(masks, 0).repeat(3, axis=0)
#                 colored_mask = np.moveaxis(colored_mask, 0, -1)
#                 masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=color)
#                 image_overlay = masked.filled()
#                 image_combined = cv2.addWeighted(img, 1 - alpha, image_overlay, alpha, 0)
#
#         cv2.imshow("Image", image_combined)
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             break
#         # cv2.waitKey(1)
#     else:
#         break
# cap.release()
#
#
# # Closes all the framesd
# cv2.destroyAllWindows()
#
