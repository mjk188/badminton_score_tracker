from utilities import *


cap = cv2.VideoCapture("../Videos/badminton_sample.mp4")  # For Video

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


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('../results/badminton_sample_results.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

masks_seg=[]

while True:
    success, img = cap.read()

    if success == True:
        if len(masks_seg)==0:
            print('segmenting badminton area')
            boxes_seg, masks_seg, cls_seg, probs_seg = predict_on_image(model_seg, img, conf=0.05)
        # image_with_masks = np.copy(img)
        for box_i, mask_i, cls_i, probs_i in zip(boxes_seg, masks_seg, cls_seg, probs_seg):
            currentClass = classNames_seg[int(cls_i)]
            img,cnts = overlay(img, mask_i, currentClass, alpha=0.3)

        results = model1(img, stream=True,device="mps")

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                # cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames1[cls]
                print(currentClass)
                if conf>0:
                    myColor = (255, 0, 0)

                    cvzone.putTextRect(img, f'{classNames1[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                       colorT=(255,255,255),colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        results = model2(img, stream=True, device="mps")
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                # cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames2[cls]
                print(currentClass)
                if conf > 0.5 and currentClass == 'person':
                    myColor = (255, 0, 255)

                    cvzone.putTextRect(img, f'{classNames2[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
        result.write(img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        # cv2.waitKey(1)
    else:
        break
cap.release()
result.release()

# Closes all the framesd
cv2.destroyAllWindows()

print("The video was successfully saved")

