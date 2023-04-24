from utilities import *


cap = cv2.VideoCapture("../Videos/sample4.mp4")  # For Video

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
player1_score = 0
player2_score = 0

p1_board=0
p2_board=0

box1_contour = dict()
box3_contour = dict()
box2_contour = dict()
box4_contour = dict()

flag=1

bordersize=10

while True and flag:

    success, img = cap.read()
    img = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    if success == True:
        if len(masks_seg)==0:
            print('segmenting badminton area')
            boxes_seg, masks_seg, cls_seg, probs_seg = predict_on_image(model_seg, img, conf=0.5)
        # image_with_masks = np.copy(img)
            for box_i, mask_i, cls_i, probs_i in zip(boxes_seg, masks_seg, cls_seg, probs_seg):
                currentClass = classNames_seg[int(cls_i)]
                img,cnts = overlay(img, mask_i, currentClass, alpha=0.3)
                # print(len(cnts))
                area = [cv2.contourArea(c) for c in cnts]
                index = pd.Series(area).idxmax()
                # print(index)
                if currentClass=='box1':
                    box1_contour['cnts']=cnts
                    box1_contour['index']=index
                    print(currentClass,' initialized')
                    print(box1_contour['index'])
                elif currentClass == 'box2':
                    box2_contour['cnts']=cnts
                    box2_contour['index']=index
                    print(currentClass, ' initialized')
                elif currentClass == 'box3':
                    box3_contour['cnts']=cnts
                    box3_contour['index']=index
                    print(currentClass, ' initialized')
                elif currentClass == 'box4':
                    box4_contour['cnts']=cnts
                    box4_contour['index']=index
                    print(currentClass, ' initialized')

            print('segmentation done')
        #
        cv2.drawContours(img, box1_contour['cnts'], box1_contour['index'], (36, 0, 12), 2)
        cv2.drawContours(img, box2_contour['cnts'], box2_contour['index'], (36, 0, 12), 2)
        cv2.drawContours(img, box3_contour['cnts'], box3_contour['index'], (36, 0, 12), 2)
        cv2.drawContours(img, box4_contour['cnts'], box4_contour['index'], (36, 0, 12), 2)



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
                # print(currentClass)
                if conf>0.3:
                    myColor = (255, 0, 0)

                    cvzone.putTextRect(img, f'{classNames1[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                       colorT=(255,255,255),colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                    middle_point=(int((x1+x2)/2),int((y1+y2)/2))
                    box1_tracker = cv2.pointPolygonTest(box1_contour['cnts'][box1_contour['index']], middle_point, False)
                    box2_tracker = cv2.pointPolygonTest(box2_contour['cnts'][box2_contour['index']], middle_point, False)
                    box3_tracker = cv2.pointPolygonTest(box3_contour['cnts'][box3_contour['index']], middle_point, False)
                    box4_tracker = cv2.pointPolygonTest(box4_contour['cnts'][box4_contour['index']], middle_point, False)
                    # print('box2_tracker',box2_tracker)
                    if box1_tracker>0 or box2_tracker>0:
                        print (["positive" if box1_tracker>0 else "negative"  ],["positive" if box2_tracker>0 else "negative"  ])
                        if box1_tracker > 0 :
                            player2_score=player2_score+box1_tracker
                        elif box2_tracker>0:
                            player2_score = player2_score + box2_tracker

                        if player2_score>4:
                            p2_board=p2_board+1
                            print('player2_score', p2_board)
                            player2_score=0
                            # flag=0


                    elif box3_tracker>0 or box4_tracker>0:
                        print (["positive" if box3_tracker>0 else "negative"  ],["positive" if box4_tracker>0 else "negative"  ])
                        if box3_tracker > 0 :
                            player1_score=player1_score+box3_tracker
                        elif box4_tracker>0:
                            player1_score = player1_score + box4_tracker
                        if player1_score>2:
                            p1_board=p1_board+1
                            print('player1_score', p1_board)
                            player1_score=0
                            # flag=0

                    else :

                        player1_score=0
                        player2_score=0
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
                # print(currentClass)
                if conf > 0.5 and currentClass == 'person':
                    myColor = (255, 0, 255)

                    cvzone.putTextRect(img, f'{classNames2[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        cv2.putText(img, 'Player2 Score: '+str(p2_board), (91, 200), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
        cv2.putText(img, 'Player1 Score: '+str(p1_board), (1190, 150), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
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

