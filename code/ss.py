import cv2
from utilities import *
import pandas as pd
image = cv2.imread('../Videos/frame_0521.jpg')
model_seg = YOLO("../Yolo-Weights/court_segmentation.pt")
classNames_seg= ['box3', 'netarea', 'box4', 'box2', 'box1']
# print(image)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(gray, 120, 255, 1)
# cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

point1 = (25, 50)
point2 = (500, 700)
point3 = (500, 1000)





boxes_seg, masks_seg, cls_seg, probs_seg = predict_on_image(model_seg, image, conf=0.05)
for box_i, mask_i, cls_i, probs_i in zip(boxes_seg, masks_seg, cls_seg, probs_seg):

    currentClass = classNames_seg[int(cls_i)]
    print(currentClass)
    # print(mask_i.shape)
    img,cnts = overlay(image, mask_i, currentClass, alpha=0.3)
    # print(len(cnts))
    area=[cv2.contourArea(c) for c in cnts]
    index=pd.Series(area).idxmax()
    # print(area)
    print(index)
    # Perform check if point is inside contour/shape
    # for c in cnts:
    cv2.drawContours(img, cnts, index, (36, 0, 12), 2)
    result1 = cv2.pointPolygonTest(cnts[index], point1, False)
    result2 = cv2.pointPolygonTest(cnts[index], point2, False)
    result3 = cv2.pointPolygonTest(cnts[index], point3, False)
    # Draw points
    cv2.circle(img, point1, 8, (100, 100, 255), -1)
    cv2.putText(img, 'point1', (point1[0] -10, point1[1] -20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
    cv2.circle(img, point2, 8, (200, 100, 55), -1)
    cv2.putText(img, 'point2', (point2[0] -10, point2[1] -20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
    cv2.circle(img, point3, 8, (150, 50, 155), -1)
    cv2.putText(img, 'point3', (point3[0] -10, point3[1] -20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)

    print('point1:', result1)
    print('point2:', result2)
    print('point3:', result3)

cv2.imshow('image', img)
cv2.waitKey()