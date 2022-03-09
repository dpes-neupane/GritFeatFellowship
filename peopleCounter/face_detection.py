import cv2 as cv
from mtcnn import MTCNN
import os
print(os.getcwd())
img = cv.imread("./peopleCounter/nirmalBadri.jpg")
# img = cv.cvtColor(cv.imread('.\\nirmalBadri.jpg'), cv.COLOR_BGR2RGB)
print(img.shape)
# cv.imshow("  ", img)

detector = MTCNN()
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame  = cap.read()
    faces = detector.detect_faces(frame)
    # print(faces)
    for face in faces:
        img = cv.rectangle(img, (face['box'][0],  face['box'][1]), (face['box'][0] +face['box'][2],face['box'][1]+face['box'][3]), 255, 2 )
    # # img = cv.rectangle(img, )
    cv.imshow("faces", frame)
    if cv.waitKey(0) & 0xff == ord('q'):
        break
cap.release()
cv.destroyAllWindows()