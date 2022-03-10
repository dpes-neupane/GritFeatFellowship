import cv2 as cv
# from mtcnn import MTCNN
import os
import numpy as np
# print(os.getcwd())
img = cv.imread("./peopleCounter/nirmalBadri.jpg")
# img = cv.cvtColor(cv.imread('.\\nirmalBadri.jpg'), cv.COLOR_BGR2RGB)
# print(img.shape)
# cv.imshow("  ", img)
def mtcnnFaceDetector():
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
    
    
dnn = cv.dnn.readNetFromCaffe("./peopleCounter/caffeModel/deploy.prototxt", "./peopleCounter/caffeModel/res10_300x300_ssd_iter_140000.caffemodel")

def caffeDnn(img):
    (h,w) = img.shape[:2]
    # print(dnn.getUnconnectedOutLayersNames())
    # print(dnn)
    layers_names = dnn.getLayerNames()
    # print(layers_names)
    blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300), (104, 117, 123), False, False)
    dnn.setInput(blob)
    # print(blob)
    detections = dnn.forward()
    # print(detections, detections.shape)
    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf >=0.8:
            box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            text = f"{conf:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.rectangle(img, (startX, startY), (endX, endY), 255, 5)
            cv.putText(img, text, (startX, y),cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    cv.imshow("", img)
    



def capture():
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        caffeDnn(frame)
        if cv.waitKey(10) & 0xff == ord('q'):
            break
    cv.destroyAllWindows()

capture()