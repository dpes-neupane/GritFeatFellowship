from codecs import backslashreplace_errors
import cv2 as cv
from matplotlib.pyplot import box
import numpy as np
import os

IMAGE = ".\\nirmalBadri.jpg"

#this function tries to remove the background and find all the moving objects in the video
def capture():
    video =  os.getcwd() + "\\video2.mp4"
    print(video)
    cap = cv.VideoCapture(video)
    backSub = cv.createBackgroundSubtractorMOG2()
    if cap.isOpened() == False:
        print("file not opened")
    while cap.isOpened():
        ret, frame = cap.read()
        mask = cv.GaussianBlur(frame, (5,5), 0)
        mask = backSub.apply(mask)
        
        _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv.contourArea(cnt) > 500:
                cv.drawContours(frame, cnt, -1, (255,255,255), 3)
                rect = cv.boundingRect(cnt)
                x,y,w,h = rect 
                cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
                
                
            
        
        if not ret: 
            break
        cv.imshow("mask", mask)
        cv.imshow("frame",frame)
        if cv.waitKey(10) & 0xff == ord("q"):
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    
def load_model():
    dnn = cv.dnn.readNet(".\\peopleCounter\\yolov3.weights", ".\\peopleCounter\\yolov3.cfg")    
    with open(".\\peopleCounter\\coco.names") as fp:
        classes = [line.strip() for line in fp.readlines()]
    layers_names = dnn.getLayerNames()
    output_layers = [layers_names[int(i)-1] for i in dnn.getUnconnectedOutLayers()]
    return dnn, output_layers, classes
    

def load_image(image):
    # image = cv.imread(image)
    image = cv.resize(image, None, fx=0.4, fy=0.4)
    height, width, _ = image.shape
    blob = cv.dnn.blobFromImage(image=image, scalefactor=0.00392,size=(320, 320), mean=(0,0,0), swapRB=True, crop=False)
    return image, blob, height, width


def predict(input, output_layers, dnn):
    dnn.setInput(input)
    outputs = dnn.forward(output_layers)
    return outputs

def get_detection_values(outputs, width, height):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            center_x = int(detect[0]* width)
            center_y = int(detect[1]* height)
            w = int(detect[2]*width)
            h = int(detect[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confs.append(float(conf))
            class_ids.append(class_id)
    return boxes, confs, class_ids
            
            
def bounding_boxes(boxes, confs, class_ids, image, classes):
    indices = cv.dnn.NMSBoxes(boxes, confs, 0.5, 0.2) #yo padna baki xa
    font = cv.FONT_HERSHEY_COMPLEX
    for i in range(len(boxes)):
        label = str(classes[class_ids[i]])
        if i in indices and label == "person":
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            cv.rectangle(image, (x,y), (x+w, y+h), 255, 1)
            cv.putText(image, label, (x, y-5), font, 1, 255, 1)
    cv.imshow("yolo", image)
    
    
    
    
    
def yoloTracker():
    print(os.getcwd())
    video =  ".\\peopleCounter\\video2.mp4"
    cap = cv.VideoCapture(video)
    if cap.isOpened() == False:
        print("file not opened")
    dnn, output_layers, classes = load_model()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        image, blob, height, width = load_image(frame)
        outputs = predict(blob, output_layers, dnn)
        boxes, confs, class_ids = get_detection_values(outputs, width, height)
        bounding_boxes(boxes, confs, class_ids, image, classes)
        if not ret: 
            break
        
        if cv.waitKey(10) & 0xff == ord("q"):
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    
    
yoloTracker()