import cv2
import numpy as np
from Rect import *

net=cv2.dnn.readNetFromCaffe("models/textLoc/CGIM_loc_deploy_300.prototxt",
                             "models/textLoc/CGIM_loc_deploy_300.caffemodel")

def getLocRects(image):
    image=image.copy()
    # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    (h, w, c) = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5, False, False)
    net.setInput(blob, "data")
    detection = net.forward("detection_out")
    plateRects = []
    for i in np.arange(0, detection.shape[2]):
        conf = detection[0, 0, i, 2]
        indx = int(detection[0, 0, i, 1])
        if conf > 0:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(endX, w)
            endY = min(endY, h)

            width=endX-startX
            height=endY-startY
            if indx==1:
                plateRect=Rect("text",startX, startY,width,height,conf)
                plateRects.append(plateRect)
    return plateRects

