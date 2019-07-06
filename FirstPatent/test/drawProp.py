import cv2
import tensorflow as tf

import matplotlib.pyplot as plt
from random import randint

from noCharRec import *

img_dir="testImgs/drawProb/special"


class Box:
    def __init__(self,xmin,ymin,xmax,ymax,prop):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.prop = prop


def slide(img,step,confThreshold,overlapThreshold):

    imgH=img.shape[0]
    imgW=img.shape[1]

    boxW=imgH/2-4
    boxH=imgH
    cnt=int((imgW-boxW)/step)

    props=[]
    boxesClu=[]
    curClu=[]

    lastXmin=-100
    lastXmax=lastXmin+boxW

    for s in range(cnt):
        xmin=int(s*step)
        xmax=int(xmin+boxW)

        curImg=img[0:boxH,xmin:xmax]
        curProp=getCharProb(curImg)

        props.append(curProp[0][1])

        box=Box(xmin,0,xmax,boxH,curProp[0][1])

        if curProp[0][1]>confThreshold:
            if (lastXmax-box.xmin)/boxW>overlapThreshold:
                curClu.append(box)
            else:
                boxesClu.append(curClu)
                curClu = []
            lastXmax = box.xmax

    boxesClu.append(curClu)
    boxesClu=boxesClu[1:]
    finalBoxes=[]

    for boxclu in boxesClu:
        boxclu.sort(key=lambda box:-box.prop)
        if len(boxclu)>0:
            finalBoxes.append(boxclu[0])
    return finalBoxes,props

if __name__=="__main__":

    for file_name in os.listdir(img_dir):
        imgPath=os.path.join(img_dir,file_name)
        img= cv2.imread(imgPath)
        finalImg=img.copy()
        finalBoxes,props=slide(img,3,0.9,0.6)

        # print(len(finalBoxes))
        for box in finalBoxes:
            cv2.rectangle(finalImg, (box.xmin, box.ymin), (box.xmax, box.ymax),
                            (randint(0, 255), randint(0, 255), randint(0, 255)), 2)
            charImg=img[box.ymin:box.ymax,box.xmin:box.xmax]

        cv2.imshow("finalImg",finalImg)
        plt.plot(range(len(props)),props)
        plt.show()




