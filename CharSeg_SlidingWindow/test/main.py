import os
import cv2
from drawProp import *
img_dir="testImgs/common/"   #special,common

for img_file in os.listdir(img_dir):
    img_path=os.path.join(img_dir,img_file)

    ind1=img_path.find("-")
    ind2=img_path.rfind("]")
    img=cv2.imread(img_path)

    imgH = img.shape[0]
    imgW = img.shape[1]
    img=cv2.resize(img,(imgW,imgH))

    rects,probs=slide(img,5,0.9,0.6)
    finalImg = img.copy()
    for rect in rects:
        cv2.rectangle(finalImg, (rect.xmin, rect.ymin), (rect.xmax, rect.ymax),
                      (randint(0, 255), randint(0, 255), randint(0, 255)),2)

        charImg=img[rect.ymin:rect.ymax,rect.xmin:rect.xmax]
    cv2.imshow("finalImg", finalImg)
    # cv2.imwrite("common.jpg",finalImg)   #common
    plt.plot(range(len(probs)),probs,"k")
    plt.show()
    cv2.waitKey(0)