import os
import cv2
import numpy as np
from textLocate import *
from textRecognition import *
from plateLandmark import *
import time


# file_dir="test_images/"
# file_dir = "D:/forTensorflow/plateLandmarkDetTrain2/CA/test/images"  #自制倾斜图片  -非真实图片,无测试意义,矫正不放宽89.6%,矫正放宽-89.89%
file_dir="D:/forTensorflow/plateLandmarkDetTrain1/CA/images_clear/" #原带倾斜的图(clear)    不矫正-70.07% 矫正放宽-96.54%
# file_dir = "D:/forCaffe/textAreaDet/plate_images_rename/"    #仿射后300*600的车牌(clear)   本就是矫正后-96.94%
# file_dir = "D:/forCaffe/textAreaDet/test/"

debug=True

def main():

    correct_num = 0
    total_num = 0

    for img_file_name in os.listdir(file_dir):

        ind1=img_file_name.find("[TAG-")
        ind2 = img_file_name.find("]",ind1)
        label = img_file_name[ind1+5:ind2]
        # label = img_file_name.split("_")[0]

        img_file_path = os.path.join(file_dir,img_file_name)
        print(img_file_path)

        org_image = cv2.imread(img_file_path)

        # 车牌仿射-倾斜矫正
        start_time=time.time()
        affine_image,plateCorners=get_plate_Affine(org_image)

        if debug:
            colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
            image_show_land_mark=org_image.copy()
            for ind,point in enumerate(plateCorners):
                cv2.circle(image_show_land_mark, (point[0], point[1]), 2, colors[ind], 2)
            cv2.imshow("image_show_land_mark", image_show_land_mark)

        #选择是否使用矫正后的图片进行后面的操作
        next_image=affine_image #org_image,affine_image

        # 车牌字符区域定位
        textRects=getLocRects(next_image)

        image_show_text_loc=next_image.copy()
        text_iamge=None
        is_correct=False

        for rect in textRects:
            startX,startY,width,height=rect.x,rect.y,rect.width,rect.height
            text_image=next_image[startY:startY+height,startX:startX+width]
            pred_text=CTCRec(text_image)

            if debug:
                cv2.rectangle(image_show_text_loc, (startX, startY), (startX + width, startY + height), (255, 0, 0), 2)
                cv2.putText(image_show_text_loc,pred_text,(startX,startY-5),1,2,(0,0,255),2)
            is_correct= pred_text==label

        end_time=time.time()
        used_time=end_time-start_time
        correct_num=correct_num+1 if is_correct else correct_num
        total_num+=1

        print("total_num:{},acc:{:.4f},time:{}s,{}----{}".format(total_num,correct_num/total_num,used_time,label,pred_text))

        if debug and len(label)!=7 and is_correct:
            cv2.imshow("image_show_text_loc",image_show_text_loc)
            print("pred_text:",pred_text)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()