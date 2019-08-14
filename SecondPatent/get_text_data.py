import os
import cv2
from random import randint
import numpy as np


root="D:/plateData/projectPlate/noMergedInfo/CA/"
save_root="D:/forCaffe/textAreaDet/"
STATE="CA"

width=600
height=300

char_labels_dir=os.path.join(root,"charLabel",STATE)
useful_image_dir=os.path.join(root,"usefulImgs",STATE)
labels_dir=os.path.join(root,"Labels",STATE)

save_plage_img_dir=os.path.join(save_root,"plate_images")
save_text_img_dir=os.path.join(save_root,"text_images")
save_label_dir=os.path.join(save_root,"labels")

if not os.path.exists(save_plage_img_dir):
    os.makedirs(save_plage_img_dir)
if not os.path.exists(save_text_img_dir):
    os.makedirs(save_text_img_dir)
if not os.path.exists(save_label_dir):
    os.makedirs(save_label_dir)


print("len:",len(os.listdir(labels_dir)))
save_id=0
for plate_label in os.listdir(labels_dir):
    if not plate_label.endswith("json"):
        print("continue1:",plate_label)
        continue
    image_file = plate_label.replace("json", "jpg")
    image_path=os.path.join(useful_image_dir, image_file)
    plate_label_path=os.path.join(labels_dir, plate_label)
    char_labels_path=os.path.join(char_labels_dir, plate_label)

    if not os.path.exists(image_path):
        print("continue2:", image_path)
        continue

    print(image_path)
    image = cv2.imread(image_path)
    with open(plate_label_path) as f:
        regions = eval(f.read())['region']

    plate_no_points = []
    plate_no_xmin = regions[0]['x']
    plate_no_xmax = regions[0]['x']
    plate_no_ymin = regions[0]['y']
    plate_no_ymax = regions[0]['y']

    for point in regions:
        x = point['x']
        y = point['y']
        plate_no_xmin = min(plate_no_xmin, x)
        plate_no_xmax = max(plate_no_xmax, x)
        plate_no_ymin = min(plate_no_ymin, y)
        plate_no_ymax = max(plate_no_ymax, y)
        plate_no_points.append([x, y])

    #pst1:原车牌区域四个角点,pst2:仿射后300x600的拉正车牌
    pst1 = np.float32(plate_no_points)
    pst2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pst1, pst2)
    dst = cv2.warpPerspective(image, M, (width, height))  #拉正后的车牌

    #车牌仿射矩阵的逆矩阵
    M_INV = np.linalg.inv(M)

    #车牌区域逆投影变换,仿射到原来整张大图片,这时只有原本的车牌区域是有像素值的,其他都是黑的
    plate_inv = cv2.warpPerspective(dst, M_INV, (image.shape[1], image.shape[0]))
    # cv2.imshow("plate_inv", plate_inv)
    # cv2.imwrite("plate_inv.jpg", plate_inv)

    plate_img_show=dst.copy()

    with open(char_labels_path) as f:
        chars = eval(f.read())['chars']

    chars_avg_ymin=0
    chars_avg_ymax=0
    text_xmin=600
    text_xmax=0
    tag=""
    for char in chars:
        xmin = int(char['x'])
        ymin = int(char['y'])
        char_width = int(char['width'])
        char_height = int(char['height'])
        xmax = xmin + char_width
        ymax = ymin + char_height
        chars_avg_ymin += ymin
        chars_avg_ymax += ymax
        text_xmin = min(text_xmin, xmin)
        text_xmax = max(text_xmax, xmax)
        tag +=char['name']

        #在拉正的车牌上画出字符框
        # cv2.rectangle(plate_img_show, (xmin, ymin), (xmax, ymax),
        #               (randint(0, 255), randint(0, 255), randint(0, 255)), 2)


    text_xmin = max(0, text_xmin-10)
    text_xmax = min(width, text_xmax+10)

    chars_avg_ymin = chars_avg_ymin // len(chars)
    chars_avg_ymax = chars_avg_ymax// len(chars)
    chars_avg_ymin = max(0,chars_avg_ymin-5)
    chars_avg_ymax = min(height, chars_avg_ymax + 5)

    ##在拉正的车牌上画出字符区域框
    # cv2.rectangle(plate_img_show, (text_xmin, chars_avg_ymin), (text_xmax, chars_avg_ymax),
    #               (0, 0 , 255), 2)

    # print("pos1:",(text_xmin, chars_avg_ymin), (text_xmax, chars_avg_ymax))

    #text_img:拉正后的字符区域
    # print(chars_avg_ymin,chars_avg_ymax,text_xmin,text_xmax)

    text_img = dst[chars_avg_ymin:chars_avg_ymax,text_xmin:text_xmax]
    #plate_no_img:未被拉正的原车牌区域
    plate_no_img = image[plate_no_ymin:plate_no_ymax, plate_no_xmin:plate_no_xmax]

    # 直接对字符区域进行逆仿射,和对车牌区域逆仿射一样,原图上除了字符区域,其他都是黑色
    # 但是这样仿射其实是有点不对的,因为会把字符左上角仿射到原车牌的左上角
    # 因为原仿射矩阵是根据车牌区域求得的,不是根据字符区域求得的
    text_inv = cv2.warpPerspective(text_img, M_INV, (image.shape[1], image.shape[0]))
    # cv2.imshow("text_inv",text_inv)
    # cv2.imwrite("text_inv.jpg", text_inv)

    #通过逆仿射矩阵和仿射后字符区域四个角点坐标可以计算这四个点坐标对应的原图上的位置
    p1_inv = np.matmul(M_INV,np.array([text_xmin,chars_avg_ymin,1.0]))
    p2_inv = np.matmul(M_INV,np.array([text_xmax, chars_avg_ymin, 1.0]))
    p3_inv = np.matmul(M_INV,np.array([text_xmin, chars_avg_ymax, 1.0]))
    p4_inv = np.matmul(M_INV,np.array([text_xmax, chars_avg_ymax, 1.0]))

    text_points=[p1_inv,p2_inv,p3_inv,p4_inv]
    #注意！！！逆投影变换回去后，要除以z,保证z为1
    text_no_xmin = int(p1_inv[0]/p1_inv[2])
    text_no_xmax = int(p1_inv[0]/p1_inv[2])
    text_no_ymin = int(p1_inv[1]/p1_inv[2])
    text_no_ymax = int(p1_inv[1]/p1_inv[2])

    for text_point in text_points:
        z = text_point[2]
        text_no_xmin =  int(min(text_no_xmin, int(text_point[0]/z)))
        text_no_xmax =  int(max(text_no_xmax, int(text_point[0]/z)))
        text_no_ymin =  int(min(text_no_ymin, int(text_point[1]/z)))
        text_no_ymax =  int(max(text_no_ymax, int(text_point[1]/z)))

    #原图上未被拉正的字符区域
    text_no_img=image[text_no_ymin:text_no_ymax,text_no_xmin:text_no_xmax]
    org_img_show=image.copy()
    cv2.rectangle(org_img_show,(text_no_xmin,text_no_ymin),(text_no_xmax,text_no_ymax),(0,255,0),2)

    save_plate_image_path = os.path.join(save_plage_img_dir,tag+"_"+str(save_id)+".jpg")
    save_text_image_path = os.path.join(save_text_img_dir, tag+"_"+str(save_id)+ ".jpg")

    save_label_path=os.path.join(save_label_dir,tag+"_"+str(save_id)+".txt")
    # cv2.imwrite(save_plate_image_path , plate_img_show)
    # cv2.imwrite(save_text_image_path, text_img)

    with open(save_label_path,"w") as fw:
        center_x = (text_xmin + text_xmax)/2
        center_y = (chars_avg_ymin + chars_avg_ymax) / 2
        text_width = text_xmax-text_xmin
        text_height = chars_avg_ymax - chars_avg_ymin
        fw.write("1 {} {} {} {}".format(text_xmin, chars_avg_ymin, text_xmax, chars_avg_ymax))

        # fw.write("1 {} {} {} {}".format(center_x,center_y,text_width,text_height))
        # print("pos2:",(text_xmin, chars_avg_ymin), (text_xmax, chars_avg_ymax))
    save_id+=1
    # cv2.imshow("plate_img_show", plate_img_show)
    # cv2.waitKey(0)

    # text_inv = cv2.warpPerspective(text_img, M_INV, (image.shape[1], image.shape[0]))
    # cv2.imshow("plate_inv", text_inv)

    # print(save_text_path)
    # cv2.imwrite(save_text_path, text_img)
    # cv2.imwrite(save_texts_org_path, text_no_img)

    #显示
    # cv2.imshow("image", org_img_show )
    # cv2.imshow("text_img", text_img)
    # cv2.imshow("text_no_img",text_no_img)
    #
    # cv2.imshow("plate",plate_img_show)
    # cv2.imshow("plate-no", plate_no_img)
    # cv2.waitKey(0)
