import cv2
import os

from keras.models import Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, Input, AveragePooling2D
import keras

from nets import LprLocNet

import numpy as np

images_dir = "D:/forTensorflow/plateLandmarkDetTrain2/CA/train/images"

save_model_name="models/plateCornerDetCA"

input_width=80
input_height=40
channals_num=3
batch_size=64

image_ext="jpg"
label_ext="json"

def loadData(image_dir,landmark_object):
    imgs_path = os.listdir(image_dir)

    images_data=[]
    labels=[]

    for image_name in imgs_path:
        img_path = os.path.join(image_dir, image_name)

        image = cv2.imread(img_path)
        print(img_path)
        (org_h, org_w) = image.shape[:2]

        input_image=landmark_object.preprocess(image)
        images_data.append(input_image)

        if len(images_data)%5==0:
            print("Successfully load {} pictures.".format(len(images_data)))

        # 加载标签
        batch_label_path=img_path.replace("images","labels").replace(image_ext,label_ext)

        cur_label=[]
        with open(batch_label_path,"r") as fr:
            postions=eval(fr.read())

            # postions = list(map(int, fr.readline().strip().split(" ")))
            # print("postions:", postions)
            # for posId in range(0,8,2):
            #     cv2.circle(image,(postions[posId],postions[posId+1]),2,(0,0,255),2)
            # cv2.imshow("image",image)
            # cv2.waitKey(0)

            normalize_postions = []
            for ind, pos in enumerate(postions):
                if ind % 2 == 0:
                    normalize_postions.append(pos / org_w)
                    # normalize_postions.append(pos / w * image_width)
                else:
                    normalize_postions.append(pos / org_h)
                    # normalize_postions.append(pos / h * image_height)
            cur_label.extend(normalize_postions)
            labels.append(cur_label)

    return np.array(images_data),np.array(labels)

def train():
    landmark_object = LprLocNet.LprLocNet(input_width, input_height, channals_num, is_training=True)
    landmark_model = landmark_object.constructDetModel()
    # print(landmark_model.summary())

    print("Start training...")
    training_data,training_label= loadData(images_dir,landmark_object)
    print("Successflly load {} pictures and labels...".format(len(training_data)))

    lr = 0.005
    decay = 0.95
    for i in range(100):
        print("lr:", lr)
        adam = keras.optimizers.Adam(lr=lr)
        landmark_model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        train_history = landmark_model.fit(x=training_data, y=training_label, validation_split=0.2,
                                           epochs=10, batch_size=batch_size, verbose=1)
        lr *= decay
        if i %10==0:
            #同时保存网络架构和参数,在测试时即可直接通过keras.models.load_model直接加载网络架构和对应的参数
            landmark_model.save(save_model_name+str(i)+".h5")
            print("Succcessfully save model:", save_model_name + str(i) + ".h5")

train()