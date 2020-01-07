#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import json
import time
import cv2
import os


model_path = "train_landmark/models/plateCornerDetCA90.pb"
support_image_extensions=[".jpg",".png",".jpeg",".bmp"]

input_width = 80
input_height = 40
channals_num=3

show_width = 600
show_height = 300

sessLandamrk=tf.Session()
with tf.gfile.FastGFile(model_path, "rb") as fr:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fr.read())
    sessLandamrk.graph.as_default()
    tf.import_graph_def(graph_def, name="")

sessLandamrk.run(tf.global_variables_initializer())

_landmark_output = sessLandamrk.graph.get_tensor_by_name('landmark_output:0')
_inputs = sessLandamrk.graph.get_tensor_by_name('input_1:0')

def preprocess(image_org):
    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    if channals_num == 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_norm = image / 255
    image_resized = cv2.resize(image_norm, (input_width, input_height))
    image_reshape_np = np.resize(image_resized, (input_height,input_width, channals_num))
    return image_reshape_np

def get_plate_Affine(image):
    #加载pb模型
    image_ori = image
    org_height, org_width = image_ori.shape[:2]
    processed_image = preprocess(image_ori)
    image_data = np.array([processed_image])
    landmark_output = sessLandamrk.run(_landmark_output, feed_dict={_inputs: image_data})

    plateCorners = []
    for i in range(4):
        x = int(landmark_output[0][2 * i] * org_width)
        y = int(landmark_output[0][2 * i + 1] * org_height)
        plateCorners.append([x, y])

    plateCorners = np.float32(plateCorners)

    #原始仿射-可能会比较紧
    past2 = np.float32([[0, 0], [show_width, 0], [show_width, show_height], [0, show_height]])
    M1 = cv2.getPerspectiveTransform(plateCorners, past2)
    dst1 = cv2.warpPerspective(image_ori, M1, (show_width, show_height))

    #宽松仿射-左右放宽20像素,上下不放宽
    padding = 30
    past2 = np.float32([[padding,30/2], [show_width-padding, 30/2], [show_width-padding, show_height-30/2], [padding, show_height-30/2]])
    M2 = cv2.getPerspectiveTransform(plateCorners, past2)
    dst2 = cv2.warpPerspective(image_ori, M2, (show_width, show_height))

    # cv2.imshow("dst1", dst1)
    # cv2.imshow("dst2", dst2)

    return dst2,plateCorners

