#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import cv2
import os
import time   #noCharRecmodelColor_new7000
pb_path = 'model/nocharModel/noCharRecmodelColor-500.pb'  #noCharRecModel_old1000.pb
depth=3
img_with=32
img_height=32

sessNoChar=tf.Session()

with tf.gfile.FastGFile(pb_path, 'rb') as fr:
    noCharGraph = tf.GraphDef()
    noCharGraph.ParseFromString(fr.read())
    sessNoChar.graph.as_default()
    tf.import_graph_def(noCharGraph, name='')

inputs = sessNoChar.graph.get_tensor_by_name('inputs:0')
logits = sessNoChar.graph.get_tensor_by_name('logits:0')
keep_prob=sessNoChar.graph.get_tensor_by_name('keep_prob:0')

def getCharProb(image):

    image_org = cv2.resize(image, (img_with, img_height))
    image = np.resize(image_org, (img_height, img_with, depth))
    image = np.array(image, dtype=np.uint8)

    image_np = np.expand_dims(image, axis=0)
    start_time=time.time()
    logits_ = sessNoChar.run(logits, feed_dict={inputs: image_np,keep_prob:1.0})
    # print("size: %s time: %.8f s logit:%.4f " % (image.shape, time.time() - start_time,logits_[0][1]))
    # cv2.imshow("image_org",image_org)
    # cv2.waitKey(0)
    return logits_

