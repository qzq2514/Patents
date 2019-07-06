import cv2
import glob
import numpy as np
import os
import tensorflow as tf
from net import modelRegularIInception
from tensorflow.python.framework import graph_util
from random import randint
#from skimage import util
#import skimage as sk


images_path = 'D:/forTensorflow/noCharTrain/slideWindosSeg/train/'
test_path ='D:/forTensorflow/noCharTrain/slideWindosSeg/test/'

modelCKPT = 'modelNochar/ckpt/noCharRecModel.ckpt'
modelPB = 'modelNochar/pb/'

pb_name="noCharRecmodelColor-"
snapshot = 500

batch_size=56
class_num = 2

img_width=32
img_height=32
channals=3

charDict = {'nochar':0, 'char': 1}

def get_train_data(images_path):
    images = []
    labels = []
    for train_class in os.listdir(images_path):
        for pic in os.listdir(images_path + train_class):

            images.append(images_path + train_class + '/' + pic)
            labels.append(charDict[train_class])

    train_data = np.array([images, labels])
    train_data = train_data.transpose()
    np.random.shuffle(train_data)
    train_images = list(train_data[:, 0])
    train_labels = list(train_data[:, 1])
    return np.array(train_images), np.array(train_labels)


def next_batch_set(flag, indices, images_path, labels, batch_size=batch_size):
    if not flag:
        indices = np.random.choice(len(images_path), batch_size)
    try:
        batch_image_path = images_path[indices]
        batch_labels = labels[indices]
    except Exception as e:
        return None, None
    

    batch_images = []
    curId=0
    for image_file in batch_image_path:
        #type=image_file.split(".")[-1]
        #if type!="jpg" and type!="jpeg" and type!="png" and type!="bmp":
           #continue
        if channals==3:
           image = cv2.imread(image_file)
        elif channals==1:
           image = cv2.imread(image_file,0)

        if image is None:
           # print(image_file)
           batch_labels=np.delete(batch_labels,curId,axis=0)
           continue
        curId+=1
        image = cv2.resize(image, (img_width, img_height))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # gauss_img = add_gaussian_noise(image)
        image = np.resize(image, (img_width, img_height, channals))
        batch_images.append(image)

    batch_images = np.array(batch_images)
    return batch_images, batch_labels


def train():
    inputs = tf.placeholder(tf.float32, shape=[None, img_width, img_height, channals], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    keep_prob = tf.placeholder(tf.float32)

    cls_model = modelRegularIInception.modelRegularIInception(is_training=True, num_classes=class_num)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs,keep_prob)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    classes = postprocessed_dict['classes']
    logits  = postprocessed_dict['logits']
    classes_= tf.identity(classes, name='classes')
    logits_ = tf.identity(logits, name='logits')
    acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.01, global_step, 150, 0.9)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = optimizer.minimize(loss, global_step)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        maxAccu=0
        curBest=sess
        images, targets = get_train_data(images_path)
        iters = len(images) // batch_size - 1
        test_images, test_targets = get_train_data(test_path)
        for epoch_num in range(50):
            for iter_num in range(iters):
                indices = [k for k in range(iter_num * batch_size, (iter_num+1) * batch_size)]
                batch_images, batch_labels = next_batch_set(True, indices, images, targets)
                if batch_images is None:
                    continue
                train_dict = {inputs: batch_images, labels: batch_labels,keep_prob:0.5}

                test_batch_images, test_batch_labels = next_batch_set(False, indices, test_images, test_targets)
                if test_batch_images is None:
                    continue
                test_dict = {inputs: test_batch_images, labels: test_batch_labels,keep_prob:1.0}

                sess.run(train_step, feed_dict=train_dict)


                # loss_, acc_ = sess.run([loss, acc], feed_dict=train_dict)

                loss_, acc_, lr_ = sess.run([loss, acc, optimizer._learning_rate], feed_dict=train_dict)

                val_acc = sess.run([acc], feed_dict=test_dict)

                # if acc_>maxAccu:
                #     maxAccu=acc_
                #     curBest=sess

                curBest=sess
                if (iter_num+1)%snapshot==0:
                    #预测概率节点 保存成PB
                    constant_graph = graph_util.convert_variables_to_constants(curBest, curBest.graph_def, ['logits'])
                    with tf.gfile.FastGFile(modelPB+pb_name+str(iter_num+1)+'.pb', mode='wb') as f:
                        f.write(constant_graph.SerializeToString())
                if (iter_num+1)%snapshot==0:
                    #保存ckpt
                    saver.save(curBest, modelCKPT,global_step=iter_num+1)

                train_text = 'qzq46-epoch:{} iters:{} , loss: {}, acc: {}, lr: {}, val: {}'.format( epoch_num, iter_num, loss_, acc_, lr_, val_acc[0])
                print(train_text)
            

train()

