import tensorflow as tf
import numpy as np
import os
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_width=120
input_height=40

num_hidden = 64
batch_size = 1

char_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V","W","X","Y","Z","#"]

model_path = "models/CTCRec/CTC_CA_NoRNN_BN_dropout-10000.pb"
max_step_downsampling_num=4

sessCTCRec=tf.Session()

with tf.gfile.FastGFile(model_path, "rb") as fr:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fr.read())
    sessCTCRec.graph.as_default()
    tf.import_graph_def(graph_def, name="")

sessCTCRec.run(tf.global_variables_initializer())

inputs = sessCTCRec.graph.get_tensor_by_name('inputs:0')
seq_len_placeholder = sessCTCRec.graph.get_tensor_by_name('seq_len_gt:0')
dense_predictions = sessCTCRec.graph.get_tensor_by_name('dense_predictions:0')

def pharse_decode(decode_predictions_):
    for ind, val in enumerate(decode_predictions_):
        pred_number = ''
        for code in val:
            pred_number += char_set[code]
        pred_number = pred_number.strip("#")
    return pred_number

def CTCRec(textImage):
    org_color_image = textImage
    color_image_resized = cv2.resize(org_color_image, (input_width, input_height))
    color_image_tran_batch=color_image_resized[np.newaxis,:]

    seq_len = np.ones(batch_size) * input_width/max_step_downsampling_num

    eval_dict={inputs: color_image_tran_batch,seq_len_placeholder: seq_len}

    dense_predictions_ = sessCTCRec.run(dense_predictions, feed_dict=eval_dict)

    recString = pharse_decode(dense_predictions_)

    return recString


if __name__ == "__main__":
    eval_dir = "C:/Users/cgim/Desktop/test"
    for file_name in os.listdir(eval_dir):
        file_path = os.path.join(eval_dir,file_name)
        text_image=cv2.imread(file_path)
        print(CTCRec(text_image))

        cv2.imshow("text_image",text_image)
        cv2.waitKey(0)