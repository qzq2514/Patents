import tensorflow as tf
import tensorflow.contrib.slim as slim

class CTCRecognizer(object):
    def __init__(self, is_training,keep_prop,num_classes,num_hidden):
        self.is_training=is_training
        self.num_classes=num_classes
        self.num_hidden=num_hidden
        self.keep_prop=keep_prop

    def preprocess(self, inputs):
        #将transpose的操作写在预处理中,这样以后测试时候不用每次都transpose
        # tran_inputs=tf.transpose(inputs,[0,2,1,3])
        #
        # NORMALIZER=0.017
        # processed_inputs = tf.to_float(tran_inputs)
        # red, green, blue = tf.split(processed_inputs, num_or_size_splits=3, axis=3)
        # preprocessed_input = (tf.multiply(blue, 0.2989) +tf.multiply(green, 0.5870)+
        #                       tf.multiply(red, 0.1140))* NORMALIZER
        #preprocessed_input=tf.squeeze(preprocessed_input,axis=3)
        return inputs

    def block(self,inputs,output_channels,stride=1):
        res_channels=output_channels-output_channels/2

        block_net=slim.separable_convolution2d(inputs,None,[3,3],stride=stride)
        block_net=slim.batch_norm(block_net)

        block_net=slim.convolution2d(block_net,output_channels/2,[3,3])

        block_net = slim.batch_norm(block_net)
        res_net=slim.convolution2d(inputs,res_channels,[3,3],stride=stride)

        return tf.concat([block_net,res_net],axis=3)


    def LPR_small_block(self,inputs,out_channals):
        # channels=inputs.get_shape().as_list()[-1]

        small_block_net = slim.convolution2d(inputs,
                                             num_outputs=out_channals/4,kernel_size=[1,1],stride=1)
        small_block_net = slim.convolution2d(small_block_net,
                                             num_outputs=out_channals / 4, kernel_size=[3,1],stride=1)
        small_block_net = slim.convolution2d(small_block_net,
                                             num_outputs=out_channals / 4, kernel_size=[1, 3],stride=1)
        small_block_net = slim.convolution2d(small_block_net,
                                             num_outputs=out_channals, kernel_size=[1, 1],stride=1)
        return small_block_net


    def inference(self,inputs,seq_length):

        print("CTCRecognizer_new")
        with slim.arg_scope(self.CTC_arg_scope(is_training=self.is_training,
                                               batch_norm_decay=0.8)):
            print("input:",inputs)
            net = slim.convolution2d(inputs, 64, kernel_size=[3,3], stride=1)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=1)
            net = self.LPR_small_block(net,128)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=[2, 1])
            net = self.LPR_small_block(net, 256)
            net = self.LPR_small_block(net, 256)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=[1, 2])
            net = slim.dropout(net,keep_prob=self.keep_prop)
            net = slim.convolution2d(net,256,kernel_size=[4, 1],stride=1)
            net = slim.dropout(net, keep_prob=self.keep_prop)
            net = slim.convolution2d(net, self.num_classes, kernel_size=[4, 1], stride=1)
            # logits = tf.reduce_mean(net, axis=2)
            logits = slim.avg_pool2d(net, kernel_size=[1,15],stride=[1,15])
            logits=tf.squeeze(logits, [2], name='SpatialSqueeze')

            # 必须保证最终的输出是(max_time_step,batch_size,num_class)的形式供后面计算CTC Loss
            logits = tf.transpose(logits, (1, 0, 2))
            #在max_step_downsampling_num=2时,保证logits的shape为:(30 , b , 38)

            print("logits:", logits)
            # input("Pause")
        return logits

    def beam_searcn(self,logits,seq_len,is_merge=False):
        decoded_logits,log_prob=tf.nn.ctc_beam_search_decoder(logits,seq_len,merge_repeated=is_merge)
        return decoded_logits

    def decode_a_seq(self,indexes, spars_tensor,chars):
        decoded = []
        for m in indexes:
            # print("m:",m)
            str_id = spars_tensor[1][m]
            print(m,"---",str_id)
            str = chars[str_id]
            decoded.append(str)
        return decoded #sparse_tensor[0]是N*2的indices

    def decode_sparse_tensor(self,sparse_tensor,chars):
        decoded_indexes = list()
        current_i = 0
        current_seq = []
        # print(sparse_tensor)
        for offset, i_and_index in enumerate(sparse_tensor[0]):  # sparse_tensor[0]是N*2的indices
            i = i_and_index[0]                                   # 一行是一个样本
            # print("i_and_index:",i_and_index)
            if i != current_i:                                   # current_is是当前样本的id
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = list()                             # current_seq是当前样本预测值在sparse_tensor的values中对应的下标
            current_seq.append(offset)                           # 之后通过下标就可以从sparse_tensor中找到对应的值
        decoded_indexes.append(current_seq)
        result = []
        for index in decoded_indexes:
            result.append(self.decode_a_seq(index, sparse_tensor,chars))
        return result

    def get_edit_distance_mean(self,decoded_logits_placeholder,sparse_labels_placeholder):
        # 计算两个稀疏矩阵代表的序列的编辑距离,在预测和标签在样本数量上长度不匹配时可以作为一种评判模型的指标,没有他可无妨
        edit_distance_mean = tf.reduce_mean(tf.edit_distance(tf.cast(decoded_logits_placeholder[0], tf.int32), sparse_labels_placeholder))
        return edit_distance_mean

    #传入实值,而非Tensor
    def get_accuarcy(self,decoded_logits,sparse_labels,chars):
        #通过稀疏矩阵解析得到最终的预测结果
        sparse_labels_list = self.decode_sparse_tensor(sparse_labels, chars)
        decoded_list=self.decode_sparse_tensor(decoded_logits,chars)
        true_numer = 0

        if len(decoded_list) != len(sparse_labels_list):
            # print("len(decoded_list)", len(decoded_list), "len(sparse_labels_list)", len(sparse_labels_list),
            #       " test and detect length desn't match")
            return None       #edit_distance起作用

        for idx, pred_number in enumerate(decoded_list):
            groundTruth_number = sparse_labels_list[idx]
            cur_correct = (pred_number == groundTruth_number)
            info_str="{}:{}-({}) <-------> {}-({})".\
                format(cur_correct,groundTruth_number,len(groundTruth_number),pred_number,len(pred_number))
            print(info_str)
            if cur_correct:
                true_numer = true_numer + 1

        accuary=true_numer * 1.0 / len(decoded_list)
        return accuary

    # logits:(24, 50, 67)
    # sparse_groundtrouth:是tf.SparseTensor类型,三元组,其中包括(indices, values, shape)
    def loss(self,logits,sparse_groundtrouth,seq_len):
        loss_all=tf.nn.ctc_loss(labels=sparse_groundtrouth,inputs=logits,sequence_length=seq_len)
        loss_mean=tf.reduce_mean(loss_all)

        # tf.add_to_collection("Loss", loss_mean)
        # loss_all = tf.add_n(tf.get_collection("Loss"), name="total_loss")

        #计算正则损失
        # regularization_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # loss_all = loss_mean+regularization_loss
        return loss_mean


    def CTC_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):
        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        with slim.arg_scope(
            [slim.convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.max_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc
