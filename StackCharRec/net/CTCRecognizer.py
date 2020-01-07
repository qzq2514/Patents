import tensorflow as tf
import tensorflow.contrib.slim as slim

class CTCRecognizer(object):
    def __init__(self, is_training,keep_prop,num_classes,num_hidden):
        self.is_training=is_training
        self.num_classes=num_classes
        self.num_hidden=num_hidden
        self.keep_prop=keep_prop
        self.reduction_ratio=16

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

    def SE_Moudle(self,inputs):

        num_channels=inputs.get_shape().as_list()[-1]
        with tf.variable_scope("SE_Moudle"):
            moudle = tf.reduce_mean(inputs,axis=[1,2],keepdims=True,name="global_avg_pooling")
            moudle = slim.convolution2d(moudle,num_outputs=num_channels/self.reduction_ratio,
                                        kernel_size=1,stride=1,normalizer_fn=None,scope="dim_decrease")
            channel_weights = slim.convolution2d(moudle,num_outputs=num_channels,kernel_size=1,
                                        stride=1,activation_fn=tf.nn.sigmoid,normalizer_fn=None,
                                        scope="dim_increase")
            scale = inputs * channel_weights
            return scale

    def block(self,inputs,output_channels,stride=1,name="None"):
        with tf.variable_scope(name):

            res_channels=output_channels-output_channels/2

            block_net=slim.separable_convolution2d(inputs,None,[3,3],stride=stride)

            block_net=slim.batch_norm(block_net)

            block_net=slim.convolution2d(block_net,output_channels/2,[3,3])

            block_net = slim.batch_norm(block_net)

            # se_block_net=self.SE_Moudle(block_net)

            res_net=slim.convolution2d(inputs,res_channels,[3,3],stride=stride)

            return tf.concat([block_net,res_net],axis=3)


    def bootleneck(self,inputs,bottleneck_channel_upsample_rate,bottleneck_output_channels,stride,name):
        with tf.variable_scope(name):
            input_channels=inputs.get_shape().as_list()[-1]

            # MobileNetV2的第一个核心:reverted residual
            # ResNet在bottleneck中是先降维,再升维度,但是MobileNetV2在是先升维再降维
            # 论文中bottleneck_channel_upsample_rate=6
            bottleneck=slim.convolution2d(inputs,int(input_channels*bottleneck_channel_upsample_rate),
                                          kernel_size=[1,1],stride=1,scope="pw_conv1")
            bottleneck=slim.batch_norm(bottleneck,scope="bn1")

            # num_outputs:pointwise 卷积的卷积核个数，如果为空，将跳过pointwise卷积的步骤,
            # 后面我们通过一般的1x1卷积自己实现pointwise
            bottleneck = slim.separable_convolution2d(bottleneck,num_outputs=None,stride=stride,
                                                      depth_multiplier=1,kernel_size=[3,3],scope="dw_conv")
            bottleneck = slim.batch_norm(bottleneck, scope="bn2")


            bottleneck = slim.convolution2d(bottleneck, bottleneck_output_channels,activation_fn=None,
                                            kernel_size=[1, 1], stride=1, scope="pw_conv2")

            # bottleNeck的第二个pointwise_conv不使用激活函数
            # 也是Mobile的第二个核心:linear bottleneck
            # 因为非线性激活函数在高维空间内可以保证非线性，但在低维空间内非线性降低，会导致低维空间的信息损失
            bottleneck = slim.batch_norm(bottleneck, scope="bn3",activation_fn=None)

            bottle_channels=bottleneck.get_shape().as_list()[-1]
            if bottle_channels==input_channels:
                bottleneck_output=tf.add(bottleneck,inputs)
            else:
                bottleneck_output=bottleneck
            return bottleneck_output

    def inference(self,inputs,seq_length):
        print("CTCRecognizer")
        with slim.arg_scope(self.mobileNet_arg_scope()):
            with slim.arg_scope(
                    [slim.convolution2d],
                    weights_initializer=slim.initializers.xavier_initializer(),
                    biases_initializer=slim.init_ops.zeros_initializer(),
                    weights_regularizer=slim.l2_regularizer(0.005),
                    activation_fn=None) as sc1:
                with slim.arg_scope(
                        [slim.batch_norm], is_training=self.is_training,
                        activation_fn=tf.nn.relu, fused=True, decay=0.95) as sc2:
                    #input:(b, 60, 30, 3)   NHWC

                    # 一定要等到batch_norm的均值和方差稳定了,在测试时才能有高精度
                    # 为了使得均值方差快速稳定,可以另batch_norm的decay变小，0.95不够就0.90甚至更小
                    net = slim.convolution2d(inputs, num_outputs=32, kernel_size=[3, 3], stride=2,
                                             padding='SAME', scope='conv_1')
                    net = slim.batch_norm(net, scope="b1")
                    net = self.bootleneck(net, 1, 16, 1, "bottleneck1")
                    net = self.bootleneck(net, 6, 24, 2, "bottleneck2_1")
                    net = self.bootleneck(net, 6, 24, 1, "bottleneck2_2")
                    print("net1:",net)

                    net = self.bootleneck(net, 6, 32, 2, "bottleneck3_1")
                    net = self.bootleneck(net, 6, 32, 1, "bottleneck3_2")
                    net = self.bootleneck(net, 6, 32, 1, "bottleneck3_3")
                    print("net2:", net)

                    net = self.bootleneck(net, 6, 64, 2, "bottleneck4_1")
                    net = self.bootleneck(net, 6, 64, 1, "bottleneck4_2")
                    net = self.bootleneck(net, 6, 64, 1, "bottleneck4_3")
                    net = self.bootleneck(net, 6, 64, 1, "bottleneck4_4")
                    print("net3:", net)

                    net = self.bootleneck(net, 6, 96, 1, "bottleneck5_1")
                    net = self.bootleneck(net, 6, 96, 1, "bottleneck5_2")
                    net = self.bootleneck(net, 6, 96, 1, "bottleneck5_3")
                    print("net4:", net)

                    net = self.bootleneck(net, 6, 160, 2, "bottleneck6_1")
                    net = self.bootleneck(net, 6, 160, 1, "bottleneck6_2")
                    net = self.bootleneck(net, 6, 160, 1, "bottleneck6_3")
                    print("net5:", net)

                    #input("Pause")
                    net = self.bootleneck(net, 6, 320, 1, "bottleneck7_1")

                    net = slim.convolution2d(net, num_outputs=1280, kernel_size=[3, 3],
                                             stride=1, scope='conv_2')
                    net = slim.batch_norm(net, scope="b2")
                    net = slim.avg_pool2d(net, kernel_size=[7, 7], stride=1)
                    net = slim.convolution2d(net, num_outputs=self.num_classes, kernel_size=[3, 3],
                                             stride=1, scope='conv_3')

                    logits = tf.reshape(net, shape=[-1, self.num_classes])

                # 必须保证最终的输出是(max_time_step,batch_size,num_class)的形式
                # 供后面计算CTC Loss
                # logits = tf.transpose(rnn_reshape_inv, (1, 0, 2)) #(15,b,28) HNC
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

    def mobileNet_arg_scope(self,weight_decay=0.0):
      with slim.arg_scope(
          [slim.convolution2d, slim.separable_convolution2d],
          weights_initializer=slim.initializers.xavier_initializer(),
          biases_initializer=slim.init_ops.zeros_initializer(),
          weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
        return sc