# coding=utf-8
# =============================================
# @Time      : 2022-07-20 10:09
# @Author    : DongWei1998
# @FileName  : UNet.py
# @Software  : PyCharm
# =============================================

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization,ReLU,Conv2DTranspose,Dropout,Cropping2D
import numpy as np
from utils import parameter,data_help



# 上采样
class double_conv2d_bn(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=(2, 2), activation='relu',
                              padding="valid")
        self.conv2 = Conv2D(filters=128, kernel_size=(2, 2), activation='relu',
                            padding="valid")
        self.conv3 = Conv2D(filters=256, kernel_size=(2, 2), activation='relu',
                            padding="valid")

        self.conv2 = Conv2D(out_channels,kernel_size=kernel_size)

        self.bn1 = BatchNormalization(out_channels)
        self.bn2 = BatchNormalization(out_channels)

    def forward(self, x):
        out = ReLU(self.bn1(self.conv1(x)))
        out = ReLU(self.bn2(self.conv2(out)))
        return out


# 卷积层
class deconv2d_bn(layers.Layer):
    def __init__(self, out_channels,strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = tf.nn.conv2d_transpose(filters=out_channels,strides=strides)
        self.bn1 = BatchNormalization(out_channels)

    def forward(self, x):
        out = ReLU(self.bn1(self.conv1(x)))
        return out


class Unet(tf.keras.Model):
    def __init__(self,args):
        super(Unet, self).__init__()
        self.args = args
        self.conve_1_1 = Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")
        self.conve_1_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding="same")
        self.pool_1 = MaxPooling2D(pool_size=(2, 2), padding='valid')

        self.conve_2_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding="same")
        self.conve_2_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding="same")
        self.pool_2 = MaxPooling2D(pool_size=(2, 2), padding='valid')

        self.conve_3_1 = Conv2D(filters=256, kernel_size=3, activation='relu', padding="same")
        self.conve_3_2 = Conv2D(filters=256, kernel_size=3, activation='relu', padding="same")
        self.pool_3 = MaxPooling2D(pool_size=(2, 2), padding='valid')

        self.conve_4_1 = Conv2D(filters=512, kernel_size=3, activation='relu', padding="same")
        self.conve_4_2 = Conv2D(filters=512, kernel_size=3, activation='relu', padding="same")
        self.dropout_1 = Dropout(rate=0.5)
        self.pool_4 = MaxPooling2D(pool_size=(2, 2), padding='valid')

        self.conve_5_1 = Conv2D(filters=1024, kernel_size=3, activation='relu', padding="same")
        self.conve_5_2 = Conv2D(filters=1024, kernel_size=3, activation='relu', padding="same")
        self.dropout_2 = Dropout(rate=0.5)


        self.upsample_1 = Conv2DTranspose(filters=512, kernel_size=2,strides=(2,2))
        self.conve_6_1 = Conv2D(filters=512, kernel_size=3, activation='relu', padding="same")
        self.conve_6_2 = Conv2D(filters=512, kernel_size=3, activation='relu', padding="same")

        self.upsample_2 = Conv2DTranspose(filters=256, kernel_size=2, strides=(2, 2))
        self.conve_7_1 = Conv2D(filters=256, kernel_size=3, activation='relu', padding="same")
        self.conve_7_2 = Conv2D(filters=256, kernel_size=3, activation='relu', padding="same")

        self.upsample_3 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2, 2))
        self.conve_8_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding="same")
        self.conve_8_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding="same")

        self.upsample_4 = Conv2DTranspose(filters=64, kernel_size=2, strides=(2, 2))
        self.conve_9_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding="same")
        self.conve_9_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding="same")



        self.final_conve = Conv2D(filters=2, kernel_size=1, activation='relu', padding="same")


    def _encoder(self,inputs):
        end_points = {}
        # encoder
        conve_1_1 = self.conve_1_1(inputs)
        conve_1_2 = self.conve_1_2(conve_1_1)
        end_points['block1'] = conve_1_2
        pool_1 = self.pool_1(conve_1_2)


        conve_2_1 = self.conve_2_1(pool_1)
        conve_2_2 = self.conve_2_2(conve_2_1)
        end_points['block2'] = conve_2_2
        pool_2 = self.pool_2(conve_2_2)


        conve_3_1 = self.conve_3_1(pool_2)
        conve_3_2 = self.conve_3_2(conve_3_1)
        end_points['block3'] = conve_3_2
        pool_3 = self.pool_3(conve_3_2)

        conve_4_1 = self.conve_4_1(pool_3)
        conve_4_2 = self.conve_4_2(conve_4_1)
        conve_4_2 = self.dropout_1(conve_4_2)
        end_points['block4'] = conve_4_2
        pool_4 = self.pool_4(conve_4_2)

        conve_5_1 = self.conve_5_1(pool_4)
        conve_5_2 = self.conve_5_2(conve_5_1)
        conve_5_2 = self.dropout_1(conve_5_2)
        end_points['block5'] = conve_5_2

        return end_points,conve_5_2


    def _crop(self,A,B):
        # 获取当前特征图的尺寸
        _, x_height, x_width, _ = A.shape
        # 获取要融合的特征图的尺寸
        _, s_height, s_width, _ = B.shape
        # 获取特征图的大小差异
        h_crop = s_height - x_height
        w_crop = s_width - x_width
        # 若特征图大小相同不进行裁剪
        if h_crop == 0 and w_crop == 0:
            B = B
        # 若特征图大小不同，使级联时像素大小一致
        else:
            # 获取特征图裁剪后的特征图的大小
            cropping = ((h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2))
            # 特征图裁剪
            B = Cropping2D(cropping=cropping)(B)
        return B



    # 每个批次的平均loss
    def _average(self,epoch_loss):
        return sum(epoch_loss) / len(epoch_loss)

    def _decoder(self,end_points,conve_5_2):


        # 上采样
        up_net_1 = self.upsample_1(conve_5_2)
        # crop
        crop_1 = self._crop(up_net_1, end_points['block4'])
        cat_5_4 = tf.concat([up_net_1, crop_1], axis=-1)
        conve_6_1 = self.conve_6_1(cat_5_4)
        conve_6_2 = self.conve_6_2(conve_6_1)
        end_points['block6'] = conve_6_2


        up_net_2 = self.upsample_2(conve_6_2)
        crop_2 = self._crop(up_net_2, end_points['block3'])
        cat_6_3 = tf.concat([up_net_2, crop_2], axis=-1)
        conve_7_1 = self.conve_7_1(cat_6_3)
        conve_7_2 = self.conve_7_2(conve_7_1)
        end_points['block7'] = conve_7_2

        up_net_3 = self.upsample_3(conve_7_2)
        crop_3 = self._crop(up_net_3, end_points['block2'])
        cat_7_2 = tf.concat([up_net_3, crop_3], axis=-1)
        conve_8_1 = self.conve_8_1(cat_7_2)
        conve_8_2 = self.conve_8_2(conve_8_1)
        end_points['block8'] = conve_8_2

        up_net_4 = self.upsample_4(conve_8_2)
        crop_4 = self._crop(up_net_4, end_points['block1'])
        cat_8_1 = tf.concat([up_net_4, crop_4], axis=-1)
        conve_9_1 = self.conve_9_1(cat_8_1)
        conve_9_2 = self.conve_9_2(conve_9_1)
        conve_9_3 = self.final_conve(conve_9_2)
        end_points['block9'] = conve_9_3

        return end_points,conve_9_3


    def save_ckpt_model(self,output_dir,checkpoint,model_ckpt_name):
        # 模型保存
        # ckpt管理器
        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=output_dir,
            max_to_keep=2,
            checkpoint_name=model_ckpt_name)


    def _train_step(self, inputs ,labels):

        with tf.GradientTape() as tape:
            # 编码部分  卷积 池化
            end_points, conve_5_2 = self._encoder(inputs)

            # 解码部分
            end_points,logits = self._decoder(end_points,conve_5_2)


            # 得到预测值
            predictions = tf.nn.softmax(logits, name='prob')

            # 获取损失
            # predict = tf.argmax(logits, -1)
            # step_loss = self.loss(labels,logits)
            step_loss = self.get_loss(labels,logits)

            # 优化器
            variables = self.trainable_variables
            gradients = tape.gradient(step_loss, variables)
            gradients = tf.distribute.get_replica_context().all_reduce('sum', gradients)
            # Processing aggregated gradients.
            self.optimizer.apply_gradients(grads_and_vars=zip(gradients, variables),
                                      experimental_aggregate_gradients=False)

            acc = self.accuracy(logits, labels)




        return logits, predictions, end_points,step_loss,acc

    def env_model(self,env_dataset):
        loss_epoch = []
        step = 0
        for input_batch, target_batch in env_dataset:
            step += 1
            logits, predictions, end_points, step_loss, acc = self._train_step(input_batch, target_batch)
            # 每个时间步的loss
            self.args.logger.info('env model batch {}, loss:{:.4f}'.format(
                step, step_loss.numpy()))
            loss_epoch.append(step_loss.numpy())
        self.args.logger.info('Env load loss:{:.4f}'.format(self._average(loss_epoch)))

    def get_loss(self, labels,logits ):
        """
        构建损失函数
        :param logits: [N, H, W, num_class] --> 预测的置信度
        :param labels: [N, H, W] --> 实际mask类别标签
        :return:
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss, name='loss')
        return loss


    def accuracy(self, logits, labels):
        """
        构建准确率
        :param logits: [N, H, W, num_class] --> 预测的置信度
        :param labels: [N, H, W] --> 实际mask类别标签
        :return:
        """

        pred = tf.argmax(logits, -1, output_type=labels.dtype)  # 选择最大的置信度作为预测值
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))
        return accuracy


    def _predict_step(self, inputs):
        with tf.GradientTape() as tape:
            # 编码部分  卷积 池化
            end_points, conve_5_2 = self._encoder(inputs)

            # 解码部分
            end_points,logits = self._decoder(end_points,conve_5_2)


            # 得到预测值
            predictions = tf.nn.softmax(logits,name='prob')




            return logits,predictions



if __name__ == '__main__':
    args = parameter.parser_opt(model='train')
    model = Unet(args)
    inp = np.random.rand(1,572,572,3)
    logits, predictions, end_points = model(inp)


    print(logits,predictions)

    # model.summary()