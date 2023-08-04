# coding=utf-8
# =============================================
# @Time      : 2022-07-20 17:10
# @Author    : DongWei1998
# @FileName  : data_help.py
# @Software  : PyCharm
# =============================================
import random

import cv2,base64
import numpy as np
import tensorflow as tf
import glob




class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,args, shuffle=True):
        self.args = args
        self.batch_size = self.args.batch_size
        self.data_img = sorted(glob.glob(self.args.train_data_file_image))
        self.data_mask = sorted(glob.glob(self.args.train_data_file_label))
        self.indexes = np.arange(len(self.data_img))
        self.shuffle = shuffle

        # 获取数据

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return int(tf.math.ceil(len(self.data_img) / float(self.batch_size)).numpy())

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_data_img = [self.data_img[k] for k in batch_indexs]
        batch_data_mask = [self.data_mask[k] for k in batch_indexs]

        # 生成数据
        X, Y = self.data_generation(batch_data_img, batch_data_mask)
        # self.args.logger.info("数据读入完成，数据形状为:{}-{}".format(X.shape, Y.shape))
        return X,Y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_data_img, batch_data_mask):
        images = []
        masks = []

        # 生成数据
        for data_img, data_mask in zip(batch_data_img, batch_data_mask):
            # x_train数据
            image = cv2.imread(data_img,cv2.COLOR_RGB2GRAY)
            image = tf.image.rgb_to_grayscale(image)
            image = np.array(image, dtype=np.float)
            # image = cv2.resize(image, self.args.input_image_shape)
            # image = image / 255.0
            # image = cv2.resize(image,self.args.input_image_shape)
            image = list(image)
            images.append(image)
            # y_train数据
            mask = cv2.imread(data_mask,cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.args.input_image_shape)
            mask = mask / 255.0

            mask = list(mask)
            masks.append(mask)

        return np.array(images,dtype=np.float), np.array(masks,dtype=np.int)



if __name__ == '__main__':


    # # # 数据生成器
    # # training_generator = DataGenerator(_)
    # # for (x, y) in training_generator:
    # #     print('x_batch:', x.shape)
    # #     print('y_batcxh:', y.shape)
    #
    # # image = cv2.imread('../datasets/train/image/13.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('../datasets/train/image/13.png', cv2.COLOR_RGB2GRAY)
    # image = tf.image.rgb_to_grayscale(image)
    # # image = cv2.resize(image, (512,512))
    # print(type(image),image.shape)
    # # if image != tf.float32:
    # #     # 数据的取值范围直接变成[0,1.0]
    # #     image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    # y_train数据
    masks = []
    mask = cv2.imread('../datasets/train/label/0.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (512,512))
    print(mask)
    mask = mask / 255.0
    print(mask.shape,type(mask))
    mask = list(mask)
    print(type(mask))
    print(masks.append(mask))
    print(masks)

    saveResult('ll.png', mask)


