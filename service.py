# coding=utf-8
# =============================================
# @Time      : 2022-07-22 9:45
# @Author    : DongWei1998
# @FileName  : service.py
# @Software  : PyCharm
# =============================================
from flask import Flask, jsonify, request
from utils import parameter,UNet
import tensorflow as tf
import cv2
import numpy as np




class Predictor(object):
    def __init__(self, args):
        self.args = args
        # 构建模型
        self.unet_model = UNet.Unet(self.args)
        # 模型保存器
        checkpoint = tf.train.Checkpoint(network=self.unet_model)
        # 模型恢复
        checkpoint.restore(tf.train.latest_checkpoint(self.args.output_dir))
        self.args.logger.info(f'Loading model from {self.args.output_dir}')


    def predict_(self,images):
        # 构建模型
        logits,predictions = self.unet_model._predict_step(inputs=images)

        predict = tf.argmax(logits, -1)  # 预测类别下标（1表示细胞膜的位置，0表示其他）, [N,H,W]
        # todo 做其他的数据处理
        saveResult('tt_2.png', predict)
        return predict




Unlabelled = [0, 0, 0]
CellMembrane = [255, 255, 255]
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]

COLOR_DICT = np.array([
    Unlabelled, CellMembrane, Sky, Building, Pole, Road, Pavement,
    Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist
])

def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out

def saveResult(save_path, npyfile, num_class=2):
    for i, item in enumerate(npyfile):
        item = item[:, :, 0] if len(item.shape) == 3 else item  # 数据转换为二维的
        img = labelVisualize(num_class, COLOR_DICT, item)
        cv2.imwrite(save_path,img)
        # io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)


if __name__ == '__main__':

    app = Flask(__name__)

    app.config['JSON_AS_ASCII'] = False

    model = 'server'
    args = parameter.parser_opt(model)

    my_model = Predictor(args)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            args.logger.info(f'=======Receipt of the request=======')
            data = request.files
            # data = request.get_json()
            if 'input' not in data:
                return 'input字段不存在', 500
            # 解码
            data_info = data['input'].read()
            img_array = np.frombuffer(data_info, np.uint8)
            img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # image = cv2.imread(data['input'], cv2.COLOR_RGB2GRAY)
            image = tf.image.rgb_to_grayscale(img_array)
            image = np.array([list(image)], dtype=np.float)
            image /= 255.0  # 0 - 255 to 0.0 - 1.0
            args.logger.info(f'输入图片的形状为{image.shape}')
            if image is None:
                args.logger.error(f"not a image, please check your input")
                return jsonify(
                    {'code': 500,
                     'msg': 'not a image！！！'
                     })
            else:
                predictions = my_model.predict_(image)
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'predictions': str(predictions),
                })
        except Exception as e:
            return jsonify(
                {'code': 500,
                 'msg': e
                 })

    # 启动
    app.run(host='0.0.0.0', port=99884)