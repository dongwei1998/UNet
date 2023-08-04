# coding=utf-8
# =============================================
# @Time      : 2022-07-20 17:08
# @Author    : DongWei1998
# @FileName  : parameter.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config
import shutil



# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def parser_opt(model):
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    args.mode = os.environ.get("mode")
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    # 清除模型以及可视化文件
    if model == 'train':
        args.model_ckpt_name = os.environ.get('model_ckpt_name')

        args.train_data_file_image = os.path.join(os.environ.get('train_data_file'),'image/*.png')
        args.train_data_file_label = os.path.join(os.environ.get('train_data_file'),'label/*.png')

        args.text_data_file_image = os.path.join(os.environ.get('text_data_file'), 'image/*.png')
        args.text_data_file_image = os.path.join(os.environ.get('text_data_file'), 'label/*.png')
        args.batch_size = int(os.environ.get('batch_size'))
        args.input_image_shape = tuple(int(i) for i in os.environ.get('input_image_shape').split(','))
        args.num_calss = int(os.environ.get('num_calss'))
        args.output_dir = os.environ.get('output_dir')
        args.model_ckpt_name = os.environ.get('model_ckpt_name')
        args.num_epochs = int(os.environ.get('num_epochs'))
        args.learning_rate = float(os.environ.get('learning_rate'))
        args.ckpt_model_num = int(os.environ.get('ckpt_model_num'))
        args.step_env_model = int(os.environ.get('step_env_model'))
    elif model =='env':
        pass
    elif model == 'server':
        args.batch_size = int(os.environ.get('batch_size'))
        args.input_image_shape = tuple(int(i) for i in os.environ.get('input_image_shape').split(','))
        args.num_calss = int(os.environ.get('num_calss'))
        args.output_dir = os.environ.get('output_dir')
        args.model_ckpt_name = os.environ.get('model_ckpt_name')
        args.num_epochs = int(os.environ.get('num_epochs'))
        args.learning_rate = float(os.environ.get('learning_rate'))
    else:
        raise print('请给定model参数，可选【traian env test】')
    return args


if __name__ == '__main__':
    args = parser_opt('train')