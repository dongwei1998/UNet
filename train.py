# coding=utf-8
# =============================================
# @Time      : 2022-07-20 10:04
# @Author    : DongWei1998
# @FileName  : train.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
from utils import parameter
from utils import UNet,data_help,gpu_git
import tensorflow as tf
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.initial_learning_rate / (step + 1)


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred)

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)


def start_train():

    args = parameter.parser_opt(model='train')
    tf.device = gpu_git.check_gpus(mode=args.mode, logger=args.logger)
    # 加载数据
    dataset = data_help.DataGenerator(args)
    # 模型构建
    unet_model = UNet.Unet(args)

    # 定义优化
    optimizer = tf.keras.optimizers.Adam(
        MyLRSchedule(args.learning_rate),
        beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # 模型优化器
    unet_model.compile(
        optimizer=optimizer,
        loss=MaskedLoss(),
    )
    # 模型保存器
    checkpoint = tf.train.Checkpoint(network=unet_model, optimizer=optimizer)
    unet_model.save_ckpt_model(args.output_dir, checkpoint, args.model_ckpt_name)
    # 模型恢复
    if tf.train.latest_checkpoint(args.output_dir) == None:
        args.logger.info(f'The new training model from {args.output_dir}')
    else:
        # 模型恢复
        checkpoint.restore(tf.train.latest_checkpoint(args.output_dir))
        args.logger.info(f'Loading model from {args.output_dir}')
    n_step = 0
    for epoch in range(args.num_epochs):
        loss_epoch = []
        acc_epoch = []
        # dataset = [(np.random.rand(4, 256, 256, 1),
        #             tf.constant(list(np.random.choice([0, 1], size=262144, )), shape=[4, 256, 256]))]
        for images_input_batch, labels_target_batch in dataset:
            n_step += 1
            logits, predictions, end_points, total_loss, acc = unet_model._train_step(inputs=images_input_batch,
                                                                                      labels=labels_target_batch)
            # 查看各个层的形状
            if n_step == 1:
                for end_point in end_points:
                    args.logger.debug("【{}】层输出:{}".format(end_point, end_points[end_point].get_shape()))
            # 每个批次的loss
            loss_epoch.append(total_loss.numpy())
            acc_epoch.append(acc.numpy())
            args.logger.info('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                epoch + 1, n_step, total_loss.numpy(),acc.numpy()
            ))

            # 模型保存
            if n_step % args.ckpt_model_num == 0:
                unet_model.ckpt_manager.save()
                args.logger.info('epoch {}, save model at {}'.format(
                    epoch + 1, args.output_dir
                ))
            # if n_step % args.step_env_model == 0:
            #     unet_model.env_model(env_dataset=None)
        args.logger.info('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
            epoch + 1, n_step, unet_model._average(loss_epoch),unet_model._average(acc_epoch)
        ))


if __name__ == '__main__':
    start_train()
