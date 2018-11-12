# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
from PIL import Image
from datetime import datetime
import math
import time
import Input_Data
import InceptionV4
import cv2
import os
slim = tf.contrib.slim

LEARNING_RATE = 0.0001
STEPS = 5000
BATCH = 50
N_CLASSES = 2
IMAGE_WEIGHT = 299
IMAGE_HEIGHT = 299
BATCH_SIZE = 32
CAPACITY = 32
EPOCH = 10

craterDir = "train"
INPUT_DATA = "F:\\Program\\Data_test\\dogvscat\\train"
INPUT_DATA_VAL = "F:\\Program\\Data_test\\dogvscat\\val"
CKPT_FILE = "./Model_parameter/inception_v4.ckpt"

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV4/Logits,InceptionV4/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV4/Logits,InceptionV4/AuxLogit'


def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    variables_to_restore = []

    # 枚举inception_V3模型中所有的参数，然后判断是否需要从加载列表中移除。
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(",")]
    variable_to_train = []
    # 枚举所有需要训练的参数前缀， 并通过所有这些前缀找到所有参数。
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variable_to_train.extend(variables)
    return variable_to_train


def accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor,
    """
    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        acc = tf.reduce_mean(correct) * 100.0
        # tf.summary.scalar(scope + '/accuracy', accuracy)
    return acc


def main():
    train_image_list, train_label_list = Input_Data.get_image_label_list(INPUT_DATA)
    val_image_list, val_label_list = Input_Data.get_image_label_list(INPUT_DATA_VAL)
    train_image_bath, train_label_batch = \
        Input_Data.get_image_label_batch(train_image_list,
                                         train_label_list,
                                         IMAGE_WEIGHT,
                                         IMAGE_HEIGHT,
                                         BATCH_SIZE,
                                         CAPACITY)
    val_image_bath, val_label_batch = \
        Input_Data.get_image_label_batch(val_image_list,
                                         val_label_list,
                                         IMAGE_WEIGHT,
                                         IMAGE_HEIGHT,
                                         BATCH_SIZE,
                                         CAPACITY)
    batch_total = int(len(train_image_list) / BATCH_SIZE) + 1

    images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_WEIGHT, IMAGE_HEIGHT, 3], name="input_images")
    labels = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES], name="labels")  # one_hot

    with slim.arg_scope(InceptionV4.inception_v4_arg_scope()):
        logits, _ = InceptionV4.inception_v4(images, num_classes=N_CLASSES)
    trainable_variables = get_trainable_variables()

    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).\
            minimize(total_loss, var_list=trainable_variables)
    acc = accuracy(logits, labels)

    # 定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        get_tuned_variables(),
        ignore_missing_vars=True)
    # 定义保存新的训练好的模型函数
    saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        init = tf.global_variables_initializer()
        sess.run(init)

        # 加载谷歌已经训练好的模型
        print("Loading tuned variables from %s" % CKPT_FILE)
        load_fn(sess)

        # net_vars = variables_to_restore
        # saver_net = tf.train.Saver(net_vars)
        # checkpoint_path = 'F:\\Data\\Test\\log4\\inception_v4.ckpt'
        # saver_net.restore(sess, checkpoint_path)

        try:
            for i in range(EPOCH):  # 每一轮迭代
                if coord.should_stop():
                    break
                print('****** Epoch{}/{} ******'.format(i, EPOCH))
                for step in range(batch_total):  # 每一个batch
                    # 获取每一个batch中batch_size个样本和标签
                    images_batch, labels_batch = sess.run([train_image_bath, train_label_batch])
                    _, tra_loss, tra_acc = sess.run([train_step, total_loss, acc],
                                                    feed_dict={images: images_batch,
                                                               labels: labels_batch})
                    if step % 10 == 0 or (step + 1) == batch_total:
                        tra_loss = format(tra_loss, "3.3f")
                        # tra_acc = format(tra_acc, "3.3%")
                        print('{}/{}[************************************] - tra_loss: {} tra_acc: {}%'
                              .format(step, batch_total, tra_loss, tra_acc))
                    if step % 50 == 0 or (step + 1) == batch_total:
                        val_images, val_labels = sess.run([val_image_bath, val_label_batch])
                        val_loss, val_accuracy = sess.run([total_loss, acc],
                                                          feed_dict={images: val_images,
                                                                     labels: val_labels})
                        val_loss = format(val_loss, "3.3f")
                        # val_accuracy = format(val_accuracy, "3.3%")
                        print('{}/{}[====================================] - val_loss: {} val_acc: {}%'
                              .format(step, batch_total, val_loss, val_accuracy, "0.2f"))
                        # saver2.save(sess, model_path, global_step=i)
                        # saver2.save(sess, model_path, global_step=i, write_meta_graph=False)
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)

# def evaluation():
#     with tf.


if __name__ == "__main__":
    main()


