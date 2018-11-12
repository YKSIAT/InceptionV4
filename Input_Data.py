# -*- coding: UTF-8 -*-
import tensorflow as tf
import glob
import random
import math
import os
import numpy as np
from matplotlib.cbook import flatten

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 警告显示等级，这里只显示Waring,Error. 3 只显示Error.

source_path = "F:\\Data\\Test\\test1025"

IMAGE_WEIGHT = 299
IMAGE_HEIGHT = 299
BATCH_size = 32
CAPACITY = 32


def get_image_label_list(src_path):
    """

    :param src_path:
    :return:
    """
    """
    input：The path of source datasets.
    Return: Image directory list and corresponding labels list.

    """
    current_label = 0
    category_list = []
    image_list = []
    label_list = []
    list_images_0 = []
    list_labels_0 = []
    list_images_1 = []
    list_labels_1 = []
    images_list_0_Aug = []
    images_label_0_Aug = []
    images_list_1_Aug = []
    images_label_1_Aug = []

    num = []

    # 遍历文件夹下所有你的文件夹以列表的形式返回,sub_dirs[0]为根目录
    sub_dirs = [x[0] for x in os.walk(src_path)]
    # print(sub_dirs)
    sub_dirs.sort()  # 默认升序排列
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        category_list.append(os.path.basename(sub_dir))
    print("Default category number is:")
    # print(category_list)
    for idx, val in enumerate(category_list):
        print('category name: {},label name :{}'.format(val, idx))
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        dir_name = os.path.basename(sub_dir)
        # 获取当前目录下所有的jpg图片
        extensions = ['jpg', 'jpeg']
        sub_file_list = []
        sub_label_list = []
        for extension in extensions:
            file_glob = os.path.join(src_path, dir_name, '*.' + extension)
            sub_file_list.extend(glob.glob(file_glob))
        sub_label_list.extend([current_label] * len(sub_file_list))
        if current_label == 0:
            list_images_0.extend(sub_file_list)
            list_labels_0.extend(sub_label_list)
        else:
            list_images_1.extend(sub_file_list)
            list_labels_1.extend(sub_label_list)
        # image_list.extend(sub_file_list)
        # label_list.extend(sub_label_list)
        current_label += 1
        # print("A total of %d pictures in the %s folder" % (len(sub_file_list), sub_dir))
        print("A total of %d pictures in the %s folder" % (len(sub_file_list), dir_name))
    print("A total of %d images " % (len(image_list)))
    print("list_images_0 is :", len(list_images_0))
    print("list_images_1 is :", len(list_images_1))
    if len(list_images_0) > len(list_images_1):
        num.append(int(len(list_images_0) / len(list_images_1)))    # Multiple
        num.append(int(len(list_images_0) % len(list_images_1)))    # Difference
        images_list_1_Aug.extend(list_images_1 * num[0])
        images_list_1_Aug.extend(random.sample(list_images_1, num[1]))
        images_label_1_Aug.extend(list_labels_1 * num[0])
        images_label_1_Aug.extend(random.sample(list_labels_1, num[1]))
        images_list_0_Aug.extend(list_images_0)
        images_label_0_Aug.extend(list_labels_0)
        print("num is:", num)
        num.clear()
    else:
        num.append(int(len(list_images_1) / len(list_images_0)))    # Multiple
        num.append(int(len(list_images_1) % len(list_images_0)))    # Difference
        images_list_0_Aug.extend(list_images_0 * num[0])
        images_list_0_Aug.extend(random.sample(list_images_0, num[1]))
        images_label_0_Aug.extend(list_labels_0 * num[0])
        images_label_0_Aug.extend(random.sample(list_labels_0, num[1]))
        images_list_1_Aug.extend(list_images_1)
        images_label_1_Aug.extend(list_labels_1)
        print("num is:", num)
        num.clear()
    print("images_list_0_Aug is :", len(images_list_0_Aug))
    print("labels_list_0_Aug is :", len(images_label_0_Aug))
    print("images_list_1_Aug is :", len(images_list_1_Aug))
    print("labels_list_1_Aug is :", len(images_label_1_Aug))
    image_list.extend(list(flatten(zip(images_list_0_Aug, images_list_1_Aug))))
    label_list.extend(list(flatten(zip(images_label_0_Aug, images_label_1_Aug))))
    print("image_list is:", image_list)
    print("label_list is:", label_list)

    temp = np.array([image_list, label_list])
    print(temp)
    temp = temp.transpose()
    # np.random.shuffle(temp)
    final_image_list = list(temp[:, 0])
    final_label_list = list(temp[:, 1])
    final_label_list = [int(i) for i in final_label_list]
    # for i in final_image_list:
    #     print(i)
    print(final_label_list)

    return final_image_list, final_label_list


def get_image_label_batch(images_list, labels_list, image_weight, image_height, batch_size, capacity):
    """

    :param images_list:
    :param labels_list:
    :param image_weight:
    :param image_height:
    :param batch_size:
    :param capacity:
    :return:
    """
    image_list = tf.cast(images_list, tf.string)
    label_list = tf.cast(labels_list, tf.int32)
    # classes = tf.cast(classes, tf.int16)
    print("hah is:", labels_list)

    input_queue = tf.train.slice_input_producer([image_list, label_list], num_epochs=None, shuffle=False)

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # image = tf.image.resize_image_with_crop_or_pad(image, image_weight, image_height)
    image = tf.image.resize_images(image, [image_weight, image_height], method=0)     # 0:Bilinear interpolation
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    classes = 2
    label_batch = tf.one_hot(label_batch, depth=classes)
    label_batch = tf.reshape(tf.cast(label_batch, dtype=tf.int32), [batch_size, classes])
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     for i in range(50):
    #         k = sess.run(input_queue)
    #         print('************************')
    #         print(i, k)
    #
    #     coord.request_stop()
    #     coord.join(threads)
    return image_batch, label_batch


# if __name__ == "__main__":
    # images, labels = get_image_label_list(source_path)
    # sess = tf.Session()
    # get_image_label_batch(images, labels, IMAGE_WEIGHT, IMAGE_HEIGHT, BATCH_size, CAPACITY)
    # sess.close()
    # get_image_label_batch(images, labels, IMAGE_WEIGHT, IMAGE_HEIGHT, BATCH_size, CAPACITY)
#     print("Done！")








