# -*- coding: utf-8 -*-
import tensorflow as tf
import Input_Data
LEARNING_RATE = 0.0001
STEPS = 5000
BATCH = 50
N_CLASSES = 2
IMAGE_WEIGHT = 299
IMAGE_HEIGHT = 299
BATCH_SIZE = 10
CAPACITY = 100
EPOCH = 10

# INPUT_DATA = "F:\\Program\\Data_test\\dogvscat\\train"
INPUT_DATA = "F:\\Data\\Test\\test1025"

images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_WEIGHT, IMAGE_HEIGHT, 3], name="input_images")
labels = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES], name="labels")  # one_hot

train_image_list, train_label_list = Input_Data.get_image_label_list(INPUT_DATA)
train_image_bath, train_label_batch = \
        Input_Data.get_image_label_batch(train_image_list,
                                         train_label_list,
                                         IMAGE_WEIGHT,
                                         IMAGE_HEIGHT,
                                         BATCH_SIZE,
                                         CAPACITY)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        for i in range(50):
            images_batch, labels_batch = sess.run([train_image_bath, train_label_batch])
            print(labels_batch)
        # print(len(train_image_bath))
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)





