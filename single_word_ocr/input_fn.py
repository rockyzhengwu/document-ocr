#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import tensorflow as tf

def _decode_record(record_proto, aug=False):
  feature_map = {
          'images': tf.FixedLenFeature((), tf.string),
          'labels' : tf.VarLenFeature(tf.int64),
          'imagenames': tf.FixedLenFeature((), tf.string),
          }
  features = tf.parse_single_example(record_proto, features=feature_map)
  images = tf.image.decode_png(features['images'], channels=1)

  images = tf.image.resize_images(images, [32, 32] )
  images.set_shape([32, 32, 1])
  images = tf.cast(images, tf.float32)
  labels = tf.cast(features['labels'], tf.int32)

  #images = _image_augmentation(images)
  example = {
          "images": images / 255.0 ,
          "labels" : tf.squeeze(tf.sparse_tensor_to_dense(labels)),
          }
  return example

def _image_augmentation(image):
  #image = tf.image.random_flip_up_down(image)
  image = tf.image.random_brightness(image, max_delta=0.3)
  image = tf.image.random_contrast(image, 0.8, 1.2)
  return image


def input_fn(tf_record_dir, batch_size, mode):
  dataset = tf.data.TFRecordDataset(tf_record_dir)
  aug = False
  if mode == "train":
    dataset = dataset.repeat().shuffle(buffer_size=10000)
    aug = True
  else :
    dataset = dataset.repeat(1)
  dataset = dataset.map(lambda x: _decode_record(x, aug))
  dataset = dataset.batch(batch_size=batch_size)
  return dataset.make_one_shot_iterator().get_next()

if __name__ == "__main__":
  tfrecord_dir = '/data/zhengwu_workspace/ocr/dataset/single_word_gen/tfrecords/train.tfrecord'
  iterator = input_fn(tfrecord_dir, 3, 'train')
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch = sess.run(iterator)
    print(batch['labels'].shape)
    print(batch['images'].shape)

