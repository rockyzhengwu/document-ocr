#-*-coding:utf-8-*-

import tensorflow as tf
import glob
import random
import image_augment
import numpy as np

_MAX_LENGTH = 1024

def standardize(img):
  mean = np.mean(img)
  std = np.std(img)
  img = (img - mean) / std
  return img

def _decode_record(record_proto, channel_size):
  feature_map = {
          'images': tf.FixedLenFeature((), tf.string),
          'labels' : tf.VarLenFeature(  tf.int64),
          'imagenames': tf.FixedLenFeature((), tf.string),
          }
  features = tf.parse_single_example(record_proto, features=feature_map)

  images = tf.image.decode_jpeg(features['images'], channels=channel_size)
  images = tf.py_func(image_augment.augment_images, [images], tf.uint8)
  images = tf.cast(images, tf.float32)
  images = images / 255.0 
  image_w = tf.cast(tf.shape(images)[1], tf.int32)
  sequence_length = tf.cast(tf.shape(images)[1] / 4, tf.int32)
  paddings = tf.convert_to_tensor([[0, 0], [0, _MAX_LENGTH - image_w], [0,  0]])
  images = tf.pad(images, paddings)
  images.set_shape([32, _MAX_LENGTH, channel_size])
  labels = tf.cast(features['labels'], tf.int32)
  example = {
          "images":  images,
          "labels" : labels,
          "sequence_length": sequence_length
          }
  return example


def _decode_record_estimator(record_proto, channel_size):
  feature_map = {
      'images': tf.FixedLenFeature((), tf.string),
      'labels' : tf.VarLenFeature(tf.int64),
      'imagenames': tf.FixedLenFeature((), tf.string),
  }
  features = tf.parse_single_example(record_proto, features=feature_map)
  images = tf.image.decode_jpeg(features['images'], channels=channel_size)
  image_w = tf.cast(tf.shape(images)[1], tf.int32)
  paddings = tf.convert_to_tensor([[0, 0], [0, _MAX_LENGTH - image_w], [0,  0]])
  images = tf.pad(images, paddings)
  images.set_shape([32, _MAX_LENGTH, channel_size])
  images = tf.cast(images, tf.float32)
  labels = tf.cast(features['labels'], tf.int32)
  sequence_length = tf.cast(tf.shape(images)[1]/ 4, tf.int32)
  features = {
      "images": images ,
      "sequence_length": sequence_length
  }
  return features, labels

def input_fn(tfrecord_path, batch_size, is_training=True, channel_size=3 ):
  filenames = glob.glob(tfrecord_path)
  print(filenames)
  random.shuffle(filenames)
  dataset = tf.data.TFRecordDataset(filenames)
  if is_training:
    dataset = dataset.repeat().shuffle(buffer_size=10000)
  else:
    data = dataset.repeat(1)
  dataset = dataset.map(lambda x: _decode_record(x, channel_size))
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.prefetch(buffer_size=10000)
  return dataset.make_one_shot_iterator()

