#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import os
import tensorflow as tf
import random
import cv2
import json

CHAR_MAP_DICT = json.load(open("vocab.json"))

def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _string_to_int(label):
  int_list = []
  for c in list(label):
    if CHAR_MAP_DICT.get(c) is None:
      print("inot in :",c)
      continue
    int_list.append(CHAR_MAP_DICT[c])
  return int_list

def create_tf_record(data_dir, tfrecords_path):
  image_names = []
  for root, dirs, files in os.walk(data_dir):
    image_names +=[os.path.join(root, name) for name in files]
  random.shuffle(image_names)
  writer = tf.python_io.TFRecordWriter(tfrecords_path)
  print("handle image : %d"%(len(image_names)))
  i = 0
  for image_name in image_names:
    if i % 10000 == 0:
      print(i, len(image_names))
    i+=1
    im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    try:
      is_success, image_buffer = cv2.imencode('.png', im)
    except Exception as e:
      continue
    if not is_success:
      continue
    label = int(image_name.split("/")[-2])
    features = tf.train.Features(feature={
         'labels': _int64_feature(label),
          'images': _bytes_feature(image_buffer.tostring()),
          'imagenames': _bytes_feature(image_name.encode("utf-8"))})
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())
  writer.close()

