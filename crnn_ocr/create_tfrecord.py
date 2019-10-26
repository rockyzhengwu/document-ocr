#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: wu.zheng midday.me

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import json
import tensorflow as tf
import cv2
from multiprocessing import Pool

_IMAGE_HEIGHT = 32

tf.app.flags.DEFINE_string(
    'image_dir', '', 'Dataset root folder with images.')

tf.app.flags.DEFINE_string(
    'image_list', 'ocr/dataset/text_line_gen/labels.txt', 'Path of dataset annotation file.')

tf.app.flags.DEFINE_string(
    'data_dir', 'ocr/dataset/text_line_gen_tfrecords', 'Directory where tfrecords are written to.')

tf.app.flags.DEFINE_float(
    'validation_split_fraction', 0.1, 'Fraction of training data to use for validation.')

tf.app.flags.DEFINE_boolean(
    'shuffle_list', True, 'Whether shuffle data in annotation file list.')

tf.app.flags.DEFINE_string(
    'vocab_file', './simple_vocab.json', 'Path to char map json file') 

tf.app.flags.DEFINE_integer('max_seq_length', 1024, 'max sequence length')
tf.app.flags.DEFINE_integer("channel_size", 1, 'image channle size')


FLAGS = tf.app.flags.FLAGS

_MAGE_MAX_LENGTH = FLAGS.max_seq_length
_MAX_LABEL_LENGTH = 150

if FLAGS.channel_size == 3:
  GRAY = False
else:
  GRAY = True
VOCAB_DICT = json.load(open(FLAGS.vocab_file, 'r'))


def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _string_to_int(label):
  # convert string label to int list by char map
  int_list = []
  for c in list(label):
    if VOCAB_DICT.get(c) is None:
      # todo same unk
      int_list.append(VOCAB_DICT.get("<UNK>"))
    else:
      int_list.append(VOCAB_DICT[c])
  return int_list

def _write_tfrecord(writer_path, anno_lines ):
  writer= tf.io.TFRecordWriter(writer_path)
  for i, line in enumerate(anno_lines):
    line = line.strip('\n')
    line = line.strip()
    image_name = line.split(" ")[0]
    image_path = image_name
    label = " ".join(line.split(" ")[1:])
    if not label:
      print("label is None")
      continue
    if FLAGS.channel_size==1:
      image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE )
    else:
      image = cv2.imread(image_path)
    if image is None: 
      print("image is None")
      continue 
    h, w = image.shape[:2]
    height = _IMAGE_HEIGHT
    width = int(w * height / h)
    try:
      image = cv2.resize(image, (width, height))
    except Exception as e:
      print(e)
      continue
    if width > _MAGE_MAX_LENGTH:
      image = cv2.resize(image, (_MAGE_MAX_LENGTH, height))
    is_success, image_buffer = cv2.imencode('.png', image)
    if not is_success:
      print("encoder image error")
      continue
    image_name = image_name if sys.version_info[0] < 3 else image_name.encode('utf-8') 
    labels_ids = _string_to_int(label) 
    if len(labels_ids) < 1  or len(labels_ids) > _MAX_LABEL_LENGTH - 1:
      print("labels_ids is too long or short")
      continue
    features = tf.train.Features(feature={
       "labels":_int64_feature(labels_ids),
       'images': _bytes_feature(image_buffer.tostring()),
       'imagenames': _bytes_feature(image_name)
    })
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())
  writer.close()



def start_create_process(anno_lines, num_shards, num_thread, dataset_split):
  with Pool(num_thread) as pool:
    total_num = len(anno_lines)
    every_shard_num = int(total_num / num_shards)
    shard_anno_lines= []
    for i in range(num_shards-1):
      shard_anno_lines.append(anno_lines[i* every_shard_num: (i+1) * every_shard_num])
    shard_anno_lines.append(anno_lines[-1*every_shard_num: ])
    writer_list = [os.path.join(FLAGS.data_dir, 'ocr-%s-%d.tfrecord')%(dataset_split, i) for i in range(num_shards)]
    assert len(shard_anno_lines) == len(writer_list)
    args = list(zip(writer_list, shard_anno_lines))
    pool.starmap(_write_tfrecord, args)


def _convert_dataset():
  with open(FLAGS.anno_file, 'r') as anno_fp:
    anno_lines = anno_fp.readlines()
  print(FLAGS.anno_file)
  print(len(anno_lines))
  if FLAGS.shuffle_list:
    random.shuffle(anno_lines)

  if not os.path.exists(FLAGS.data_dir):
    os.mkdir(FLAGS.data_dir)
  split_train_index = int(len(anno_lines) * 0.97)
  split_test_index = int(len(anno_lines) * 0.99)
  train_anno_lines = anno_lines[:split_train_index]
  test_anno_lines = anno_lines[split_train_index: split_test_index]
  validation_anno_lines = anno_lines[split_test_index:]
  start_create_process(train_anno_lines, 100, 10, 'train')
  start_create_process(validation_anno_lines, 10, 10,  'validation')
  start_create_process(test_anno_lines, 10, 10,  'test')



def main(unused_argv):
  _convert_dataset()

if __name__ == '__main__':
  tf.app.run()
