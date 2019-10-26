#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: wu.zheng midday.me

import tensorflow as tf
import cv2
import  json
import os
import model
import numpy as np
import random
from tensorflow.python.saved_model import tag_constants
import Levenshtein


tf.app.flags.DEFINE_string(
    'image_dir', 'ocr/dataset/text_line_gen', 'Path to the directory containing images.')
tf.app.flags.DEFINE_string(
    'image_list', 'ocr/dataset/text_line_gen/labels.txt', 'Path to the images list txt file.')
tf.app.flags.DEFINE_string(
    'model_dir', 'ocr/models/crnn_gen_model', 'Base directory for the model.')
tf.app.flags.DEFINE_integer(
    'lstm_hidden_layers', 2, 'The number of stacked LSTM cell.')
tf.app.flags.DEFINE_integer(
    'lstm_hidden_uints', 256, 'The number of units in each LSTM cell')
tf.app.flags.DEFINE_string(
    'char_map_json_file', '', 'Path to char map json file')

tf.flags.DEFINE_boolean('export', False, 'if export model')
tf.flags.DEFINE_boolean('eval', False, 'if evaluate model')
tf.app.flags.DEFINE_integer('max_seq_length', 1024 , '')
tf.app.flags.DEFINE_integer('channel_size', 3, 'image channels')

FLAGS = tf.app.flags.FLAGS
_IMAGE_HEIGHT = 32
_MAX_LENGTH = FLAGS.max_seq_length

def _int_to_string(value, char_map_dict):
  word = char_map_dict.get(int(value))
  if word is None :
    word = " "
  elif len(char_map_dict.keys()) == int(value):
    word = ""
  else:
    word = word.strip("\n")
  return word


def _sparse_matrix_to_list(sparse_matrix, char_map_dict):
  indices = sparse_matrix.indices
  values = sparse_matrix.values
  dense_shape = sparse_matrix.dense_shape
  dense_matrix =  len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)

  for i, indice in enumerate(indices):
    dense_matrix[indice[0], indice[1]] = values[i]
  string_list = []
  for row in dense_matrix:
    string = []
    for val in row:
      string.append(_int_to_string(val, char_map_dict))
    string_list.append(''.join(s for s in string if s != '*'))
  return string_list

def standardize(img):
  mean = np.mean(img)
  std = np.std(img)
  img = (img - mean) / std
  return img

class Config():
  num_classes = 0
  lstm_num_units = 256
  channel_size = FLAGS.channel_size

def load_char_map():
  char_map_dict = json.load(open(FLAGS.char_map_json_file))
  return char_map_dict

import time
def merge_text(content):
  words = list(content)
  new_words = []
  last_word = ""
  for w in words:
    if w in [' ']:
      if last_word and (last_word.islower() or last_word.isupper()):
        new_words.append(w)
      else:
        continue
    else:
      new_words.append(w)
  return "".join(new_words)


def eval():
  tf.reset_default_graph()
  char_map_dict = load_char_map()
  config = Config()
  config.num_classes = len(char_map_dict)  + 1
  id_to_char = {v:k for k, v in char_map_dict.items()}
  crnn_net = model.CRNN(config)
  with open(FLAGS.image_list, 'r') as fd:
    image_names = []
    true_labels = []
    for i, line in enumerate(fd):
      seg = " "
      line = line.strip().split(seg)
      image_names.append(line[0])
      true_labels.append(seg.join(line[1:]))

  index_list = random.choices(list(range(len(image_names))), k=50)
  image_names = [image_names[i] for i in index_list]
  labels = [true_labels[i] for i in index_list]

  saver = tf.train.Saver()
  save_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  with tf.Session() as sess:
    saver.restore(sess=sess, save_path=save_path)
    print("restored from %s"%(save_path))
    decoded, log_prob = tf.nn.ctc_greedy_decoder(crnn_net.logits, crnn_net.sequence_length, merge_repeated=True)
    if FLAGS.export:
      tensor_image_input_info = tf.saved_model.utils.build_tensor_info(crnn_net.images)
      tensor_seq_len_input_info = tf.saved_model.utils.build_tensor_info(crnn_net.sequence_length)
      tensor_is_traing_info = tf.saved_model.utils.build_tensor_info(crnn_net.is_training)
      tensor_keep_prob = tf.saved_model.utils.build_tensor_info(crnn_net.keep_prob)
      output_info = tf.saved_model.utils.build_tensor_info(decoded[0])
      signature = tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                'images': tensor_image_input_info, 
                'sequence_length':tensor_seq_len_input_info, 
                "is_training":tensor_is_traing_info, 
                "keep_prob":tensor_keep_prob }, 
              outputs={'decoded': output_info})

      ex_dir = str(int(time.time()))
      builder = tf.saved_model.builder.SavedModelBuilder("./all_exported_models/%s/"%(ex_dir,))
      builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING], signature_def_map={"predict": signature})
      builder.save()
      print("exported model at ")
    ignore = 0
    error_count = 0
    total_count = 0
    for i, image_name in enumerate(image_names):
      image_path = os.path.join(FLAGS.image_dir, image_name)
      if FLAGS.channel_size == 3:
        image = cv2.imread(image_path)
      else:
        image = cv2.imread(image_path , cv2.IMREAD_GRAYSCALE)
      if image is None:
        print('ignore')
        ignore+=1
        continue
      h, w  = image.shape[:2]
      height = _IMAGE_HEIGHT
      width = int(w * height / h)
      image = cv2.resize(image, (width, height))
      image = np.array(image, dtype=np.float32)
      image = image / 255.0 
      seq_len = np.array([width / 4], dtype=np.int32)
      print("length: ",  seq_len)
      if FLAGS.channel_size ==1:
        image = image[:,:,np.newaxis]
        cv2.imwrite("test.png", image*255.0)
      image = np.expand_dims(image, axis=0)
      start = time.time()
      print(image.shape)
      logit, preds, prob = sess.run(
              [crnn_net.logits, decoded, log_prob ],
              feed_dict={
                  crnn_net.images: image,
                  crnn_net.sequence_length:seq_len,
                  crnn_net.keep_prob: 1.0,
                  crnn_net.is_training:False})
      preds = _sparse_matrix_to_list(preds[0], id_to_char)
      cost_time = time.time() - start
      res_text = preds[0]       
      res_text = merge_text(res_text)
      err_count = Levenshtein.distance(labels[i], res_text)
      total_count += len(labels[i])
      error_count += err_count
      print(image_name)
      print('true label {:s} \n predict result: {:s} cost:{:f} \n error_count:{:d}'.format(labels[i], preds[0], cost_time, err_count,) )
    print(1 - 1.0 * error_count / total_count)

if __name__ == "__main__":
  eval()
