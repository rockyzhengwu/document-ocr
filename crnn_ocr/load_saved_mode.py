#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import tensorflow as tf
import cv2
import numpy as np
import json

EXPORT_PATH = "./exported_model"
_IMAGE_HEIGHT = 32
_MAX_LENGTH = 1024
char_map_path = './vocab.json'
CHAR_TO_ID = json.load(open(char_map_path))
ID_TO_CHAR = {v:k for k,v in CHAR_TO_ID.items()}
CHAR_SIZE = len(CHAR_TO_ID)

def pre_process_image(image):
  h, w,_  = image.shape
  height = _IMAGE_HEIGHT
  width = int(w * height / h)
  image = cv2.resize(image, (width, height))
  image = np.array(image, dtype=np.float32)
  if width > _MAX_LENGTH:
    image = cv2.resize(image, (_MAX_LENGTH, height))
    width = _MAX_LENGTH
  image = image / 255.0
  seq_len = np.array([width / 4], dtype=np.int32)
  image = np.expand_dims(image, axis=0)
  return image, seq_len

def load_image(image_path):
  image = cv2.imread(image_path)
  return pre_process_image(image)


def _int_to_string(value):
  return ID_TO_CHAR.get(int(value), "~")

def _sparse_matrix_to_list(sparse_matrix ):
  indices = sparse_matrix.indices
  values = sparse_matrix.values
  dense_shape = sparse_matrix.dense_shape
  dense_matrix =  CHAR_SIZE * np.ones(dense_shape, dtype=np.int32)

  for i, indice in enumerate(indices):
    dense_matrix[indice[0], indice[1]] = values[i]
  string_list = []
  for row in dense_matrix:
    string = []
    for val in row:
      string.append(_int_to_string(val ))
    string_list.append(''.join(s for s in string if s != '*'))
  return string_list

class Model():
  def __init__(self,):
    self.sess = tf.Session()
    tf.saved_model.loader.load(
        self.sess, 
        [tf.saved_model.tag_constants.SERVING], EXPORT_PATH)

    graph = tf.get_default_graph()
    self.image = graph.get_tensor_by_name('images:0')
    self.sequence_length = graph.get_tensor_by_name("sequence_length:0")
    self.is_trainig = graph.get_tensor_by_name("training:0")
    self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
    self.logits = graph.get_tensor_by_name("lstm_layers/logits:0")
    self.decoded ,_ = tf.nn.ctc_beam_search_decoder(
        self.logits, 
        self.sequence_length, 
        merge_repeated=True, 
        beam_width=10, 
        top_paths=1)

  def predict(self, im):
    image, seq_len = pre_process_image(im)
    feed_dict = {
        self.image: image, 
        self.sequence_length:seq_len, 
        self.keep_prob:1.0, 
        self.is_trainig:True}

    decoded = self.sess.run(self.decoded, feed_dict = feed_dict)
    pred = _sparse_matrix_to_list(decoded[0])
    return pred

if __name__ == "__main__":
  image_path = './497af4b4-08c0-40cd-b46e-b1576d13e689_6.jpg'
  im = cv2.imread(image_path)
  model = Model()
  model.predict(im)

