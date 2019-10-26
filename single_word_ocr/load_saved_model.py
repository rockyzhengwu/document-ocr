#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import tensorflow as tf
import cv2
import numpy as np
import json
import time
import os

EXPORT_PATH = "single_word_model/densenet_exported/1565353481"
VOCAB_PATH = './gbk.json'


def load_charset_map():
  char_set = json.load(open(VOCAB_PATH))
  char_set = {v:k for k, v in char_set.items()}
  return char_set

def pre_process_image(im):
  image_size = 64
  h, w = im.shape[:2]
  if h > image_size  or w > image_size:
    im = cv2.resize(im, (image_size, image_size))
  else:
    pad_height = int((image_size - h)  /  2)
    pad_width = int((image_size - w) / 2)
    im = np.pad(im, ((pad_height, image_size-h - pad_height), (pad_width, image_size - w - pad_width)),
                mode='constant', constant_values=((255, 255),(255, 255)))
  print(im.shape)
  im = im / 255.0 
  im = im.reshape([1, image_size, image_size, 1])
  return im

class Model():
  def __init__(self):
    self.sess = tf.Session()
    tf.saved_model.loader.load(self.sess, ['serve'], EXPORT_PATH)
    graph = tf.get_default_graph()
    self.input_image = graph.get_tensor_by_name('image:0')
    self.predict_prob = graph.get_tensor_by_name('prob:0')
    self.predict_ids = graph.get_tensor_by_name('prediction:0')
    self.char_dict = load_charset_map()

  def predict(self, im):
    im = pre_process_image(im)
    feed_dict = {self.input_image: im }
    predict_prob, predict_ids = self.sess.run([self.predict_prob, self.predict_ids], feed_dict=feed_dict)
    out_index = int(predict_ids)
    print(self.predict_prob, self.char_dict[out_index])
    return self.char_dict[int(out_index)]



def predict(image_path, true_label="", counter=0):
  im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE )
  print(im.shape)
  word_im_list = [im]
  for i, word_im in enumerate(word_im_list):
    cv2.imwrite(os.path.join('word_image', str(counter)+".png"), word_im)
    start = time.time()
    word = model.predict(word_im)
    cost = time.time() - start
    print('image_path: %s true_label: %s predict_label: %s cost : %f'%(image_path, true_label, word, cost))
    return true_label == word

def load_image_path(file_path):
  image_path_list = []
  label_path_list = []
  with open(file_path) as f:
    for _, line in enumerate(f):
      line = line.strip("\n")
      line = line.split()
      if len(line) < 2:
        continue
      image_path = line[0]
      label = line[1]
      label_path_list.append(label)
      image_path_list.append(image_path)
    return image_path_list, label_path_list


model = Model()

if __name__ == "__main__":
  file_path = '/home/zhengwu/data/pdfs/books_line/d83a286f-08c9-5cb4-8dc0-65162a99781e/labels.txt'
  image_path_list, label_path_list = load_image_path(file_path)
  counter = 0
  right_counter = 0
  for image_path, label in zip(image_path_list, label_path_list):
    print(image_path)
    print(label)
    res = predict(image_path, label, counter)
    if res:
      right_counter +=1
    counter += 1
    if counter > 1000:
      break
  print(right_counter*1.0 / counter)



