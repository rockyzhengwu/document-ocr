#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import os
import json
import cv2
import random
import numpy as np
import time
from dataset import image_augment
from dataset import data_util
import xml.etree.ElementTree as ET

IMAGE_WIDTH=768
IMAGE_HEIGHT=1024
NUM_CLASS=3

LABEL_NAMES = ["text_line", 'table', 'image']


def load_image_list(image_data_path):
  image_path_list = []
  with open(image_data_path) as f:
    for _, line in enumerate(f):
      line = line.strip("\n")
      image_path_list.append(line)
  return image_path_list

def get_int_value(item, name):
  return int(item.get(name))

def get_item_box(item):
  x = get_int_value(item, 'left')
  y = get_int_value(item, 'top')
  w = get_int_value(item, 'width')
  h = get_int_value(item, 'height')
  return x, y, w, h

def load_label_data(label_file_path):
  tree = ET.parse(label_file_path)  
  root = tree.getroot()
  label_data = {}
  image_list= root.findall('image')
  text_list= root.findall('text')
  label_data['image_path']=root.get("image_path")
  page_height = int(root.get('height'))
  page_width = int(root.get('width'))
  label_data['width'] =  page_width
  label_data['height'] = page_height
  label_data['images'] = []
  label_data['texts'] = []
  for image in image_list:
    box = get_item_box(image)
    label_data['images'].append(box)
  for text in text_list:
    box = get_item_box(text)
    label_data['texts'].append(box)
  return label_data

def get_shape_by_type(shapes, label):
  shape_list = []
  for shape in shapes:
    if shape['label'] == label:
      shape_list.append(shape)
  return shape_list

def fill_image(label_image, boxs, label_value, w_factor, h_factor):
  for box in boxs:
    x, y, w, h = box
    min_x, min_y = x, y
    max_x, max_y = x + w , y + h
    area = (max_x - min_x) * (max_y - min_y)
    point_box = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    point_box = np.array(point_box)
    point_box = point_box.reshape((4,2))
    point_box[:,0] = point_box[:,0] * w_factor
    point_box[:,1] = point_box[:,1] * h_factor
    label_image = cv2.fillPoly(label_image, point_box.astype(np.int32)[np.newaxis, :,: ], label_value)
  return label_image


def data_generator(list_path, image_dir batch_size, mode='train'):
  label_file_list = load_image_list(list_path)
  print("example size:", len(label_file_list))
  image_batch = []
  label_batch = []
  while True:
    random.shuffle(label_file_list)
    for xml_path in label_file_list:
      #if 'book' not in image_path:
      #    continue
      label_data=load_label_data(xml_path)
      image_path = os.path.join(image_dir, label_data['image_path'])
      image = cv2.imread(image_path)
      if image is None:
        continue
      #h, w = image.shape[:2]
      h = label_data['height']
      w = label_data['width']

      h_factor =  IMAGE_HEIGHT / h
      w_factor =  IMAGE_WIDTH / w
      image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

      label_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1))
      images= label_data['images']
      label_image = fill_image(label_image, images , 1, w_factor, h_factor)

      texts = label_data['texts']

      label_image = fill_image(label_image, texts, 2, w_factor, h_factor)
      if mode == 'train':
        image, _ = image_augment.augment_with_segmap(image, label_image, NUM_CLASS)
      if len(label_image.shape) == 3:
        label_image = label_image.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
      # todo image augmentation, 图像增强
      label_batch.append(label_image)
      image = image / 255.0
      image_batch.append(image)
      if len(image_batch) == batch_size:
        yield image_batch, label_batch
        image_batch = []
        label_batch = []
    if mode!='train':
      break


def get_batch(list_dir, image_dir, batch_size, mode='train', workers=1, max_queue_size=256):
  try:
    enqueuer = data_util.GeneratorEnqueuer(data_generator(list_dir, image_dir, batch_size, mode))
    enqueuer.start(max_queue_size=max_queue_size, workers=workers)
    enqueuer.is_running()
    generator_output = None
    while True:
      while enqueuer.is_running():
        if not enqueuer.queue.empty():
          generator_output = enqueuer.queue.get()
          break
        else:
          time.sleep(0.01)
      yield generator_output
      generator_output = None
  except Exception as e:
    print('load data error: ', e)
  finally:
    if enqueuer is not None:
        enqueuer.stop()


