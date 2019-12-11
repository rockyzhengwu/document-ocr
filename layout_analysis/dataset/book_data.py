#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import os
import json
import cv2
import random
import numpy as np
import time
from dataset import image_augmation
from dataset import data_util
import xml.etree.ElementTree as ET

IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
NUM_CLASS=3


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


def data_generator(list_path, image_dir, batch_size, mode='train'):
  label_file_list = load_image_list(list_path)
  print("example size:", len(label_file_list))
  image_batch = []
  label_batch = []
  mask_batch = []
  xml_path_batch = []
  scal_list=[0.3, 0.5, 1.0, 2.0, 3.0]
  while True:
    random.shuffle(label_file_list)
    for xml_path in label_file_list:
      xml_path = os.path.join(image_dir, xml_path)
      label_data=load_label_data(xml_path)
      image_path = os.path.join(image_dir, label_data['image_path'])
      image = cv2.imread(image_path)
      image_labels = np.array(label_data['images']) 
      text_labels = np.array(label_data['texts']) 
      # todo 图像增强
      #aug_image = image_augmation.image_aug(image.copy())
      aug_image = image.copy()
      #rd_scale = np.random.choice(scal_list, 1)
      ##rd_scale = 1.0
      #r_image = cv2.resize(aug_image, dsize=None, fx=rd_scale, fy=rd_scale)
      #image_labels = image_labels * rd_scale
      #text_labels = text_labels * rd_scale
      r_image = aug_image

      if image is None:
        continue
      #h, w = image.shape[:2]
      h = label_data['height']
      w = label_data['width']

      image_h, image_w = r_image.shape[:2]
      h_ratio = image_h / h
      w_ratio = image_w / w
      if image_h > image_w:
        factor =  IMAGE_HEIGHT / image_h
        new_h = IMAGE_HEIGHT
        new_w = int(image_w * factor)
      else:
        factor =  IMAGE_WIDTH / image_w
        new_w = IMAGE_WIDTH
        new_h = int(image_h * factor)
      # todo resize

      w_factor =  new_w / w
      h_factor =  new_h / h
      r_image = cv2.resize(r_image, (new_w, new_h))
      label_image = np.zeros((new_h, new_w))
      mask = np.ones((new_h, new_w))

      label_image = fill_image(label_image, image_labels , 1, w_factor, h_factor)
      label_image = fill_image(label_image, text_labels, 2, w_factor, h_factor)

      train_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
      train_image[0:new_h, 0:new_w] = r_image
      train_label = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
      train_label[0:new_h, 0:new_w] = label_image

      mask = np.ones((new_h, new_w))
      train_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
      train_mask[0:new_h, 0:new_w] = mask


      label_batch.append(train_label)
      train_image = train_image / 255.0
      image_batch.append(train_image)
      mask_batch.append(train_mask)
      xml_path_batch.append(xml_path)
      if len(image_batch) == batch_size:
        yield image_batch, label_batch, mask_batch, xml_path_batch
        image_batch = []
        label_batch = []
        mask_batch = []
        xml_path_batch = []
    if mode!='train':
      break


def get_batch(list_dir, image_dir, batch_size, mode='train', workers=1, max_queue_size=32):
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


import random

def get_random_color():
  color = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
  return color

def mask_to_bbox(mask, im, num_class, out_path=None, out_file_name=None):
  bbox_list = []
  mask = mask.astype(np.uint8)
  for i in range(1, num_class, 1):
    c_bbox_list = []
    c_mask = np.zeros_like(mask)
    c_mask[np.where(mask==i)] = 255
    contours, hierarchy = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = get_random_color()
    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area < 50:
        continue
      epsilon = 0.005 * cv2.arcLength(cnt,True)
      approx = cv2.approxPolyDP(cnt,epsilon,True)
      (x, y, w, h) = cv2.boundingRect(approx)
      c_bbox_list.append([x,  y, x+w, y+h])
      if out_path is not None:
        cv2.putText(im, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        im=cv2.rectangle(im, pt1=(x, y), pt2=(x+w, y+h), color=color, thickness=1)
    bbox_list.append(c_bbox_list)
  if out_path is not None:
    outf = os.path.join(out_path, out_file_name)
    print(outf)
    cv2.imwrite(outf, im)
  return bbox_list


