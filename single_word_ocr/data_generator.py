#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import cv2
import random
import image_augment
import numpy as np

import time
import data_util

def load_image_list(image_data_file_path):
  image_path_list = []
  label_list = []
  with open(image_data_file_path) as f:
    for _, line in enumerate(f):
      line = line.strip("\n").split()
      if len(line)<2:
        continue
      image_path = " ".join(line[0:-1])
      label = int(line[-1])
      image_path_list.append(image_path)
      label_list.append(label)
  return image_path_list, label_list

def pre_process_image(image):
  image_size = 64
  h, w = image.shape[:2]
  if h == image_size and w == image_size:
    pass
  elif h > image_size  or w > image_size:
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
  else:
    pad_height = int((image_size - h)  /  2)
    pad_width = int((image_size - w) / 2)
    image = image / 255.0
    image = np.pad(image, ((pad_height, image_size-h - pad_height), (pad_width, image_size-w-pad_width),(0,0)), mode='constant' )
  return image


def data_generator(image_data_file_path, batch_size, mode='train'):
  image_path_list, label_list = load_image_list(image_data_file_path)
  print(len(image_path_list), len(label_list))
  index_list = list(range(len(image_path_list)))
  while True:
    #if mode=='train':
    random.shuffle(index_list)
    image_batch = []
    label_batch = []
    image_path_batch = []
    for idx in index_list:
      image_path = image_path_list[idx]
      label = label_list[idx]
      #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
      image = cv2.imread(image_path)
      # todo crop image
      if image is None:
        print('image is none: %s'%(image_path))
        continue
      image_batch.append(pre_process_image(image))
      label_batch.append(label)
      image_path_batch.append(image_path)
      if len(image_batch) == batch_size:
        yield image_batch, label_batch, image_path_batch
        image_batch = []
        label_batch = []
        image_path_batch = []
      if mode=='train':
        for i in range(1):
          # image_aug = image_augment.augment_image(image)
          image_aug = image
          image_batch.append(pre_process_image(image_aug))
          label_batch.append(label)
          if len(image_batch) == batch_size:
            yield image_batch, label_batch, image_path_batch
            image_batch = []
            label_batch = []
            image_path_batch= []
    if mode!='train':
        break

def get_batch(data_dir, batch_size, mode='train', workers=1, max_queue_size=32):
  enqueuer = data_util.GeneratorEnqueuer(data_generator(data_dir, batch_size, mode))
  enqueuer.start(max_queue_size=max_queue_size, workers=workers)
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
