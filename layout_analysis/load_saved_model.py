#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import tensorflow as tf
import os
import copy
import numpy as np
import cv2


MODEL_PATH='/data/zhengwu_workspace/ocr/models/book_export_models/image_txt/book/1571968585'
IMAGE_HEIGHT=1024
IMAGE_WIDTH=768
NUM_CLASS=3
COLOR_LIST=[(0, 255,0), (0, 0, 255), (0, 255, 255)]

def mask_to_bbox(mask, image, num_class, area_threhold=0, out_path=None, out_file_name=None):
  bbox_list = []
  im = copy.copy(image)
  mask = mask.astype(np.uint8)
  for i in range(1, num_class, 1):
    c_bbox_list = []
    c_mask = np.zeros_like(mask)
    c_mask[np.where(mask==i)] = 255
    bimg , countours, hier = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in countours:
      area = cv2.contourArea(cnt)
      if area < area_threhold:
        continue
      epsilon = 0.005 * cv2.arcLength(cnt,True)
      approx = cv2.approxPolyDP(cnt,epsilon,True)
      (x, y, w, h) = cv2.boundingRect(approx)
      c_bbox_list.append([x,  y, x+w, y+h])
      if out_path is not None:
        color = COLOR_LIST[i-1]
        im=cv2.rectangle(im, pt1=(x, y), pt2=(x+w, y+h),color=color, thickness=2)
    bbox_list.append(c_bbox_list)
  if out_path is not None:
    outf = os.path.join(out_path, out_file_name)
    cv2.imwrite(outf, im)
  return bbox_list

def resize_bbox(bbox_list, w_factor, h_factor, class_names):
  bbox_map = {}
  for c,  c_bbox_list in enumerate(bbox_list):
    c_name = class_names[c]
    bbox_map[c_name] = []
    for bbox in c_bbox_list:
      new_bbox = [bbox[0]/ w_factor, bbox[1]/ h_factor, bbox[2]/w_factor, bbox[3]/h_factor]
      new_bbox = list(map(int, new_bbox))
      bbox_map[c_name].append(new_bbox)
  return bbox_map


class Model(object):
  def __init__(self, model_dir, area_threhold, class_names):
    self.model_dir = model_dir
    self.area_threhold = area_threhold
    self.num_class = len(class_names) + 1
    self.class_names = class_names 
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.model_dir)
    self.image = self.graph.get_tensor_by_name('image:0')
    self.prob  = self.graph.get_tensor_by_name("prob:0")


  def predict(self, img):
    h, w = img.shape[:2]
    h_factor = IMAGE_HEIGHT / h
    w_factor = IMAGE_WIDTH / w
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    feed_dict = {self.image:[img/255.0]}
    prob = self.sess.run([self.prob], feed_dict=feed_dict)
    prob = prob[0][0]
    mask = np.argmax(prob, axis=-1)

    mask = mask.astype(np.uint8)
    bbox_list = mask_to_bbox(mask, img, self.num_class, self.area_threhold,  "./",  "server_predict.png")
    bbox_map = resize_bbox(bbox_list, w_factor, h_factor, self.class_names)
    return bbox_map


if __name__ =="__main__":
    image_path=""
    img = cv2.imread(image_path)
    model = Model(MODEL_PATH, 10, ['image', 'text'])
    model.predict(img)
