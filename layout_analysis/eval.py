#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import os
import model
import tensorflow as tf
import numpy as np
import cv2
import copy
import time
from tensorflow.python.saved_model import tag_constants
from dataset import data_factory
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  help='an integer for the accumulator', )
parser.add_argument('--export_dir', default='./export_models', help='export model dir')
parser.add_argument('--export', dest='export',type=bool, help='if export saved model')
parser.add_argument('--data_dir', help='valid data dir')
parser.set_defaults(export=False)

COLOR_LIST=[(0, 255, 0),(0, 0, 255), (0, 255, 255)]

def mask_to_bbox(mask, image, num_class, out_path=None, out_file_name=None):
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
      if area < 50:
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
    print(outf)
    cv2.imwrite(outf, im)
  return bbox_list


def metrics(true_mask, predict_mask):
  pixel_accuracy = np.sum(predict_mask==true_mask) / true_mask.size
  mean_pixel_accuracy = mean_accuraccy(true_mask, predict_mask)
  print("pixel_accuracy:", pixel_accuracy)
  print("mean_pixel_accuracy:", mean_pixel_accuracy)
  return pixel_accuracy, mean_pixel_accuracy

def mean_accuraccy(true_mask, predict_mask):
  """
  computes mean accuraccy: 1/n_cl * sum_i(n_ii/t_i)
  """
  s, cl = per_class_accuraccy(true_mask, predict_mask)
  return np.sum(s) / cl.size

def per_class_accuraccy(true_mask, predict_mask):
  """
  computes pixel by pixel accuraccy per class in target
  sum_i(n_ii/t_i)
  """
  cl = np.unique(predict_mask)
  n_cl = cl.size
  s = np.zeros(n_cl)
  for i, c in enumerate(cl):
    s[i] = (predict_mask[predict_mask== true_mask] == c).sum() / (predict_mask== c).sum()
  return (s, cl)


def compute_iou(groundtruth_box, detection_box):
  g_ymin, g_xmin, g_ymax, g_xmax = groundtruth_box
  d_ymin, d_xmin, d_ymax, d_xmax = detection_box

  xa = max(g_xmin, d_xmin)
  ya = max(g_ymin, d_ymin)
  xb = min(g_xmax, d_xmax)
  yb = min(g_ymax, d_ymax)

  intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

  boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
  boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

  return intersection / float(boxAArea + boxBArea - intersection)

def bbox_accuracy(true_bbox_list, predict_bbox_list):
  total_true = len(true_bbox_list[0])
  total_predict = len(predict_bbox_list[0])
  tp = 0
  for true_bbox in true_bbox_list[0]:
    for predict_bbox in predict_bbox_list[0]:
      iou = compute_iou(true_bbox, predict_bbox)
      if iou >= 0.9:
        tp +=1
        break
  precision = 1.0 * tp / total_predict
  recal =  1.0 * tp / total_true
  f1 = 2 * precision * recal / (precision + recal)
  print(total_true, total_predict, tp)
  return total_true, total_predict, tp


def predict():
  data_config = data_factory.get_data_config(args.name)
  check_points_path = tf.train.latest_checkpoint(data_config.get("check_points_path"))
  num_class = data_config.get("num_class")
  test_model = model.UnetModel(num_class, False)
  saver = tf.train.Saver()
  total_true = 0
  total_predict = 0
  total_tp = 0
  counter = 0
  with tf.Session() as sess:
    saver.restore(sess=sess, save_path=check_points_path)
    if args.export:
      tensor_image_input_info = tf.saved_model.utils.build_tensor_info(test_model.image)
      tensor_prob_output_info = tf.saved_model.utils.build_tensor_info(test_model.prob)
      signature=tf.saved_model.signature_def_utils.build_signature_def(
              inputs={"images":tensor_image_input_info, },
              outputs={"prob":tensor_prob_output_info})
      ex_dir = str(int(time.time()))
      export_dir = os.path.join(args.export_dir, args.name, ex_dir)
      builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
      builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING], signature_def_map={"predict":signature})
      builder.save()
      print("model exporte at %s"%(export_dir))
      exit(0)
    data_iterator = data_factory.get_data_iterator(args.name, mode='test', batch_size=1)
    for batch_data in  data_iterator:
      if batch_data is None:
        continue
      images = np.array(batch_data[0])
      labels = batch_data[1]
      labels = np.array(labels[0])
      logits = sess.run([test_model.logits], feed_dict={test_model.image:images})
      mask = logits[0]
      mask = np.argmax(mask[0], axis=-1)
      mask = mask.astype(np.uint8)
      im = images[0] * 255.0
      labels = labels.astype(np.uint8)
      test_out_path = './test_out'
      if not os.path.exists(test_out_path):
        os.makedirs(test_out_path)
      true_bbox_list = mask_to_bbox(labels, im, num_class,  out_path=test_out_path, out_file_name=str(counter)+"_true.png")
      predict_bbox_list = mask_to_bbox(mask, im, num_class, out_path=test_out_path, out_file_name=str(counter) + "_predict.png")
      metrics(labels, mask)
      counter += 1
      if counter > 20:
        break

if __name__ == "__main__":
  args = parser.parse_args()
  print(args)
  predict()

