#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: wu.zheng midday.me

import os
import tensorflow as tf
import numpy as np
from dataset import data_factory
import model
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
# trian
parser.add_argument('--max_step', default=500000, help='max step for train')
parser.add_argument('--learning_rate', default=0.00001, help='init learning rate')
parser.add_argument('--checkpoints_dir', default='./checkpoints', help='checkpoints dir')
parser.add_argument('--batch_size', default=4, help='batch size', type=int)

# data
parser.add_argument('--train_label_list', help='train label list ')
parser.add_argument('--image_dir', help='image dir')
parser.add_argument('--data_name', help='data generator name')
parser.add_argument('--num_class', default=3, help='num of classes', type=int)
parser.add_argument('--image_width', default=768, help='image width', type=int)
parser.add_argument('--image_height', default=768, help='image height', type=int)



def train():
  train_model = model.UnetModel(args.num_class, is_training=True , dice_loss=True)
  global_step = tf.train.get_or_create_global_step()
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  saver = tf.train.Saver()
  with tf.control_dependencies(update_ops):
    loss_op = tf.train.AdamOptimizer(args.learning_rate).minimize(train_model.loss, global_step=global_step)
    train_op = tf.group([loss_op, update_ops])
  #tf.summary.scalar(name='seg_loss', tensor=train_model.loss)
  #tf.summary.scalar(name='aux_loss', tensor=train_model.aux_loss)
  pred = tf.argmax(train_model.logits, axis=-1, output_type=tf.int32)
  pred = tf.cast(pred, dtype=tf.float32)
  pred = tf.expand_dims(pred, axis=-1)
  label = tf.cast(train_model.label, dtype=tf.float32)
  tf.summary.image('image', train_model.image * 255.0)
  tf.summary.image('label', label* 50.0)
  tf.summary.image('mask', train_model.mask* 50.0)
  tf.summary.image('pred', pred * 50.0)
  tf.summary.scalar(name="loss", tensor=train_model.loss)
  tf.summary.scalar(name="acc", tensor=train_model.acc)
  merge_summary_op = tf.summary.merge_all()
  data_generator_fn = data_factory.get_data(args.data_name)
  with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(args.checkpoints_dir)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    if ckpt:
      st = int(ckpt.split("-")[-1])
      saver.restore(sess, ckpt)
      sess.run(global_step.assign(st))
      print("restore from %s"%(ckpt))
    summary_writer = tf.summary.FileWriter(args.checkpoints_dir)
    summary_writer.add_graph(sess.graph)
    data_iterator = data_generator_fn.get_batch(args.train_label_list, args.image_dir, args.batch_size, 'train')
    for batch_data in data_iterator:
      if batch_data is None:
        print("Warning: batch_data is None")
        continue
      images = np.array(batch_data[0])
      labels = np.array(batch_data[1])
      labels = np.expand_dims(labels, -1)
      mask = np.array(batch_data[2])
      mask = np.expand_dims(mask, -1)
      _, loss, _, acc, s, summary = sess.run([train_op, train_model.loss, train_model.acc_op, train_model.acc, global_step, merge_summary_op], 
          feed_dict={train_model.image: images, train_model.label: labels, train_model.mask: mask})
      print("step:%d loss : %f acc: %f"%(s, loss , acc ))
      if s % 100 == 0:
        summary_writer.add_summary(summary=summary, global_step=s)
        saver.save(sess=sess, save_path=os.path.join(args.checkpoints_dir, 'model'), global_step=s)
      if s > args.max_step:
        print("train finish")
        break
        
if __name__ == '__main__':
  args = parser.parse_args()
  train()

