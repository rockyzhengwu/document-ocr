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
parser.add_argument('--name',  help='an integer for the accumulator', )
parser.add_argument('--max_step', default=500000, help='max step for train')


def train():
  NUM_CLASS = data_factory.get_data_config(args.name)['num_class']
  CHECK_POINTS_PATH = data_factory.get_data_config(args.name)['check_points_path']
  train_model = model.UnetModel(NUM_CLASS, is_training=True )
  global_step = tf.train.get_or_create_global_step()
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  saver = tf.train.Saver()
  with tf.control_dependencies(update_ops):
    loss_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(train_model.loss, global_step=global_step)
    train_op = tf.group([loss_op, update_ops])
  tf.summary.scalar(name='seg_loss', tensor=train_model.seg_loss)
  tf.summary.scalar(name='aux_loss', tensor=train_model.aux_loss)
  tf.summary.scalar(name="loss", tensor=train_model.loss)
  tf.summary.scalar(name="acc", tensor=train_model.acc)
  merge_summary_op = tf.summary.merge_all()
  with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(CHECK_POINTS_PATH)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    if ckpt:
      st = int(ckpt.split("-")[-1])
      saver.restore(sess, ckpt)
      sess.run(global_step.assign(st))
      print("restore from %s"%(ckpt))
    summary_writer = tf.summary.FileWriter(CHECK_POINTS_PATH)
    summary_writer.add_graph(sess.graph)
    for batch_data in data_factory.get_data_iterator(args.name):
      if batch_data is None:
        print("Warning: batch_data is None")
        continue
      labels = np.array(batch_data[1])
      labels = np.expand_dims(labels, -1)
      images = np.array(batch_data[0])
      _, loss, _, acc, s, summary = sess.run([train_op, train_model.loss, train_model.acc_op, train_model.acc, global_step, merge_summary_op], 
          feed_dict={train_model.image: images, train_model.label: labels})
      print("step:%d loss : %f acc: %f"%(s, loss , acc ))
      if s % 100 == 0:
        summary_writer.add_summary(summary=summary, global_step=s)
        saver.save(sess=sess, save_path=os.path.join(CHECK_POINTS_PATH, 'model'), global_step=s)
      if s > args.max_step:
        print("train finish")
        break
        
if __name__ == '__main__':
  args = parser.parse_args()
  train()

