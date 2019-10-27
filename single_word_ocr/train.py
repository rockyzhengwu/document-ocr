#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me


import os
import time
import numpy as np
import tensorflow as tf
import densenet
import data_generator
import argparse

parser = argparse.ArgumentParser(description="SingleWord Reconition")
parser.add_argument("--train_image_list", help='train data label dir')
parser.add_argument('--test_image_list', help='test_data_dir')
parser.add_argument('--checkpoint_path', help='checkpoint path')
parser.add_argument('--batch_size', default=128, help='batch size', type=int)
parser.add_argument('--num_class', help='num of class', type=int)

def train():
  batch_size = args.batch_size
  num_class = args.num_class
  model = densenet.DenseNet(batch_size=batch_size, num_classes=num_class)
  global_step = tf.train.get_or_create_global_step()
  start_learning_rate= 0.0001
  learning_rate = tf.train.exponential_decay(
    start_learning_rate,
    global_step,
    100000,
    0.98,
    staircase=False,
    name="learning_rate"
  )
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  train_op= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=model.loss, global_step=global_step)
  train_op = tf.group([train_op, update_ops])
  #optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss=model.loss)
  saver = tf.train.Saver()
  tf.summary.scalar(name='loss', tensor=model.loss)
  #tf.summary.scalar(name='softmax_loss', tensor=model.softmax_loss)
  #tf.summary.scalar(name='center_loss', tensor=model.center_loss)
  tf.summary.scalar(name='accuracy', tensor=model.accuracy)
  merge_summary_op = tf.summary.merge_all()
  sess_config = tf.ConfigProto(allow_soft_placement=True,)
  with tf.Session(config=sess_config) as sess:
    ckpt = tf.train.latest_checkpoint(args.checkpoint_path)
    if ckpt:
      print("restore form %s "%(ckpt))
      st = int(ckpt.split('-')[-1])
      saver.restore(sess, ckpt)
      sess.run(global_step.assign(st))
    else:
      tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter(args.checkpoint_path)
    summary_writer.add_graph(sess.graph)
    start_time = time.time()
    step = 0
    iterator = data_generator.get_batch(args.train_image_list, batch_size)
    for batch in iterator:
      if batch is None:
        print("batch is None")
        continue
      image = batch[0]
      labels = batch[1]
      feed_dict = {model.images: image, model.labels: labels}
      _, loss, accuracy,summary, g_step, logits, lr = sess.run(
              [train_op, model.loss, model.accuracy, merge_summary_op, global_step, model.logits, learning_rate ], 
              feed_dict=feed_dict)
      if loss is None:
        print(np.max(logits), np.min(logits))
        exit(0)
      if step % 10 ==0:
        print(np.max(logits), np.min(logits))
        print("step:%d, lr: %f, loss: %f, accuracy: %f"%(g_step, lr, loss, accuracy))
      if step % 100 == 0:
        summary_writer.add_summary(summary=summary, global_step=g_step)
        saver.save(sess=sess, save_path=os.path.join(args.checkpont_path, 'model'), global_step=g_step)
      step += 1
    print("cost: ", time.time() - start_time)

if __name__ == "__main__":
  args = parser.parse_args()
  print(args)
  train()

