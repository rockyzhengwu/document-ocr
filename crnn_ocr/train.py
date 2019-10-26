#!/usr/bin/env python                                                                                                                                       
# -*- coding:utf-8 -*-
#author: wu.zheng midday.me

import tensorflow as tf
import model
import os
import json
import time
import numpy as np
import input_fn


tf.app.flags.DEFINE_string(
    'data_dir', '', 'Path to the directory containing data tf record.')

tf.app.flags.DEFINE_string(
    'model_dir', '/data/zhengwu_workspace/ocr/models/crnn_gen_model', 'Base directory for the model.')

tf.app.flags.DEFINE_integer(
    'num_threads', 4, 'The number of threads to use in batch shuffling')

tf.app.flags.DEFINE_integer(
    'step_per_eval', 5000, 'The number of training steps to run between evaluations.')

tf.app.flags.DEFINE_integer(
    'step_per_save', 100, 'The number of training steps to run between save checkpoints.')

tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_train_steps', 500000, 'The number of maximum iteration steps for training')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.0001, 'The initial learning rate for training.')

tf.app.flags.DEFINE_integer(
    'decay_steps', 30000, 'The learning rate decay steps for training.')

tf.app.flags.DEFINE_float(
    'decay_rate', 0.98, 'The learning rate decay rate for training.')

tf.app.flags.DEFINE_string(
    'char_map_json_file', './simple_vocab.json', 'Path to char map json file')

tf.app.flags.DEFINE_integer('max_seq_length', 1024 , '')
tf.app.flags.DEFINE_integer('channel_size', 3, 'image channels')

FLAGS = tf.app.flags.FLAGS

def _int_to_string(value, char_map_dict=None):
  if char_map_dict is None:
    char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))

  assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')

  for key in char_map_dict.keys():
    if char_map_dict[key] == int(value):
      return str(key)
    elif len(char_map_dict.keys()) == int(value):
      return ""
  raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))

def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
  indices = sparse_matrix.indices
  values = sparse_matrix.values
  dense_shape = sparse_matrix.dense_shape
  # the last index in sparse_matrix is ctc blanck note
  if char_map_dict is None:
    char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
  assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')
  dense_matrix =  len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)

  for i, indice in enumerate(indices):
    dense_matrix[indice[0], indice[1]] = values[i]
  string_list = []
  for row in dense_matrix:
    string = []
    for val in row:
      string.append(_int_to_string(val, char_map_dict))
    string_list.append(''.join(s for s in string if s != '*'))
  return string_list


class Config():
  num_classes = 123
  lstm_num_units = 256 
  batch_size = 1
  is_training= True
  channel_size = FLAGS.channel_size

def load_char_map():
  char_map_dict = json.load(open(FLAGS.char_map_json_file))
  return char_map_dict

def sparse_labels(labels):
  values = []
  indices = []
  max_len = max(map(len, labels))
  batch_size = len(labels)
  for i, item in enumerate(labels):
    for j, value in enumerate(item):
      ind = (i,j)
      indices.append(ind)
      values.append(value)
  sparse_labels = tf.SparseTensor(
      indices=indices, 
      values=values, 
      dense_shape=(batch_size, max_len))
  return sparse_labels


def main():
  train_tf_record = os.path.join(FLAGS.data_dir, 'ocr-train-*.tfrecord')
  eval_tf_record = os.path.join(FLAGS.data_dir, 'ocr-validation-*.tfrecord')

  char_map_dict = load_char_map()
  train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
  model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
  model_save_path = os.path.join(FLAGS.model_dir, model_name)

  config = Config()
  config.batch_size = FLAGS.batch_size
  config.num_classes = len(char_map_dict) + 1
  train_input_fn = input_fn.input_fn(train_tf_record, FLAGS.batch_size, channel_size=FLAGS.channel_size)

  crnn_model = model.CRNN(config)
  saver = tf.train.Saver()
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                             global_step,
                                             FLAGS.decay_steps,
                                             FLAGS.decay_rate,
                                             staircase = True)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op= tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate).minimize(crnn_model.loss, 
            global_step=global_step)
    train_op = tf.group([train_op, update_ops])
  decoded, log_prob = tf.nn.ctc_greedy_decoder(crnn_model.logits, crnn_model.sequence_length)
  pred_str_labels = tf.as_string(decoded[0].values)
  pred_tensor = tf.SparseTensor(indices=decoded[0].indices, values=pred_str_labels, dense_shape=decoded[0].dense_shape)
  true_str_labels = tf.as_string(crnn_model.labels.values)
  true_tensor = tf.SparseTensor(indices=crnn_model.labels.indices, values=true_str_labels, dense_shape=crnn_model.labels.dense_shape)
  edit_distance = tf.reduce_mean(tf.edit_distance(pred_tensor, true_tensor, normalize=True), name='distance')
  tf.summary.scalar(name='edit_distance', tensor= edit_distance)
  tf.summary.scalar(name='ctc_loss', tensor=crnn_model.loss)
  #tf.summary.scalar(name='learning_rate', tensor=learning_rate)
  merge_summary_op = tf.summary.merge_all()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir)
    summary_writer.add_graph(sess.graph)
    train_next_batch = train_input_fn.get_next()

    save_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    if save_path:
      saver.restore(sess=sess, save_path=save_path)
      print("restore from %s"%(save_path) )
      st = int(save_path.split("-")[-1])
      sess.run(global_step.assign(st))

    for s in range(FLAGS.max_train_steps):
      batch = sess.run(train_next_batch)
      images = batch['images']
      labels = batch['labels']
      sequence_length = batch['sequence_length']
      _, loss , lr,  summary, step, logits, dis = sess.run(
          [train_op, crnn_model.loss, learning_rate, merge_summary_op, global_step , crnn_model.logits , edit_distance ],
          feed_dict = {
            crnn_model.images:images, 
            crnn_model.labels:labels, 
            crnn_model.sequence_length:sequence_length, 
            crnn_model.keep_prob:0.5, 
            crnn_model.is_training:True})

      print("step: {step} lr: {lr} loss: {loss} acc: {dis} ".format(step=step, lr=lr, loss=loss, dis=(1-dis) ))
      if step % FLAGS.step_per_save == 0:
        summary_writer.add_summary(summary=summary, global_step=step)
        saver.save(sess=sess, save_path=model_save_path, global_step=step)

      if False and step % FLAGS.step_per_eval == 0:
        eval_input_fn = input_fn.input_fn(eval_tf_record, FLAGS.batch_size, False, channel_size=FLAGS.channel_size )
        eval_next_batch = eval_input_fn.get_next()
        all_distance =  []
        while True:
          try:
            eval_batch = sess.run(eval_next_batch)
            images = batch['images']
            labels = batch['labels']
            sequence_length = batch['sequence_length']
            train_distance = sess.run([edit_distance], 
                    feed_dict={
                      crnn_model.images:images, 
                      crnn_model.labels:labels, 
                      crnn_model.keep_prob:1.0, 
                      crnn_model.is_training:True, 
                      crnn_model.sequence_length: sequence_length})
            all_distance.append(train_distance[0])
          except tf.errors.OutOfRangeError as e:
            print("eval acc: ", 1 - np.mean(np.array(all_distance)))
            break

if __name__ == "__main__":
  main()

