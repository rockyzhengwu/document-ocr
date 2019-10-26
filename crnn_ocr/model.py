#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: wu.zheng midday.me

import tensorflow as tf
from tensorflow.contrib import rnn


class CRNN(object):
  def __init__(self, config):
    self.config = config
    self.lstm_num_units = self.config.lstm_num_units
    self.channel_size = config.channel_size
    self.num_classes = config.num_classes + 1
    self.build_graph()
    self.train_loss()

  def build_graph(self):
    self.images = tf.placeholder(shape=[None, 32, None, self.channel_size], name='images', dtype=tf.float32)
    self.batch_size = tf.shape(self.images)[0]
    self.labels = tf.sparse.placeholder(name="labels", dtype=tf.int32)

    self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    self.is_training = tf.placeholder(tf.bool, name='training')

    self.sequence_length = tf.placeholder(shape=[None], dtype=tf.int32, name='sequence_length')
    conv_output = self.cnn(self.images) # batch_size, 1, w / 4, 512
    conv_output = tf.transpose(conv_output, (0, 2, 1, 3))
    self.conv_output = tf.squeeze(conv_output, axis=2)
    self.bidirectionnal_rnn(self.conv_output, self.sequence_length)


  def train_loss(self):
    loss = tf.nn.ctc_loss(
        labels=self.labels,
        inputs=self.logits,
        sequence_length=self.sequence_length,
        ignore_longer_outputs_than_inputs=True,
        #ctc_merge_repeated=True
    )
    self.loss = tf.reduce_mean(loss, name='loss')
    return self.loss


  def bidirectionnal_rnn(self, input_tensor, input_sequence_length):
    lstm_num_units = self.config.lstm_num_units
    print("rnn input tensor ===> ", input_tensor)
    with tf.variable_scope('lstm_layers'):
      fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0, name='fw_cell_%d'%(nh)) for nh in [lstm_num_units] * 2]
      bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0, name='bw_cell_%d'%(nh)) for nh in [lstm_num_units] * 2]

      stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
          cells_fw=fw_cell_list, 
          cells_bw=bw_cell_list, 
          inputs=input_tensor, 
          sequence_length=input_sequence_length, 
          dtype=tf.float32)
      hidden_num = lstm_num_units * 2
      rnn_reshaped = tf.nn.dropout(stack_lstm_layer, keep_prob=self.keep_prob)
      w = tf.get_variable(initializer=tf.truncated_normal([hidden_num, self.num_classes], stddev=0.02), name="w")
      w_t = tf.tile(tf.expand_dims(w, 0),[self.batch_size,1,1])
      logits = tf.matmul(rnn_reshaped, w_t, name="nn_logits")
      self.logits = tf.identity(tf.transpose(logits, (1, 0, 2)), name='logits')
      return logits

  def _build_pred(self):
    decoded, log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.sequence_length)
    self.decoded = tf.identity(decoded[0], name='decoded')
    self.log_prob = tf.identity(log_prob, name='log_prob')
    if self.is_training:
      pred_str_labels = tf.as_string(self.decoded.values)
      pred_tensor = tf.SparseTensor(indices=self.decoded.indices, values=pred_str_labels, dense_shape=self.decoded.dense_shape)
      true_str_labels = tf.as_string(self.labels.values)
      true_tensor = tf.SparseTensor(indices=self.labels.indices, values=true_str_labels, dense_shape=self.labels.dense_shape)
      self.edit_distance = tf.reduce_mean(tf.edit_distance(pred_tensor, true_tensor, normalize=True), name='distance')

  def cnn(self, inputs):
    with tf.variable_scope('cnn_feature'):
      # (None, 32, w, 64)
      conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name="conv1",)
      # (None, 16, w/2, 64)
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1', padding='valid')
      # (None, 16, w/2, 128)
      conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name="conv2")
      # (None, 8, w/4, 128)
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2", padding='valid')
      # (None, 8, w/4, 256)
      conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name="conv3")
      conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name="conv4")
      # (None, 4, w/4, 256), 
      pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 1], strides=[2, 1], padding="valid", name="pool3")
      # (None, 4, w/4, 512)
      conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3), padding="same", activation=None, name='conv5')
      # (None, 4, w/4, 512)
      bnorm1 = tf.layers.batch_normalization(conv5, name="bnorm1", training=self.is_training)
      bnorm1 = tf.nn.relu(bnorm1)
      # (None, 4, w/4, 512)
      conv6 = tf.layers.conv2d(inputs=bnorm1, filters=512, kernel_size=(3, 3), padding="same", activation=None, name="conv6")
      bnorm2 = tf.layers.batch_normalization(conv6, name="bnorm2", training=self.is_training)
      bnorm2 = tf.nn.relu(bnorm2)
      # (None, 2, w/4, 512)
      pool4 = tf.layers.max_pooling2d(inputs=bnorm2, pool_size=[2, 1], strides=[2, 1], padding="valid", name="pool4")
      conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=2, strides=[2, 1], padding="same", activation=tf.nn.relu, name="conv7")
      return conv7

