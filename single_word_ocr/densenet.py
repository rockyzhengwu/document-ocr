#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me

import tensorflow as tf

class DenseNet(object):
  def __init__(self, batch_size, num_classes, mode='train', center_loss_alpha=0.95):
    self.filters = 24
    self.center_loss_alpha = center_loss_alpha

    if mode == 'train':
      self.dropout_rate = 0.5
    else:
      self.dropout_rate = 0.0

    self.num_classes = num_classes
    self.is_training = mode=='train'
    self.images = tf.placeholder(shape=[batch_size, 64, 64, 3], name='image', dtype=tf.float32)
    self.labels = tf.placeholder(shape=[batch_size], name='labels', dtype=tf.int64)
    self.logits = self.dense_net(self.images)

    if self.is_training:
      print("feautres =====> ", self.features)
      #self.center_loss, _ = self.center_loss(self.features, self.labels, self.center_loss_alpha, self.num_classes)
      #self.softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
      self.loss = self.softmax_loss 

    self.predict_prob = tf.nn.softmax(self.logits, name='prob' )
    print("predict_prob====>", self.predict_prob)
    self.prediction = tf.argmax(self.predict_prob, axis=-1, name='prediction')
    print("prediction====>", self.prediction)


    if self.is_training:
      equal = tf.equal(self.labels, self.prediction)
      self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
  
  def center_loss(self, features, label, alpha, num_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
       copy from facenet: https://github.com/davidsandberg/facenet
    """
    num_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, num_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alpha) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
      loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers 



  def bottleneck_layer(self, net, name):
    with tf.variable_scope(name) as scope:
      net = tf.layers.batch_normalization(net, training=self.is_training)
      net = tf.nn.relu(net)
      net = tf.layers.conv2d(net, use_bias=False, filters=4* self.filters, kernel_size=[1, 1], strides=(1, 1) )
      net = tf.nn.dropout(net, keep_prob=1.0-self.dropout_rate)
      net = tf.layers.batch_normalization(net, training=self.is_training)
      net = tf.nn.relu(net)
      net = tf.layers.conv2d(net, use_bias=False, filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
      net = tf.nn.dropout(net, keep_prob=1.0-self.dropout_rate )
      return net
  
  
  def dense_block(self,net, layers, name):
    with tf.variable_scope(name) as scope:
      layers_output = []
      layers_output.append(net)
      net = self.bottleneck_layer(net, 'bottleneck_0')
      layers_output.append(net)
      for i in range(layers-1):
        net = tf.concat(layers_output, axis=-1, name='concat_%d'%(i+1))
        net = self.bottleneck_layer(net, 'bootleneck_%d'%(i+1))
        layers_output.append(net)
      net = tf.concat(layers_output, axis=-1, name='concat_out')
      return net
  
  
  def transition_layer(self, net,  name):
    with tf.variable_scope(name) as scope:
      in_channel_size = net.get_shape().as_list()[-1]
      print("in_channel_size====>", in_channel_size)
      net = tf.layers.batch_normalization(net, training=self.is_training)
      net = tf.nn.relu(net)
      net = tf.layers.conv2d(net, filters= int(in_channel_size*0.5), kernel_size=(1,1), strides=(1,1))
      net = tf.nn.dropout(net, keep_prob=1-self.dropout_rate)
      net = tf.layers.average_pooling2d(net, pool_size=[2,2], strides=2, padding="VALID")
      return net
  
  
  def global_avg_pool(self, net):
    shape = net.get_shape().as_list()
    print(shape)
    width = shape[2]
    height = shape[1]
    net = tf.layers.average_pooling2d(net, pool_size=(height, width), strides=1)
    return net
  
  
  def dense_net(self, x ):
    net = tf.layers.conv2d(inputs=x, 
            use_bias=False, filters=self.filters, kernel_size=(7,7), strides=(2,2), padding="SAME", name='conv_0')
    print("first conv===> ", net)
    net = self.dense_block(net, 6, 'block_0')
    print("block_0 ====>", net)
    net = self.transition_layer(net, 'transition_0') 
    print("transition_0 ====>", net)

    net = self.dense_block(net, 12, 'block_1')
    print("block_1 ====>", net)
    net = self.transition_layer(net, 'transition_1') 
    print("transition_1 ====>", net)
  
    net = self.dense_block(net,  48,  'block_2')
    print("block_2 ====>", net)
    net = self.transition_layer(net, 'transition_2') 
    print("transition_2 ====>", net)

    net = self.dense_block(net, 32, 'block_3')

    net = tf.layers.batch_normalization(net, training=self.is_training)
    net = tf.nn.relu(net)
    net = self.global_avg_pool(net)
    # todo add center loss
    self.features = tf.squeeze(net, name='features')
    net = tf.layers.dense(net, units=self.num_classes, name='linear')
    net = tf.squeeze(net, axis=(1,2), name='logits')
    return net
    
