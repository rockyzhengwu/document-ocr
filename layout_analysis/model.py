#-*-coding:utf-8-*-


import tensorflow as tf

def down_block(inputs, filters, block_index, is_training):
  with tf.variable_scope('g_encoder') as scope:
    inputs = tf.pad(inputs, paddings=[[0, 0], [1,1], [1, 1],[0, 0]], name="pad_%d"%(block_index))
    out = tf.layers.conv2d(inputs, filters=filters, kernel_size=[4,4], strides=2, activation=tf.nn.leaky_relu,
                           name="g_conv_%d"%(block_index))
    return out


def up_block(input_a, input_b, out_filters, block_index, is_training, name_scope='g_decoder'):
  with tf.variable_scope(name_scope) as scope:
    inputs = tf.concat([input_a, input_b], axis=-1)
    out = tf.layers.conv2d_transpose(inputs, filters=out_filters, kernel_size=[4,4], strides=2, activation=None,
                                     name="g_up_cov_%d"%(block_index), padding="SAME")
    out = tf.layers.batch_normalization(out, training=is_training, name='g_up_norm_%d'%(block_index))
    out = tf.nn.relu(out)
    return out


def conv_norm_leakrelu_layer(inputs, filters, is_training, scope_name):
  with tf.variable_scope(scope_name) as scope:
    out = tf.pad(inputs, paddings=[[0, 0], [1,1], [1,1], [0, 0]] )
    out = tf.layers.conv2d(out, filters=filters, kernel_size=[4,4], strides=[2, 2], activation=None)
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.leaky_relu(out )
    return out

class UnetModel(object):
  def __init__(self, num_class, is_training=True, ):
    self.image = tf.placeholder(shape=[None, None, None, 3], name="image", dtype=tf.float32)
    self.label = tf.placeholder(shape=[None, None, None, 1], name='label', dtype=tf.int32)
    self.is_training = is_training
    self.num_class = num_class
    self._build_graph(self.is_training)
    self.prob = tf.sigmoid(self.logits, name='prob')
    self.prediction = tf.argmax(self.logits, axis=-1, name='prediction')

    self.filter_size_list = [64, 128, 256, 512, 512, 512]
    if is_training:
        self._loss()

  def _build_graph(self, is_training):
    with tf.variable_scope('encoder') as scope:
      block1 = down_block(self.image, 64, 1, is_training)
      block2 = down_block(block1, 128, 2, is_training)
      block3 = down_block(block2, 256, 3, is_training)
      block4 = down_block(block3, 512, 4, is_training)
      block5 = down_block(block4, 512, 5, is_training)
      block6 = down_block(block5, 512, 6, is_training)
      block7 = down_block(block6, 512, 7, is_training)
      center = down_block(block7, 512, 8, is_training)

    with tf.variable_scope('decoder') as scope:
      center = tf.layers.conv2d_transpose(center, 512, kernel_size=[4,4], strides=2,
                                          activation=tf.nn.relu, padding="SAME", name='g_center')

      center_norm = tf.layers.batch_normalization(center, training=is_training, name='g_center_norml')
      upblock7 = up_block(block7, center_norm, 512, 7, is_training)
      upblock6 = up_block(block6, upblock7, 512, 6, is_training)
      upblock5 = up_block(block5, upblock6, 512, 4, is_training)
      upblock4 = up_block(block4, upblock5, 256, 3, is_training)
      upblock3 = up_block(block3, upblock4, 128, 2, is_training )
      upblock2 = up_block(block2, upblock3, 64, 1, is_training)

      out = tf.concat([block1, upblock2], axis=-1)
      out = tf.layers.conv2d_transpose(out, filters=self.num_class, kernel_size=(4, 4),
        strides=2, padding="SAME", activation=None, name='g_out')
      self.logits = tf.identity(out, name='logits')

    if is_training:
      # auxiliary decoder
      decoder_scope = 'auxiliary'
      with tf.variable_scope(decoder_scope) as scope:
        aux_upblock7 = up_block(block7, center_norm, 512, 7, is_training, decoder_scope)
        aux_upblock6 = up_block(block6, aux_upblock7, 512, 6, is_training, decoder_scope)
        aux_upblock5 = up_block(block5, aux_upblock6, 512, 4, is_training, decoder_scope)
        aux_upblock4 = up_block(block4, aux_upblock5, 256, 3, is_training, decoder_scope)
        aux_upblock3 = up_block(block3, aux_upblock4, 128, 2, is_training, decoder_scope)
        aux_upblock2 = up_block(block2, aux_upblock3, 64, 1, is_training, decoder_scope)
        aux_out = tf.concat([block1, aux_upblock2], axis=-1)
        self.aux_out = tf.layers.conv2d_transpose(aux_out, filters=3, kernel_size=(4, 4),
            strides=2, padding="SAME", activation=None, name='aux_g_out')

          

  def _loss(self):
    shaped_logits = tf.reshape(self.logits, shape=[-1, self.num_class])
    labels = tf.reshape(self.label, shape=[-1, ])
    self.aux_loss = tf.losses.mean_squared_error(self.image, self.aux_out)
    self.aux_loss = 10 * self.aux_loss
    self.seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=shaped_logits))
    self.loss = self.aux_loss + self.seg_loss
    self.acc, self.acc_op = tf.metrics.accuracy(labels=labels,  predictions=tf.argmax(shaped_logits, 1))


