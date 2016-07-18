from logging import getLogger

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import variance_scaling_initializer

flags = tf.app.flags

# Debug
flags.DEFINE_boolean('is_train', True, 'training or testing')
flags.DEFINE_string('log_level', 'INFO', 'log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')

conf = flags.FLAGS

logger = getLogger()
logger.propagate = False
logger.setLevel(conf.log_level)

he_uniform = variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
data_format = 'NCHW'

def get_shape(layer):
  return layer.get_shape().as_list()

#def get_shape(layer):
#  if data_format == 'NHWC':
#    batch, height, width, channel = layer.get_shape().as_list()
#  elif data_format == 'NCHW':
#    batch, channel, height, width = layer.get_shape().as_list()
#  else:
#    raise ValueError('Unknown data_format: %s' % data_format)
#  return batch, height, width, channel

def skew(layer):
  with tf.name_scope('skew'):
    batch, height, width, channel = get_shape(layer) # [batch, height, width, channel]
    rows = tf.split(1, height, layer) # [batch, 1, width, channel]

    new_width = width + height - 1
    new_rows = []

    for idx, row in enumerate(rows):
      transposed_row = tf.transpose(tf.squeeze(row, [1]), [0, 2, 1]) # [batch, channel, width]
      squeezed_row = tf.reshape(transposed_row, [-1, width]) # [batch*channel, width]
      padded_row = tf.pad(squeezed_row, ((0, 0), (idx, height - 1 - idx))) # [batch*channel, width*2-1]

      unsqueezed_row = tf.reshape(padded_row, [-1, channel, new_width]) # [batch, channel, width*2-1]
      untransposed_row = tf.transpose(unsqueezed_row, [0, 2, 1]) # [batch, width*2-1, channel]

      assert get_shape(untransposed_row) == [batch, new_width, channel], 'wrong shape of skewed row'
      new_rows.append(untransposed_row)

    result = tf.pack(new_rows, axis=1)
    assert get_shape(result) == [None, height, new_width, channel], 'wrong shape of skewed output'
  return result

def unskew(layer, height):
  rows = tf.split(1, height, layer)

  new_rows = []
  for idx, row in enumerate(rows):
    new_rows.append(tf.slice(row, [0, idx], [-1, idx + height]))

  return tf.pack(new_rows, axis=1)

def conv2d(
    layer,
    num_outputs,
    kernel_size, # [kernel_height, kernel_width]
    stride, # [column_wise_stride, row_wise_stride]
    mask_type, # None, 'A' or 'B',
    padding='VALID',
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer,
    biases_regularizer=None,
    scope=None):
  with tf.variable_scope(scope):
    num_filters_in = layer.get_shape().as_list()[3]

    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride_size

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, \
      'kernel height and width should be odd number'

    if mask_type == 'A':
      center_h = kernel_h // 2
      center_w = kernel_w // 2

      kernel_length = kernel_h * kernel_w
      default_length = center_h * kernel_w + center_w
      
      ws = []
      for idx in xrange(num_outputs):
        for jdx in xrange(num_filters_in):
          if idx <= jdx:
            length = default_length
          else:
            length = default_length + 1

          w = tf.get_variable('weights', length,
            tf.float32, weights_initializer, weights_regularizer)
          ws.append(tf.pad(w, ((0, 0), (0, kernel_length-length))))

      weights = tf.pack(ws, axis=1)
      import ipdb; ipdb.set_trace() 

    weights_shape = [kernel_h, kernel_w, num_filters_in, num_outputs]
    weights = tf.get_variable('weights', weights_shape,
      tf.float32, weights_initializer, weights_regularizer)
    outputs = tf.nn.conv2d(layer, weights, [1, stride_h, stride_w, 1], padding=padding)

    if biases_initializer != None:
      biases = tf.get_variable('biases', [num_outputs, ],
        tf.float32, biases_initializer, biases_regularizer)
      outputs = nn.bias_add(outputs, biases)

    if activation_fn:
      outputs = activation_fn(outputs)

    return outputs

with tf.Session() as sess:
  data_format = 'NHWC'
  height, width, channel = 4, 3, 1

  if data_format == 'NHWC':
    input_shape = [None, height, width, channel]
  elif data_format == 'NCHW':
    input_shape = [None, height, width, channel]
  else:
    raise ValueError('Unknown data_format: %s' % data_format)

  inputs = tf.placeholder(tf.int32, [None, height, width, channel],)
  skewed_inputs = skewed(inputs)
  conved_inputs = conv2d(skewed_inputs, 3, [5, 5], [1, 1], 'A', scope='conv')

  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
