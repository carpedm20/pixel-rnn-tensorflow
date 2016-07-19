import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import variance_scaling_initializer

logger = logging.getLogger(__name__)

he_uniform = variance_scaling_initializer(factor=2.0, mode="FAN_IN", uniform=False)
data_format = "NCHW"

def get_shape(layer):
  return layer.get_shape().as_list()

#def get_shape(layer):
#  if data_format == "NHWC":
#    batch, height, width, channel = layer.get_shape().as_list()
#  elif data_format == "NCHW":
#    batch, channel, height, width = layer.get_shape().as_list()
#  else:
#    raise ValueError("Unknown data_format: %s" % data_format)
#  return batch, height, width, channel

def skew(inputs, scope="skew"):
  with tf.name_scope(scope):
    batch, height, width, channel = get_shape(inputs) # [batch, height, width, channel]
    rows = tf.split(1, height, inputs) # [batch, 1, width, channel]

    new_width = width + height - 1
    new_rows = []

    for idx, row in enumerate(rows):
      transposed_row = tf.transpose(tf.squeeze(row, [1]), [0, 2, 1]) # [batch, channel, width]
      squeezed_row = tf.reshape(transposed_row, [-1, width]) # [batch*channel, width]
      padded_row = tf.pad(squeezed_row, ((0, 0), (idx, height - 1 - idx))) # [batch*channel, width*2-1]

      unsqueezed_row = tf.reshape(padded_row, [-1, channel, new_width]) # [batch, channel, width*2-1]
      untransposed_row = tf.transpose(unsqueezed_row, [0, 2, 1]) # [batch, width*2-1, channel]

      assert get_shape(untransposed_row) == [batch, new_width, channel], "wrong shape of skewed row"
      new_rows.append(untransposed_row)

    outputs = tf.pack(new_rows, axis=1, name="output")
    assert get_shape(outputs) == [None, height, new_width, channel], "wrong shape of skewed output"

    logger.info('[skew] %s : %s %s -> %s %s' \
        % (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))
  return outputs

def unskew(inputs, height):
  rows = tf.split(1, height, inputs)

  new_rows = []
  for idx, row in enumerate(rows):
    new_rows.append(tf.slice(row, [0, idx], [-1, idx + height]))

  return tf.pack(new_rows, axis=1)

def conv2d(
    inputs,
    num_outputs,
    kernel_shape, # [kernel_height, kernel_width]
    mask_type, # None, "A" or "B",
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer,
    biases_regularizer=None,
    scope="conv2d"):
  with tf.variable_scope(scope):
    batch_size, height, width, channel = inputs.get_shape().as_list()

    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, \
      "kernel height and width should be odd number"

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    weights_shape = [kernel_h, kernel_w, channel, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, weights_initializer, weights_regularizer)

    if mask_type is not None:
      mask = np.ones(
        (kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)

      mask[center_h, center_w+1: ,: ,:] = 0.
      mask[center_h+1:, :, :, :] = 0.

      if mask_type == "A":
        mask[center_h,center_w,:,:] = 0.

      weights *= tf.constant(mask, dtype=tf.float32)

    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if activation_fn:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    logger.info('[conv2d_%s] %s : %s %s -> %s %s' \
        % (mask_type, scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

    return outputs

def conv1d(
    inputs,
    num_outputs,
    kernel_size,
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer,
    biases_regularizer=None,
    scope="conv1d"):
  with tf.variable_scope(scope):
    batch_size, height, _, channel = inputs.get_shape().as_list() # [batch, height, 1, channel]

    kernel_h, kernel_w = kernel_size, 1
    stride_h, stride_w = strides

    weights_shape = [kernel_h, kernel_w, channel, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, weights_initializer, weights_regularizer)

    outputs = tf.nn.conv2d(inputs, 
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if activation_fn:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    logger.info('[conv1d] %s : %s %s -> %s %s' \
        % (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

    return outputs

def diagonal_bilstm(inputs, hidden_dims, use_dynamic_rnn=False):
  skewed_inputs = skew(inputs, "skewed_i")
  column_wise_inputs = tf.transpose(skewed_inputs, [2, 0, 1, 3]) # [width, batch, height, channel]

  width, batch, height, channel = get_shape(column_wise_inputs)
  rnn_inputs = tf.reshape(column_wise_inputs, [-1, width, height * channel]) # [batch, max_time, height * channel]

  rnn_input_list = [tf.squeeze(rnn_input, squeeze_dims=[1]) 
      for rnn_input in tf.split(split_dim=1, num_split=width, value=rnn_inputs)]

  cell = DiagonalLSTMCell(hidden_dims, height, channel)

  if use_dynamic_rnn:
    outputs, states = tf.nn.dynamic_rnn(cell,
        inputs=rnn_inputs, dtype=tf.float32) # [batch, width, height * channel]
  else:
    output_list, state_list = tf.nn.rnn(cell,
        inputs=rnn_input_list, dtype=tf.float32) # width * [batch, channel]

    import ipdb; ipdb.set_trace() 
    conv_outputs = tf.reshape(tf.transpose(outputs, [])) # [batch, max_time, input_size]
    conv_states = tf.reshape(states, [])

  print outputs.get_shape()
  print states.get_shape()

class DiagonalLSTMCell(rnn_cell.RNNCell):
  def __init__(self, num_units, height, channel):
    self._num_unit_shards = 1
    self._forget_bias = 1.

    self._height = height
    self._channel = channel

    self._num_units = num_units * self._height
    self._state_size = self._num_units * 2
    self._output_size = self._num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope="DiagonalBiLSTMCell"):
    c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
    h_prev = tf.slice(state, [0, self._num_units], [-1, self._num_units])

    input_size = inputs.get_shape().with_rank(2)[1]

    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    with tf.variable_scope(scope):
      # K_is * x_i
      conv2d_inputs = tf.reshape(inputs, [-1, self._height, 1, self._channel])
      input_to_state = conv2d(conv2d_inputs, self._num_units * 4, [1, 1], "B", scope="i_to_s")
      import ipdb; ipdb.set_trace() 

      # K_ss * h_{i-1}
      states = conv1d(h_prev, 4 * self._num_units, 2, scope='s_to_s')

      # i = input_gate, g = new_input, f = forget_gate, o = output_gate
      cell_inputs = tf.concat(1, [inputs, h_prev])
      lstm_matrix = tf.nn.bias_add(tf.matmul(cell_inputs, concat_w), b)
      i, g, f, o = tf.split(1, 4, lstm_matrix)

      c = f * c_prev + i * g
      h = o * tf.tanh(c)

    logger.info('[DiagonalLSTMCell] %s : %s %s -> %s %s' \
        % (scope, inputs.name, inputs.get_shape(), h.name, h.get_shape()))

    new_state = tf.concat(1, [c, h])
    return h, new_state

class RowLSTMCell(rnn_cell.RNNCell):
  def __init__(self, num_units, kernel_shape=[3, 1]):
    self._num_units = num_units
    self._state_size = num_units * 2
    self._output_size = num_units
    self._kernel_shape = kernel_shape

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope="RowLSTMCell"):
    raise Exception("Not implemented")
