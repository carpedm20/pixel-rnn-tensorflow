import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import variance_scaling_initializer

WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
#WEIGHT_INITIALIZER = tf.uniform_unit_scaling_initializer()

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

  logger.debug('[skew] %s : %s %s -> %s %s' \
      % (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))
  return outputs

def unskew(inputs, width=None, scope="unskew"):
  with tf.name_scope(scope):
    batch, height, skewed_width, channel = get_shape(inputs)
    width = width if width else height

    new_rows = []
    rows = tf.split(1, height, inputs)

    for idx, row in enumerate(rows):
      new_rows.append(tf.slice(row, [0, 0, idx, 0], [-1, -1, width, -1]))
    outputs = tf.concat(1, new_rows, name="output")

  logger.debug('[unskew] %s : %s %s -> %s %s' \
      % (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))
  return outputs

def conv2d(
    inputs,
    num_outputs,
    kernel_shape, # [kernel_height, kernel_width]
    mask_type, # None, "A" or "B",
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    weights_initializer=WEIGHT_INITIALIZER,
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer,
    biases_regularizer=None,
    scope="conv2d"):
  with tf.variable_scope(scope):
    mask_type = mask_type.lower()
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

      if mask_type == 'a':
        mask[center_h,center_w,:,:] = 0.

      weights *= tf.constant(mask, dtype=tf.float32)
      tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    tf.add_to_collection('conv2d_outputs', outputs)

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if activation_fn:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    logger.debug('[conv2d_%s] %s : %s %s -> %s %s' \
        % (mask_type, scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

    return outputs

def conv1d(
    inputs,
    num_outputs,
    kernel_size,
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    weights_initializer=WEIGHT_INITIALIZER,
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
    tf.add_to_collection('conv1d_weights', weights)

    outputs = tf.nn.conv2d(inputs, 
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    tf.add_to_collection('conv1d_outputs', weights)

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if activation_fn:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    logger.debug('[conv1d] %s : %s %s -> %s %s' \
        % (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

    return outputs

def diagonal_bilstm(inputs, conf, scope='diagonal_bilstm'):
  with tf.variable_scope(scope):
    def reverse(inputs):
      return tf.reverse(inputs, [False, False, True, False])

    output_state_fw = diagonal_lstm(inputs, conf, scope='output_state_fw')
    output_state_bw = reverse(diagonal_lstm(reverse(inputs), conf, scope='output_state_bw'))

    tf.add_to_collection('output_state_fw', output_state_fw)
    tf.add_to_collection('output_state_bw', output_state_bw)

    if conf.use_residual:
      residual_state_fw = conv2d(output_state_fw, conf.hidden_dims * 2, [1, 1], "B", scope="residual_fw")
      output_state_fw = residual_state_fw + inputs

      residual_state_bw = conv2d(output_state_bw, conf.hidden_dims * 2, [1, 1], "B", scope="residual_bw")
      output_state_bw = residual_state_bw + inputs

      tf.add_to_collection('residual_state_fw', residual_state_fw)
      tf.add_to_collection('residual_state_bw', residual_state_bw)
      tf.add_to_collection('residual_output_state_fw', output_state_fw)
      tf.add_to_collection('residual_output_state_bw', output_state_bw)

    batch, height, width, channel = get_shape(output_state_bw)

    output_state_bw_except_last = tf.slice(output_state_bw, [0, 0, 0, 0], [-1, height-1, -1, -1])
    output_state_bw_only_last = tf.slice(output_state_bw, [0, height-1, 0, 0], [-1, 1, -1, -1])
    dummy_zeros = tf.zeros_like(output_state_bw_only_last)

    output_state_bw_with_last_zeros = tf.concat(1, [output_state_bw_except_last, dummy_zeros])

    tf.add_to_collection('output_state_bw_with_last_zeros', output_state_bw_with_last_zeros)

    return output_state_fw + output_state_bw_with_last_zeros

def diagonal_lstm(inputs, conf, scope='diagonal_lstm'):
  with tf.variable_scope(scope):
    tf.add_to_collection('lstm_inputs', inputs)

    skewed_inputs = skew(inputs, scope="skewed_i")
    tf.add_to_collection('skewed_lstm_inputs', skewed_inputs)

    # input-to-state (K_is * x_i) : 1x1 convolution. generate 4h x n x n tensor.
    input_to_state = conv2d(skewed_inputs, conf.hidden_dims * 4, [1, 1], "B", scope="i_to_s")
    column_wise_inputs = tf.transpose(
        input_to_state, [0, 2, 1, 3]) # [batch, width, height, hidden_dims * 4]

    tf.add_to_collection('skewed_conv_inputs', input_to_state)
    tf.add_to_collection('column_wise_inputs', column_wise_inputs)

    batch, width, height, channel = get_shape(column_wise_inputs)
    rnn_inputs = tf.reshape(column_wise_inputs,
        [-1, width, height * channel]) # [batch, max_time, height * hidden_dims * 4]

    tf.add_to_collection('rnn_inputs', rnn_inputs)

    rnn_input_list = [tf.squeeze(rnn_input, squeeze_dims=[1]) 
        for rnn_input in tf.split(split_dim=1, num_split=width, value=rnn_inputs)]

    cell = DiagonalLSTMCell(conf.hidden_dims, height, channel)

    if conf.use_dynamic_rnn:
      outputs, states = tf.nn.dynamic_rnn(cell,
          inputs=rnn_inputs, dtype=tf.float32) # [batch, width, height * hidden_dims]
    else:
      output_list, state_list = tf.nn.rnn(cell,
          inputs=rnn_input_list, dtype=tf.float32) # width * [batch, height * hidden_dims]

      packed_outputs = tf.pack(output_list, 1) # [batch, width, height * hidden_dims]
      width_first_outputs = tf.reshape(packed_outputs,
          [-1, width, height, conf.hidden_dims]) # [batch, width, height, hidden_dims]

      skewed_outputs = tf.transpose(width_first_outputs, [0, 2, 1, 3])
      tf.add_to_collection('skewed_outputs', skewed_outputs)

      outputs = unskew(skewed_outputs)
      tf.add_to_collection('unskewed_outputs', outputs)

    return outputs

class DiagonalLSTMCell(rnn_cell.RNNCell):
  def __init__(self, hidden_dims, height, channel):
    self._num_unit_shards = 1
    self._forget_bias = 1.

    self._height = height
    self._channel = channel

    self._hidden_dims = hidden_dims
    self._num_units = self._hidden_dims * self._height
    self._state_size = self._num_units * 2
    self._output_size = self._num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, i_to_s, state, scope="DiagonalBiLSTMCell"):
    c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
    h_prev = tf.slice(state, [0, self._num_units], [-1, self._num_units]) # [batch, height * hidden_dims]

    # i_to_s : [batch, 4 * height * hidden_dims]
    input_size = i_to_s.get_shape().with_rank(2)[1]

    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    with tf.variable_scope(scope):
      # input-to-state (K_ss * h_{i-1}) : 2x1 convolution. generate 4h x n x n tensor.
      conv1d_inputs = tf.reshape(h_prev,
          [-1, self._height, 1, self._hidden_dims], name='conv1d_inputs') # [batch, height, 1, hidden_dims]

      tf.add_to_collection('i_to_s', i_to_s)
      tf.add_to_collection('conv1d_inputs', conv1d_inputs)

      conv_s_to_s = conv1d(conv1d_inputs,
          4 * self._hidden_dims, 2, scope='s_to_s') # [batch, height, 1, hidden_dims * 4]
      s_to_s = tf.reshape(conv_s_to_s,
          [-1, self._height * self._hidden_dims * 4]) # [batch, height * hidden_dims * 4]

      tf.add_to_collection('conv_s_to_s', conv_s_to_s)
      tf.add_to_collection('s_to_s', s_to_s)

      lstm_matrix = tf.sigmoid(s_to_s + i_to_s)

      # i = input_gate, g = new_input, f = forget_gate, o = output_gate
      i, g, f, o = tf.split(1, 4, lstm_matrix)

      c = f * c_prev + i * g
      h = tf.mul(o, tf.tanh(c), name='hid')

    logger.debug('[DiagonalLSTMCell] %s : %s %s -> %s %s' \
        % (scope, i_to_s.name, i_to_s.get_shape(), h.name, h.get_shape()))

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
