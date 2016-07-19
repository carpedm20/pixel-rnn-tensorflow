import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import variance_scaling_initializer

from ops import *
from utils import *

flags = tf.app.flags

# network
flags.DEFINE_string("model", "pixel_rnn", "name of model [pixel_rnn, pixel_cnn]")
flags.DEFINE_integer("hidden_dims", 64, "dimesion of hidden states in DiagonalBiLSTM")

# Debug
flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")

conf = flags.FLAGS

# logging
logger = logging.getLogger()
logger.setLevel(conf.log_level)

def main(_):
  data_format = "NHWC"
  model = "pixel_rnn" # pixel_rnn, pixel_cnn

  with tf.Session() as sess:
    height, width, channel = 64, 64, 3
    hidden_dims = 64

    if data_format == "NHWC":
      input_shape = [None, height, width, channel]
    elif data_format == "NCHW":
      input_shape = [None, height, width, channel]
    else:
      raise ValueError("Unknown data_format: %s" % data_format)

    l = {}

    l['inputs']= tf.placeholder(tf.int32, [None, height, width, channel],)
    l['normalized_inputs'] = tf.div(tf.to_float(l['inputs']), 255., name="normalized_inputs")
    l['conv_inputs'] = conv2d(l['normalized_inputs'], hidden_dims, [7, 7], "A", scope="conv_inputs")
    
    if model == "pixel_rnn":
      network = diagonal_bilstm(l['conv_inputs'], conf.hidden_dims)
    elif model == "pixel_cnn":
      conved_inputs = conv2d(l['conv_inputs'], 3, [5, 3], [1, 1], "A", scope="conv")
    else:
      raise ValueError("Wrong model: %s" % model)

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

if __name__ == "__main__":
  tf.app.run()
