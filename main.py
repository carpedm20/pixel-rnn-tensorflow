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
    height, width, channel = 40, 30, 3

    if data_format == "NHWC":
      input_shape = [None, height, width, channel]
    elif data_format == "NCHW":
      input_shape = [None, height, width, channel]
    else:
      raise ValueError("Unknown data_format: %s" % data_format)

    inputs = tf.placeholder(tf.int32, [None, height, width, channel],)
    normalized_inputs = tf.div(tf.to_float(inputs), 255., name="normalized_inputs")

    if model == "pixel_rnn":
      network = diagonal_bilstm(normalized_inputs, conf.hidden_dims)
    elif model == "pixel_cnn":
      conved_inputs = conv2d(normalized_inputs, 3, [5, 3], [1, 1], "A", scope="conv")
    else:
      raise ValueError("Wrong model: %s" % model)

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

if __name__ == "__main__":
  tf.app.run()
