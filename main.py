import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import variance_scaling_initializer

from ops import *
from utils import *
from statistic import Statistic

flags = tf.app.flags

# network
flags.DEFINE_string("model", "pixel_rnn", "name of model [pixel_rnn, pixel_cnn]")
flags.DEFINE_integer("batch_size", 100, "size of a batch")
flags.DEFINE_integer("hidden_dims", 16, "dimesion of hidden states of LSTM or Conv layers")
flags.DEFINE_integer("recurrent_length", 2, "the length of LSTM or Conv layers")
flags.DEFINE_integer("out_hidden_dims", 32, "dimesion of hidden states of output Conv layers")
flags.DEFINE_integer("out_recurrent_length", 2, "the length of output Conv layers")
flags.DEFINE_boolean("use_residual", False, "whether to use residual connections or not")
flags.DEFINE_boolean("use_dynamic_rnn", False, "whether to use dynamic_rnn or not")

# training
flags.DEFINE_float("max_step", 100000, "# of step in an epoch")
flags.DEFINE_float("test_step", 100, "# of step to test a model")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("max_grad", 1, "max of gradient")
flags.DEFINE_float("min_grad", -1, "min of gradient")
flags.DEFINE_string("data", "mnist", "name of dataset")

# Debug
flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer("random_seed", 123, "random seed for python")

conf = flags.FLAGS

# logging
logger = logging.getLogger()
logger.setLevel(conf.log_level)

# random seed
tf.set_random_seed(conf.random_seed)
np.random.seed(conf.random_seed)

def main(_):
  model_dir = get_model_dir(conf, ['is_train', 'random_seed', 'log_level'])
  preprocess_conf(conf)

  data_format = "NHWC"
  model = "pixel_rnn" # pixel_rnn, pixel_cnn

  if conf.data == "mnist":
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    next_batch = lambda x: mnist.train.next_batch(x)[0]

    height, width, channel = 28, 28, 1

  with tf.Session() as sess:
    logger.info("Building %s." % model)

    if data_format == "NHWC":
      input_shape = [None, height, width, channel]
    elif data_format == "NCHW":
      input_shape = [None, height, width, channel]
    else:
      raise ValueError("Unknown data_format: %s" % data_format)

    l = {}

    l['inputs'] = tf.placeholder(tf.int32, [None, height, width, channel],)
    l['normalized_inputs'] = tf.div(tf.to_float(l['inputs']), 255., name="normalized_inputs")

    # residual connections of PixelRNN
    scope = "conv_inputs"
    logger.info("Building %s: %s" % (model, scope))

    # main reccurent layers
    if conf.use_residual and model == "pixel_rnn":
      l['conv_inputs'] = conv2d(l['normalized_inputs'], conf.hidden_dims * 2, [7, 7], "A", scope=scope)
    else:
      l['conv_inputs'] = conv2d(l['normalized_inputs'], conf.hidden_dims, [7, 7], "A", scope=scope)
    
    l_hid = l[scope]
    for idx in xrange(conf.recurrent_length):
      if model == "pixel_rnn":
        scope = 'diag_bilstm_%d' % idx
        l[scope] = l_hid = diagonal_bilstm(l_hid, conf, scope=scope)
      elif model == "pixel_cnn":
        scope = 'conv2d_B_%d' % idx
        l[scope] = l_hid = conv2d(l_hid, 3, [1, 1], "B", scope=scope)
      logger.info("Building %s: %s" % (model, scope))

    # output reccurent layers
    for idx in xrange(conf.out_recurrent_length):
      scope = 'conv2d_out_%d' % idx
      l[scope] = l_hid = tf.nn.relu(conv2d(l_hid, conf.out_hidden_dims, [1, 1], None, scope=scope))
      logger.info("Building %s: %s" % (model, scope))

    scope = 'output'
    if channel == 1:
      l['conv2d_out_final'] = conv2d(l_hid, 1, [1, 1], None, scope='conv2d_out_final')
      l[scope] = tf.nn.sigmoid(l['conv2d_out_final'])

      loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(l['conv2d_out_final'], l['normalized_inputs'], name='loss'))

      optimizer = tf.train.AdagradOptimizer(conf.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)

      new_grads_and_vars = \
          [(tf.clip_by_value(gv[0], conf.min_grad, conf.max_grad), gv[1]) for gv in grads_and_vars]
      optim = optimizer.apply_gradients(new_grads_and_vars)
    else:
      raise ValueError("Not implemented yet for RGB colors")

    logger.info("Building %s finished!" % model)

    stat = Statistic(sess, conf.data, model_dir, tf.trainable_variables(), conf.test_step)

    tf.initialize_all_variables().run()

    iterator = trange(conf.max_step, ncols=70, initial=stat.get_t())
    for i in iterator:
      images = next_batch(conf.batch_size).reshape([conf.batch_size, height, width, channel])

      _, cost = sess.run([optim, loss], feed_dict={l['inputs']: images})
      iterator.set_description("l: %s" % cost)
      print

      stat.on_step(cost)

if __name__ == "__main__":
  tf.app.run()
