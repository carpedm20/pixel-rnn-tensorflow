import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from ops import *
from utils import *
from statistic import Statistic

flags = tf.app.flags

# network
flags.DEFINE_string("model", "pixel_cnn", "name of model [pixel_rnn, pixel_cnn]")
flags.DEFINE_integer("batch_size", 100, "size of a batch")
flags.DEFINE_integer("hidden_dims", 64, "dimesion of hidden states of LSTM or Conv layers")
flags.DEFINE_integer("recurrent_length", 2, "the length of LSTM or Conv layers")
flags.DEFINE_integer("out_hidden_dims", 64, "dimesion of hidden states of output Conv layers")
flags.DEFINE_integer("out_recurrent_length", 2, "the length of output Conv layers")
flags.DEFINE_boolean("use_residual", False, "whether to use residual connections or not")
flags.DEFINE_boolean("use_dynamic_rnn", False, "whether to use dynamic_rnn or not")

# training
flags.DEFINE_float("max_step", 100000, "# of step in an epoch")
flags.DEFINE_float("test_step", 100, "# of step to test a model")
flags.DEFINE_float("save_step", 1000, "# of step to save a model")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")
flags.DEFINE_string("data", "mnist", "name of dataset")
flags.DEFINE_boolean("use_gpu", True, "whther to use gpu for training")

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
  model_dir = get_model_dir(conf, 
      ['max_step', 'test_step', 'save_step', 'is_train', 'random_seed', 'log_level'])
  preprocess_conf(conf)

  if conf.use_gpu:
    data_format = "NHWC"
  else:
    data_format = "NCHW"

  if conf.data == "mnist":
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    next_batch = lambda x: mnist.train.next_batch(x)[0]

    height, width, channel = 28, 28, 1

  with tf.Session() as sess:
    logger.info("Building %s starts!" % conf.model)

    if data_format == "NHWC":
      input_shape = [None, height, width, channel]
    elif data_format == "NCHW":
      input_shape = [None, height, width, channel]
    else:
      raise ValueError("Unknown data_format: %s" % data_format)

    l = {}

    l['inputs'] = tf.placeholder(tf.float32, [None, height, width, channel],)

    if conf.data =='mnist':
      l['normalized_inputs'] = l['inputs']
    else:
      l['normalized_inputs'] = tf.div(l['inputs'], 255., name="normalized_inputs")

    # input of main reccurent layers
    scope = "conv_inputs"
    logger.info("Building %s" % scope)

    if conf.use_residual and conf.model == "pixel_rnn":
      l[scope] = conv2d(l['normalized_inputs'], conf.hidden_dims * 2, [7, 7], "A", scope=scope)
    else:
      l[scope] = conv2d(l['normalized_inputs'], conf.hidden_dims, [7, 7], "A", scope=scope)
    
    # main reccurent layers
    l_hid = l[scope]
    for idx in xrange(conf.recurrent_length):
      if conf.model == "pixel_rnn":
        scope = 'LSTM%d' % idx
        l[scope] = l_hid = diagonal_bilstm(l_hid, conf, scope=scope)
      elif conf.model == "pixel_cnn":
        scope = 'CONV%d' % idx
        l[scope] = l_hid = conv2d(l_hid, 3, [1, 1], "B", scope=scope)
      else:
        raise ValueError("wrong type of model: %s" % (conf.model))
      logger.info("Building %s" % scope)

    # output reccurent layers
    for idx in xrange(conf.out_recurrent_length):
      scope = 'CONV_OUT%d' % idx
      l[scope] = l_hid = tf.nn.relu(conv2d(l_hid, conf.out_hidden_dims, [1, 1], "B", scope=scope))
      logger.info("Building %s" % scope)

    if channel == 1:
      l['conv2d_out_logits'] = conv2d(l_hid, 1, [1, 1], "B", scope='conv2d_out_logits')
      l['output'] = tf.nn.sigmoid(l['conv2d_out_logits'])

      logger.info("Building loss and optims")
      loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(l['conv2d_out_logits'], l['normalized_inputs'], name='loss'))

      optimizer = tf.train.RMSPropOptimizer(conf.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)

      new_grads_and_vars = \
          [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]
      optim = optimizer.apply_gradients(new_grads_and_vars)
    else:
      raise ValueError("Not implemented yet for RGB colors")

    logger.info("Building %s finished!" % conf.model)

    show_all_variables()
    stat = Statistic(sess, conf.data, model_dir, tf.trainable_variables(), conf.test_step)

    logger.info("Initializing all variables")

    tf.initialize_all_variables().run()
    stat.load_model()

    if conf.is_train:
      logger.info("Training starts!")

      initial_step = stat.get_t() if stat else 0
      iterator = trange(conf.max_step, ncols=70, initial=initial_step)

      for i in iterator:
        if conf.data == 'mnist':
          images = binarize(next_batch(conf.batch_size)) \
            .reshape([conf.batch_size, height, width, channel])

        _, cost, output = sess.run([
            optim, loss, l['output']
          ], feed_dict={l['inputs']: images})

        if conf.data == 'mnist':
          print
          print mprint(images[1])
          print mprint(output[1], 0.5)

        if stat:
          stat.on_step(cost)

        iterator.set_description("l: %s" % cost)
    else:
      logger.info("Image generation starts!")
      samples = np.zeros((100, height, width, 1), dtype='float32')

      for i in xrange(height):
        for j in xrange(width):
          for k in xrange(channel):
            next_sample = binarize(l['output'].eval({l['inputs']: samples}))
            samples[:, i, j, k] = next_sample[:, i, j, k]

            if conf.data == 'mnist':
              mprint(binarize(samples[0,:,:,:]))

      save_images(samples, height, width, 10, 10)


if __name__ == "__main__":
  tf.app.run()
