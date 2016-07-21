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

  data_format = "NHWC"
  model = "pixel_rnn" # pixel_rnn, pixel_cnn

  if conf.data == "mnist":
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    next_batch = lambda x: mnist.train.next_batch(x)[0]

    height, width, channel = 28, 28, 1

  with tf.Session() as sess:
    logger.info("Building %s starts!" % model)

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

    if conf.use_residual and model == "pixel_rnn":
      l[scope] = conv2d(l['normalized_inputs'], conf.hidden_dims * 2, [7, 7], "A", scope=scope)
    else:
      l[scope] = conv2d(l['normalized_inputs'], conf.hidden_dims, [7, 7], "A", scope=scope)
    
    # main reccurent layers
    l_hid = l[scope]
    for idx in xrange(conf.recurrent_length):
      if model == "pixel_rnn":
        scope = 'LSTM%d' % idx
        l[scope] = l_hid = diagonal_bilstm(l_hid, conf, scope=scope)
      elif model == "pixel_cnn":
        scope = 'CONV%d' % idx
        l[scope] = l_hid = conv2d(l_hid, 3, [1, 1], "B", scope=scope)
      logger.info("Building %s" % scope)

    # output reccurent layers
    for idx in xrange(conf.out_recurrent_length):
      scope = 'CONV_OUT%d' % idx
      l[scope] = l_hid = tf.nn.relu(conv2d(l_hid, conf.out_hidden_dims, [1, 1], None, scope=scope))
      logger.info("Building %s" % scope)

    if channel == 1:
      l['conv2d_out_final'] = conv2d(l_hid, 1, [1, 1], None, scope='conv2d_out_final')
      l['output'] = tf.nn.sigmoid(l['conv2d_out_final'])

      logger.info("Building loss and optims")
      loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(l['conv2d_out_final'], l['normalized_inputs'], name='loss'))

      optimizer = tf.train.RMSPropOptimizer(conf.learning_rate)
      #optimizer = tf.train.AdamOptimizer(conf.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)

      new_grads_and_vars = \
          [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]
      optim = optimizer.apply_gradients(new_grads_and_vars)
      #optim = optimizer.minimize(loss)
    else:
      raise ValueError("Not implemented yet for RGB colors")

    logger.info("Building %s finished!" % model)
    show_all_variables()

    #stat = Statistic(sess, conf.data, model_dir, tf.trainable_variables(), conf.test_step)
    stat = None
    initial_step = stat.get_t() if stat else 0

    logger.info("Initializing all variables")
    tf.initialize_all_variables().run()

    logger.info("Training starts!")

    def binarize(images):
      return (np.random.uniform(size=images.shape) < images).astype('float32')

    G = tf.get_default_graph()

    iterator = trange(conf.max_step, ncols=70, initial=initial_step)
    for i in iterator:
      #images = binarize(next_batch(conf.batch_size)).reshape([conf.batch_size, height, width, channel])
      images = np.ones([100, 28, 28, 1])
      images[:,10:20,:,:]=0
      _, cost, output, output_final = sess.run([optim, loss, l['output'], l['conv2d_out_final']], feed_dict={l['inputs']: images})
      print
      print mprint(images[1])
      print mprint(output[1], 0.5)
      print mprint(output_final[1], 0.5)

      #cost, out1, out2, conv = sess.run(
      #    [loss, l['output'], l['conv2d_out_final'], l['conv_inputs']], feed_dict={l['inputs']: images})
      #print
      #print out1[0][10][:9].tolist()
      #print out2[0][10][:9].tolist()
      #print images[0][10][:9].tolist()
      #print conv[0][10][:9,3].tolist()
      #print out1[0].max(), out1[0].min(), out2[0].max(), out2[0].min()

      #_, cost, out1, out2, conv = sess.run(
      #    [optim, loss, l['output'], l['conv2d_out_final'], l['conv_inputs']], feed_dict={l['inputs']: images})
      #print
      #print out1[0][10][:9].tolist()
      #print out2[0][10][:9].tolist()
      #print images[0][10][:9].tolist()
      #print conv[0][10][:9,3].tolist()
      #print out1[0].max(), out1[0].min(), out2[0].max(), out2[0].min()
      #import ipdb; ipdb.set_trace() 

      # G.get_collection('conv2d_weights')[0].eval()[:,:,0,0]

      # tf.get_collection('lstm_inputs')[0].eval({l['inputs']: images})[0,:,:,0]
      # mprint(tf.get_collection('lstm_inputs')[0].eval({l['inputs']: images})[1,:,:,0], 0.0007746)
      # mprint(tf.get_collection('skewed_lstm_inputs')[0].eval({l['inputs']: images})[1,:,:,0], 0.0007746)
      # tf.get_collection('skewed_conv_inputs')[0].eval({l['inputs']: images})[1,:,:,0]
      # mprint(tf.get_collection('skewed_conv_inputs')[0].eval({l['inputs']: images})[1,:,:,0], 0)

      # mprint(G.get_collection('column_wise_inputs')[0].eval({l['inputs']: images})[1,:,:,0], 0)
      # mprint(G.get_collection('rnn_inputs')[0].eval({l['inputs']: images})[1,:,::4*64],0)
      # for i in xrange(55): mprint([tf.get_collection('i_to_s')[i].eval({l['inputs']: images})[1,::4*64]], 0)

      # tf.get_collection('conv2d_outputs')[0].eval({l['inputs']: images})
      # mprint(tf.get_collection('conv2d_outputs')[0].eval({l['inputs']: images})[1,:,:,1], 0)
      # mprint(tf.get_collection('output_state_fw')[0].eval({l['inputs']: images})[1,:,:,-1], 0.3)

      if stat:
        stat.on_step(cost)
      iterator.set_description("l: %s" % cost)

if __name__ == "__main__":
  tf.app.run()
