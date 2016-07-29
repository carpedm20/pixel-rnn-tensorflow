import tensorflow as tf
from logging import getLogger

from ops import *
from utils import *

logger = getLogger(__name__)

class Network:
  def __init__(self, sess, conf, height, width, channel):
    logger.info("Building %s starts!" % conf.model)

    self.sess = sess
    self.data = conf.data
    self.height, self.width, self.channel = height, width, channel

    if conf.use_gpu:
      data_format = "NHWC"
    else:
      data_format = "NCHW"

    if data_format == "NHWC":
      input_shape = [None, height, width, channel]
    elif data_format == "NCHW":
      input_shape = [None, channel, height, width]
    else:
      raise ValueError("Unknown data_format: %s" % data_format)

    self.l = {}

    self.l['inputs'] = tf.placeholder(tf.float32, [None, height, width, channel],)

    if conf.data =='mnist':
      self.l['normalized_inputs'] = self.l['inputs']
    else:
      self.l['normalized_inputs'] = tf.div(self.l['inputs'], 255., name="normalized_inputs")

    # input of main reccurent layers
    scope = "conv_inputs"
    logger.info("Building %s" % scope)

    if conf.use_residual and conf.model == "pixel_rnn":
      self.l[scope] = conv2d(self.l['normalized_inputs'], conf.hidden_dims * 2, [7, 7], "A", scope=scope)
    else:
      self.l[scope] = conv2d(self.l['normalized_inputs'], conf.hidden_dims, [7, 7], "A", scope=scope)
    
    # main reccurent layers
    l_hid = self.l[scope]
    for idx in xrange(conf.recurrent_length):
      if conf.model == "pixel_rnn":
        scope = 'LSTM%d' % idx
        self.l[scope] = l_hid = diagonal_bilstm(l_hid, conf, scope=scope)
      elif conf.model == "pixel_cnn":
        scope = 'CONV%d' % idx
        self.l[scope] = l_hid = conv2d(l_hid, 3, [1, 1], "B", scope=scope)
      else:
        raise ValueError("wrong type of model: %s" % (conf.model))
      logger.info("Building %s" % scope)

    # output reccurent layers
    for idx in xrange(conf.out_recurrent_length):
      scope = 'CONV_OUT%d' % idx
      self.l[scope] = l_hid = tf.nn.relu(conv2d(l_hid, conf.out_hidden_dims, [1, 1], "B", scope=scope))
      logger.info("Building %s" % scope)

    if channel == 1:
      self.l['conv2d_out_logits'] = conv2d(l_hid, 1, [1, 1], "B", scope='conv2d_out_logits')
      self.l['output'] = tf.nn.sigmoid(self.l['conv2d_out_logits'])

      logger.info("Building loss and optims")
      self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          self.l['conv2d_out_logits'], self.l['normalized_inputs'], name='loss'))
    else:
      raise ValueError("Implementation in progress for RGB colors")

      COLOR_DIM = 256

      self.l['conv2d_out_logits'] = conv2d(l_hid, COLOR_DIM, [1, 1], "B", scope='conv2d_out_logits')

      self.l['conv2d_out_logits_flat'] = tf.reshape(
          self.l['conv2d_out_logits'], [-1, self.height * self.width, COLOR_DIM])
      self.l['normalized_inputs_flat'] = tf.reshape(
          self.l['normalized_inputs'], [-1, self.height * self.width, COLOR_DIM])

      pred_pixels = [tf.squeeze(pixel, squeeze_dims=[1])
          for pixel in tf.split(1, self.height * self.width, self.l['conv2d_out_logits_flat'])]
      target_pixels = [tf.squeeze(pixel, squeeze_dims=[1])
          for pixel in tf.split(1, self.height * self.width, self.l['normalized_inputs_flat'])]

      softmaxed_pixels = [tf.nn.softmax(pixel) for pixel in pred_pixels]

      losses = [tf.nn.sampled_softmax_loss(
          pred_pixel, tf.zeros_like(pred_pixel), pred_pixel, target_pixel, 1, COLOR_DIM) \
              for pred_pixel, target_pixel in zip(pred_pixels, target_pixels)]

      self.l['output'] = tf.nn.softmax(self.l['conv2d_out_logits'])

      logger.info("Building loss and optims")
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          self.l['conv2d_out_logits'], self.l['normalized_inputs'], name='loss'))

    optimizer = tf.train.RMSPropOptimizer(conf.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)

    new_grads_and_vars = \
        [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]
    self.optim = optimizer.apply_gradients(new_grads_and_vars)
 
    show_all_variables()

    logger.info("Building %s finished!" % conf.model)

  def predict(self, images):
    return self.sess.run(self.l['output'], {self.l['inputs']: images})

  def test(self, images, with_update=False):
    if with_update:
      _, cost = self.sess.run([
          self.optim, self.loss,
        ], feed_dict={ self.l['inputs']: images })
    else:
      cost = self.sess.run(self.loss, feed_dict={ self.l['inputs']: images })
    return cost

  def generate(self):
    samples = np.zeros((100, self.height, self.width, 1), dtype='float32')

    for i in xrange(self.height):
      for j in xrange(self.width):
        for k in xrange(self.channel):
          next_sample = binarize(self.predict(samples))
          samples[:, i, j, k] = next_sample[:, i, j, k]

          if self.data == 'mnist':
            print "=" * (self.width/2), "(%2d, %2d)" % (i, j), "=" * (self.width/2)
            mprint(next_sample[0,:,:,:])

    return samples
