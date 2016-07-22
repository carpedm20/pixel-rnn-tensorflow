import os
import numpy as np
import tensorflow as tf
from logging import getLogger

logger = getLogger(__name__)

class Statistic(object):
  def __init__(self, sess, data, model_dir, variables, test_step, max_to_keep=20):
    self.sess = sess
    self.test_step = test_step
    self.reset()

    with tf.variable_scope('t'):
      self.t_op = tf.Variable(0, trainable=False, name='t')
      self.t_add_op = self.t_op.assign_add(1)

    self.model_dir = model_dir
    self.saver = tf.train.Saver(variables + [self.t_op], max_to_keep=max_to_keep)
    self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['train_l', 'test_l']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary('%s/%s' % (data, tag), self.summary_placeholders[tag])

  def reset(self):
    pass

  def on_step(self, train_l, test_l):
    self.t = self.t_add_op.eval(session=self.sess)

    self.inject_summary({'train_l': train_l, 'test_l': test_l}, self.t)

    self.save_model(self.t)
    self.reset()

  def get_t(self):
    return self.t_op.eval(session=self.sess)

  def inject_summary(self, tag_dict, t):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, t)

  def save_model(self, t):
    logger.info("Saving checkpoints...")
    model_name = type(self).__name__

    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    self.saver.save(self.sess, self.model_dir, global_step=t)

  def load_model(self):
    logger.info("Initializing all variables")
    tf.initialize_all_variables().run()

    logger.info("Loading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(self.model_dir, ckpt_name)
      self.saver.restore(self.sess, fname)
      logger.info("Load SUCCESS: %s" % fname)
    else:
      logger.info("Load FAILED: %s" % self.model_dir)

    self.t = self.t_add_op.eval(session=self.sess)
