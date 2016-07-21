import datetime
import dateutil.tz
import numpy as np

def mprint(matrix, pivot=0.5):
  for array in matrix:
    print "".join("#" if i > pivot else " " for i in array)

def show_all_variables():
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    print "[%2d] %s %s = %s" % (idx, op.name, shape, count)
    total_count += int(count)
  print "[Total] variable size: %s" % "{:,}".format(total_count)

def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y_%m_%d_%H_%M_%S')

import os
import pprint
import tensorflow as tf

pp = pprint.PrettyPrinter().pprint

def get_model_dir(config, exceptions=None):
  attrs = config.__dict__['__flags']
  pp(attrs)

  keys = attrs.keys()
  keys.sort()
  keys.remove('data')
  keys = ['data'] + keys

  names =[]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      names.append("%s=%s" % (key, ",".join([str(i) for i in attrs[key]])
          if type(attrs[key]) == list else attrs[key]))
  return os.path.join('checkpoints', *names) + '/'

def preprocess_conf(conf):
  options = conf.__flags

  for option, value in options.items():
    option = option.lower()
