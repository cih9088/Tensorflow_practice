'''
From https://github.com/tsc2017/inception-score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py
Modified by Andy
Args:
    images: A numpy array with values ranging from -1 to 1 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary.
    splits: The number of splits of the images, default is 10.
Returns:
    mean and standard deviation of the inception across the splits.
'''

import tensorflow as tf
import os, sys
import functools
import numpy as np
import math
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan

BATCH_SIZE=64
logits = None
placeholder = None

# Run images through Inception.
def inception_logits(images, num_splits=1):
    images=tf.transpose(images,[0,2,3,1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(
    images, num_or_size_splits=num_splits)
    logits = functional_ops.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='logits:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

def get_inception_probs(inps, placeholder):
    preds = []
    n_batches = len(inps)//BATCH_SIZE
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        pred = logits.eval({placeholder:inp})[:,:1000]
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    preds=np.exp(preds)/np.sum(np.exp(preds),1,keepdims=True)
    return preds

def preds2score(preds,splits):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, splits=10, session=None):
    global logits
    global placeholder
    assert(type(images) == np.ndarray)
    assert(len(images.shape)==4)
    assert(images.shape[1]==3)
    assert(np.max(images[0])<=1)
    assert(np.min(images[0])>=-1)

    if session is None:
        session = tf.InteractiveSession()
    if placeholder is None:
        placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None], name='inception_input')
    if logits is None:
        logits = inception_logits(placeholder)

    start_time=time.time()
    preds=get_inception_probs(images, placeholder)
    #  print('Inception Score for %i samples in %i splits'% (preds.shape[0],splits))
    mean,std = preds2score(preds,splits)
    #  print('Inception Score calculation time: %f s'%(time.time()-start_time))
    return mean,std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits (default).

#  session=tf.InteractiveSession()
#
#  BATCH_SIZE=64
#
#  # Run images through Inception.
#  inception_images=tf.placeholder(tf.float32,[BATCH_SIZE,3,None,None])
#  def inception_logits(images=inception_images, num_splits=1):
#      images=tf.transpose(images,[0,2,3,1])
#      size = 299
#      images = tf.image.resize_bilinear(images, [size, size])
#      generated_images_list = array_ops.split(
#      images, num_or_size_splits=num_splits)
#      logits = functional_ops.map_fn(
#          fn=functools.partial(tfgan.eval.run_inception, output_tensor='logits:0'),
#          elems=array_ops.stack(generated_images_list),
#          parallel_iterations=1,
#          back_prop=False,
#          swap_memory=True,
#          name='RunClassifier')
#      logits = array_ops.concat(array_ops.unstack(logits), 0)
#      return logits
#
#  logits=inception_logits()
#
#  def get_inception_probs(inps):
#      preds = []
#      n_batches = len(inps)//BATCH_SIZE
#      for i in range(n_batches):
#          inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
#          pred = logits.eval({inception_images:inp})[:,:1000]
#          preds.append(pred)
#      preds = np.concatenate(preds, 0)
#      preds=np.exp(preds)/np.sum(np.exp(preds),1,keepdims=True)
#      return preds
#
#  def preds2score(preds,splits):
#      scores = []
#      for i in range(splits):
#          part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
#          kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
#          kl = np.mean(np.sum(kl, 1))
#          scores.append(np.exp(kl))
#      return np.mean(scores), np.std(scores)
#
#  def get_inception_score(images, splits=10):
#      assert(type(images) == np.ndarray)
#      assert(len(images.shape)==4)
#      assert(images.shape[1]==3)
#      assert(np.max(images[0])<=1)
#      assert(np.min(images[0])>=-1)
#
#      start_time=time.time()
#      preds=get_inception_probs(images)
#      #  print('Inception Score for %i samples in %i splits'% (preds.shape[0],splits))
#      mean,std = preds2score(preds,splits)
#      #  print('Inception Score calculation time: %f s'%(time.time()-start_time))
#      return mean,std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits (default).


###################################################################################################################
######################################## Inception score calculation ##############################################
###################################################################################################################

# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
#  from __future__ import absolute_import
#  from __future__ import division
#  from __future__ import print_function
#
#  import os.path
#  import sys
#  import tarfile
#
#  import numpy as np
#  from six.moves import urllib
#  import tensorflow as tf
#  import glob
#  import scipy.misc
#  import math
#  import sys
#
#  MODEL_DIR = '/tmp/imagenet'
#  DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
#  softmax = None
#
#  # Call this function with list of images. Each of elements should be a
#  # numpy array with values ranging from 0 to 255.
#  def get_inception_score(images, splits=10):
#    assert(type(images) == list)
#    assert(type(images[0]) == np.ndarray)
#    assert(len(images[0].shape) == 3)
#    assert(np.max(images[0]) > 10)
#    assert(np.min(images[0]) >= 0.0)
#    inps = []
#    for img in images:
#      img = img.astype(np.float32)
#      inps.append(np.expand_dims(img, 0))
#    bs = 100
#    with tf.Session() as sess:
#      preds = []
#      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
#      for i in range(n_batches):
#          sys.stdout.write(".")
#          sys.stdout.flush()
#          inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
#          inp = np.concatenate(inp, 0)
#          pred = sess.run(softmax, {'ExpandDims:0': inp})
#          preds.append(pred)
#      preds = np.concatenate(preds, 0)
#      scores = []
#      for i in range(splits):
#        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
#        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
#        kl = np.mean(np.sum(kl, 1))
#        scores.append(np.exp(kl))
#      return np.mean(scores), np.std(scores)
#
#  # This function is called automatically.
#  def _init_inception():
#    global softmax
#    if not os.path.exists(MODEL_DIR):
#      os.makedirs(MODEL_DIR)
#    filename = DATA_URL.split('/')[-1]
#    filepath = os.path.join(MODEL_DIR, filename)
#    if not os.path.exists(filepath):
#      def _progress(count, block_size, total_size):
#        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#            filename, float(count * block_size) / float(total_size) * 100.0))
#        sys.stdout.flush()
#      filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#      print()
#      statinfo = os.stat(filepath)
#      print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
#    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
#    with tf.gfile.FastGFile(os.path.join(
#        MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
#      graph_def = tf.GraphDef()
#      graph_def.ParseFromString(f.read())
#      _ = tf.import_graph_def(graph_def, name='')
#    # Works with an arbitrary minibatch size.
#    with tf.Session() as sess:
#      pool3 = sess.graph.get_tensor_by_name('pool_3:0')
#      ops = pool3.graph.get_operations()
#      for op_idx, op in enumerate(ops):
#          for o in op.outputs:
#              shape = o.get_shape()
#              shape = [s.value for s in shape]
#              new_shape = []
#              for j, s in enumerate(shape):
#                  if s == 1 and j == 0:
#                      new_shape.append(None)
#                  else:
#                      new_shape.append(s)
#              #  o.set_shape(tf.TensorShape(new_shape))
#              o._shape = tf.Tensor
#      w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
#      logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
#      softmax = tf.nn.softmax(logits)
#
#  if softmax is None:
#    _init_inception()
