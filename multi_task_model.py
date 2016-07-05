# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:28:22 2016

@author: Bing Liu (liubing@cmu.edu)

Multi-task RNN model with an attention mechanism.
    - Developped on top of the Tensorflow seq2seq_model.py example: 
      https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes


import data_utils
import seq_labeling
import seq_classification
import generate_encoder_output

class MultiTaskModel(object):
  def __init__(self, source_vocab_size, tag_vocab_size, label_vocab_size, buckets, 
               word_embedding_size, size, num_layers, max_gradient_norm, batch_size, 
               dropout_keep_prob=1.0, use_lstm=False, bidirectional_rnn=True,
               num_samples=1024, use_attention=False, 
               task=None, forward_only=False):
    self.source_vocab_size = source_vocab_size
    self.tag_vocab_size = tag_vocab_size
    self.label_vocab_size = label_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    softmax_loss_function = None

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
     
    if not forward_only and dropout_keep_prob < 1.0:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                           input_keep_prob=dropout_keep_prob,
                                           output_keep_prob=dropout_keep_prob)


    # Feeds for inputs.
    self.encoder_inputs = []
    self.tags = []    
    self.tag_weights = []    
    self.labels = []    
    self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
    
    for i in xrange(buckets[-1][0]):
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1]):
      self.tags.append(tf.placeholder(tf.float32, shape=[None], name="tag{0}".format(i)))
      self.tag_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))
    self.labels.append(tf.placeholder(tf.float32, shape=[None], name="label"))

    base_rnn_output = generate_encoder_output.generate_embedding_RNN_output(self.encoder_inputs,
                                                                            cell,
                                                                            self.source_vocab_size,
                                                                            word_embedding_size,
                                                                            dtype=dtypes.float32,
                                                                            scope=None,
                                                                            sequence_length=self.sequence_length,
                                                                            bidirectional_rnn=bidirectional_rnn)
    encoder_outputs, encoder_state, attention_states = base_rnn_output
    
    if task['tagging'] == 1:
      self.tagging_output, self.tagging_loss = seq_labeling.generate_sequence_output(
          self.source_vocab_size,       
          encoder_outputs, encoder_state, self.tags, self.sequence_length, self.tag_vocab_size, self.tag_weights,
          buckets, softmax_loss_function=softmax_loss_function, use_attention=use_attention)
    if task['intent'] == 1:
      self.classification_output, self.classification_loss = seq_classification.generate_single_output(
          encoder_state, attention_states, self.sequence_length, self.labels, self.label_vocab_size,   
          buckets, softmax_loss_function=softmax_loss_function, use_attention=use_attention)
    
    if task['tagging'] == 1:
      self.loss = self.tagging_loss
    elif task['intent'] == 1:
      self.loss = self.classification_loss

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      opt = tf.train.AdamOptimizer()
      if task['joint'] == 1:
        # backpropagate the intent and tagging loss, one may further adjust 
        # the weights for the two costs.
        gradients = tf.gradients([self.tagging_loss, self.classification_loss], params)
      elif task['tagging'] == 1:
        gradients = tf.gradients(self.tagging_loss, params)
      elif task['intent'] == 1:
        gradients = tf.gradients(self.classification_loss, params)
        
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                       max_gradient_norm)
      self.gradient_norm = norm
      self.update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables())


  def joint_step(self, session, encoder_inputs, tags, tag_weights, labels, batch_sequence_length,
           bucket_id, forward_only):
    """Run a step of the joint model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      tags: list of numpy int vectors to feed as decoder inputs.
      tag_weights: list of numpy float vectors to feed as tag weights.
      labels: list of numpy int vectors to feed as sequence class labels.
      bucket_id: which bucket of the model to use.
      batch_sequence_length: batch_sequence_length
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, output tags, and output class label.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, tag_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(tags) != tag_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(tags), tag_size))
    if len(labels) != 1:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(labels), 1))

    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.tags[l].name] = tags[l]
      input_feed[self.tag_weights[l].name] = tag_weights[l]
    input_feed[self.labels[0].name] = labels[0]

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.loss] # Loss for this batch.
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
      output_feed.append(self.classification_output[0])
    else:
      output_feed = [self.loss]
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
      output_feed.append(self.classification_output[0])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3:3+tag_size], outputs[-1]
    else:
      return None, outputs[0], outputs[1:1+tag_size], outputs[-1]


  def tagging_step(self, session, encoder_inputs, tags, tag_weights, batch_sequence_length,
           bucket_id, forward_only):
    """Run a step of the tagging model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      tags: list of numpy int vectors to feed as decoder inputs.
      tag_weights: list of numpy float vectors to feed as target weights.
      batch_sequence_length: batch_sequence_length
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the output tags.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, tag_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(tags) != tag_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(tags), tag_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.tags[l].name] = tags[l]
      input_feed[self.tag_weights[l].name] = tag_weights[l]

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.loss] # Loss for this batch.
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
    else:
      output_feed = [self.loss]
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3:3+tag_size]
    else:
      return None, outputs[0], outputs[1:1+tag_size]

  def classification_step(self, session, encoder_inputs, labels, batch_sequence_length,
           bucket_id, forward_only):
    """Run a step of the intent classification model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      labels: list of numpy int vectors to feed as sequence class labels.
      batch_sequence_length: batch_sequence_length
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the output class label.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, target_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    input_feed[self.labels[0].name] = labels[0]

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.loss,    # Loss for this batch.
                     self.classification_output[0]]
    else:
      output_feed = [self.loss,
                     self.classification_output[0],]

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3]  # Gradient norm, loss, outputs.
    else:
      return None, outputs[0], outputs[1] # No gradient norm, loss, outputs.


  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs, labels = [], [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_sequence_length_list= list()
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input, label = random.choice(data[bucket_id])
      batch_sequence_length_list.append(len(encoder_input))

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      #encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
      encoder_inputs.append(list(encoder_input + encoder_pad))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input)
      decoder_inputs.append(decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
      labels.append(label)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels = [], [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))                        
      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
#        if length_idx < decoder_size - 1:
#          target = decoder_inputs[batch_idx][length_idx + 1]
#        print (length_idx)
        if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    
    batch_labels.append(
      np.array([labels[batch_idx][0]
                for batch_idx in xrange(self.batch_size)], dtype=np.int32))
                        
    batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_sequence_length, batch_labels


  def get_one(self, data, bucket_id, sample_id):
    """Get a single sample data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs, labels = [], [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_sequence_length_list= list()
    #for _ in xrange(self.batch_size):
    encoder_input, decoder_input, label = data[bucket_id][sample_id]
    batch_sequence_length_list.append(len(encoder_input))

      # Encoder inputs are padded and then reversed.
    encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      #encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
    encoder_inputs.append(list(encoder_input + encoder_pad))

    # Decoder inputs get an extra "GO" symbol, and are padded then.
    decoder_pad_size = decoder_size - len(decoder_input)
    decoder_inputs.append(decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
    labels.append(label)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels = [], [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(1)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(1)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(1, dtype=np.float32)
      for batch_idx in xrange(1):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
#        if length_idx < decoder_size - 1:
#          target = decoder_inputs[batch_idx][length_idx + 1]
#        print (length_idx)
        if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
      
    batch_labels.append(
      np.array([labels[batch_idx][0]
                for batch_idx in xrange(1)], dtype=np.int32))
                    
    batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_sequence_length, batch_labels