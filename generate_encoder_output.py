# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:32:21 2016

@author: bliu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
#from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope

def generate_embedding_RNN_output(encoder_inputs, 
                                  cell,
                                  num_encoder_symbols,
                                  word_embedding_size,
                                  num_heads=1,
                                  dtype=dtypes.float32,
                                  scope=None, 
                                  initial_state_attention=False,
                                  sequence_length=None,
                                  bidirectional_rnn=False):
  """
  Generate RNN state outputs with word embeddings as inputs
      - Note that this example code does not include output label dependency modeling.
      One may add a loop function as in the rnn_decoder function in tf seq2seq.py 
      example to feed emitted label embedding back to RNN state.
  """
  with variable_scope.variable_scope(scope or "generate_embedding_RNN_output"):
    if bidirectional_rnn:
      encoder_cell_fw = cell
      encoder_cell_bw = cell
      embedding = variable_scope.get_variable("embedding", [num_encoder_symbols, word_embedding_size])
      encoder_embedded_inputs = list()
      encoder_embedded_inputs = [embedding_ops.embedding_lookup(embedding, encoder_input) for encoder_input in encoder_inputs]  
      encoder_outputs, encoder_state_fw, encoder_state_bw = rnn.bidirectional_rnn(
          encoder_cell_fw, encoder_cell_bw, encoder_embedded_inputs, sequence_length=sequence_length, dtype=dtype)
      encoder_state = array_ops.concat(1, [array_ops.concat(1, encoder_state_fw), array_ops.concat(1, encoder_state_bw)])
      top_states = [array_ops.reshape(e, [-1, 1, cell.output_size*2])
                    for e in encoder_outputs]
      attention_states = array_ops.concat(1, top_states)
    else:
      encoder_cell = cell
      embedding = variable_scope.get_variable("embedding", [num_encoder_symbols, word_embedding_size])
      encoder_embedded_inputs = list()
      encoder_embedded_inputs = [embedding_ops.embedding_lookup(embedding, encoder_input) for encoder_input in encoder_inputs]    
      encoder_outputs, encoder_state = rnn.rnn(
          encoder_cell, encoder_embedded_inputs, sequence_length=sequence_length, dtype=dtype)
      encoder_state = array_ops.concat(1, encoder_state)
      top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                    for e in encoder_outputs]
      attention_states = array_ops.concat(1, top_states)

    return encoder_outputs, encoder_state, attention_states
