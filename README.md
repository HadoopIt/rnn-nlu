Attention-based RNN model for Spoken Language Understanding (Intent Detection & Slot Filling)
==================
Tensorflow implementation of attention-based LSTM models for sequence classification and sequence labeling.

**Updates - 2017/07/29**
* Updated code to work with the latest TensorFlow API: r1.2
* Code cleanup and formatting
* Note that this published code does not include the modeling of output label dependencies. One may add a loop function as in the rnn_decoder function in TensorFlow <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py#L292" target="_blank">seq2seq.py</a> example to feed emitted label embedding back to RNN state. Alternatively, sequence level optimization can be performed by adding a <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf" target="_blank">CRF</a> layer on top of the RNN outputs.
* The dataset used in the paper can be found at: https://github.com/yvchen/JointSLU/tree/master/data. We used the training set in the original ATIS train/test split, which has 4978 training samples.

**Setup**

* TensorFlow, version r1.2 (https://www.tensorflow.org/api_docs/)

**Usage**:
```bash
data_dir=data/ATIS_samples
model_dir=model_tmp
max_sequence_length=50  # max length for train/valid/test sequence
task=joint  # available options: intent; tagging; joint
bidirectional_rnn=True  # available options: True; False
use_attention=True # available options: True; False

python run_multi-task_rnn.py --data_dir $data_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --task $task \
      --bidirectional_rnn $bidirectional_rnn \
      --use_attention $use_attention
```

**Reference**

* Bing Liu, Ian Lane, "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling", Interspeech, 2016 (<a href="http://www.isca-speech.org/archive/Interspeech_2016/pdfs/1352.PDF" target="_blank">PDF</a>)

```
@inproceedings{Liu+2016,
author={Bing Liu and Ian Lane},
title={Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling},
year=2016,
booktitle={Interspeech 2016},
doi={10.21437/Interspeech.2016-1352},
url={http://dx.doi.org/10.21437/Interspeech.2016-1352},
pages={685--689}
}
```

**Contact** 

Feel free to email liubing@cmu.edu for any pertinent questions/bugs regarding the code. 
