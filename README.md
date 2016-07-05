Attention-based RNN model for Spoken Language Understanding (Intent Detection & Slot Filling)
==================

Tensorflow implementation of attention-based LSTM models for sequence classification and sequence labeling.

**Setup**

* Tensorflow r0.9 (https://www.tensorflow.org/versions/r0.9/get_started/index.html)

**Usage**:
```bash
data_dir=data/ATIS_samples
model_dir=model_tmp
max_sequence_length=50  # max length for train/valid/test sequence
task=joint  # available options: intent; tagging; joint
bidirectional_rnn=True  # available options: True; False

python run_multi-task_rnn.py --data_dir $data_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --task $task \
      --bidirectional_rnn $bidirectional_rnn
```

**Reference**

* Bing Liu, Ian Lane, "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling", (to appear) in Interspeech, 2016 (http://speech.sv.cmu.edu/publications/liu-interspeech-2016.pdf)


**Contact** 

Feel free to email liubing@cmu.edu for any pertinent questions/bugs regarding the code. 
