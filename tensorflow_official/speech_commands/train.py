# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile
import pickle

FLAGS = None

class jy_summary:
    def __init__(self, max_step):
        self.accurarcy = np.zeros(max_step)
        self.entropy = np.zeros(max_step)
        self.confusion_matrix = [None for ii in range(max_step)]
        self.learning_rate = np.zeros(max_step)
        self.learning_rate_encoding = np.zeros(max_step)
        self.step = np.zeros(max_step)
        self.triplet_loss_hard = np.zeros(max_step)
        self.triplet_loss_easy = np.zeros(max_step)

    def update(self, which_step, accuracy_value, entropy_value, confusion_matrix_value, triplet_loss_hard_value, triplet_loss_easy_value, learning_rate_value, learning_rate_encoding_value):
        self.step = which_step
        self.accurarcy[which_step] = accuracy_value
        self.entropy[which_step] = entropy_value
        self.confusion_matrix[which_step] = confusion_matrix_value
        self.learning_rate[which_step] = learning_rate_value
        self.learning_rate_encoding[which_step] = learning_rate_encoding_value
        self.triplet_loss_easy[which_step] = triplet_loss_easy_value
        self.triplet_loss_hard[which_step] = triplet_loss_hard_value
    def save(self, directory):
        with open(directory, 'wb') as f:
            pickle.dump(self, f)
    def load(self, directory):
        with open(directory, 'rb') as f:
            last_point = pickle.load(f)
        # assign value from last point to current point.
        last_time_max_step = last_point.step
        self.step = last_time_max_step
        self.accurarcy[0:last_time_max_step] = last_point.accurarcy[0:last_time_max_step]
        self.entropy[0:last_time_max_step] = last_point.entropy[0:last_time_max_step]
        self.confusion_matrix[0:last_time_max_step] = last_point.confusion_matrix[0:last_time_max_step]
        self.learning_rate[0:last_time_max_step] = last_point.learning_rate[0:last_time_max_step]
        self.learning_rate_encoding[0:last_time_max_step] = last_point.learning_rate_encoding[0:last_time_max_step]
        self.triplet_loss_easy[0:last_time_max_step] = last_point.triplet_loss_easy[0:last_time_max_step]
        self.triplet_loss_hard[0:last_time_max_step] = last_point.triplet_loss_hard[0:last_time_max_step]

def main(_):
  # ****** Modified by Yi Hu ******
  yh_log = open(FLAGS.yihu_log, 'a')
  yh_log.write("********************************************************\n")
  yh_log.write(FLAGS.summaries_dir + '\n')
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, FLAGS.mfcc_normalization_flag, FLAGS.batch_normalization_flag)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)

  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

  # should you be smarter to have adaptive learning rate controlled by training accuracy and validation accuracy?
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  fingerprint_anchor = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_anchor')
  fingerprint_positive = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_positive')
  fingerprint_negative = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_negative')

  logits, loss_triplet, dropout_prob, is_training_flag, trainable_encoding_var_list = models.create_verification_model([fingerprint_anchor , fingerprint_positive, fingerprint_negative], model_settings)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(
      tf.int64, [None], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.

  # bactch normalization will use update_ops.
  control_dependencies = []
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.confusion_matrix(
      ground_truth_input, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # define learning rate
  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  learning_rate = tf.train.exponential_decay(FLAGS.starter_learning_rate, global_step,
                                             FLAGS.learning_rate_decay_steps, FLAGS.learning_rate_decay_rate, staircase=True)
  learning_rate_encoding = tf.train.exponential_decay(FLAGS.starter_learning_rate_encoding, global_step,
                                             FLAGS.learning_rate_decay_steps, FLAGS.learning_rate_decay_rate,
                                             staircase=True)

  # define three train step
  with tf.name_scope('train_conv'), tf.control_dependencies(update_ops):
    train_step_conv = tf.train.AdamOptimizer(
        learning_rate).minimize(cross_entropy_mean)

  with tf.name_scope('train_encoding_layer_only'):
    train_step_encoding_only = tf.train.AdamOptimizer(
        learning_rate_encoding).minimize(loss_triplet, var_list=trainable_encoding_var_list)

  with tf.name_scope('train_encoding'):
      train_step_encoding = tf.train.AdamOptimizer(
        learning_rate_encoding).minimize(loss_triplet)

  saver = tf.train.Saver(tf.global_variables())


  # result for training, validation and testing. write to tensorflow.
  tf.summary.scalar('cross_entropy', cross_entropy_mean)
  tf.summary.scalar('accuracy', evaluation_step)
  confusion_matrix_save = tf.expand_dims(tf.expand_dims(tf.cast(confusion_matrix, tf.float32), 0), 3)
  tf.summary.image('confusion_matrix', confusion_matrix_save)
  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
  test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

  # prepare jy_summary
  training_steps_max = np.sum(training_steps_list)
  train_summary_jy = jy_summary(training_steps_max)
  validation_summary_jy = jy_summary(training_steps_max)
  test_summary_jy = jy_summary(1)

  jy_summary_train_path = os.path.join(FLAGS.summaries_dir, FLAGS.model_architecture + '_train' + '.pickle')
  jy_summary_validation_path = os.path.join(FLAGS.summaries_dir, FLAGS.model_architecture + '_validation' + '.pickle')
  jy_summary_test_path = os.path.join(FLAGS.summaries_dir, FLAGS.model_architecture + '_test' + '.pickle')

  ## start running.
  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)
    train_summary_jy.load(jy_summary_train_path)
    validation_summary_jy.load(jy_summary_validation_path)

  tf.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))


  for training_step in xrange(start_step, training_steps_max + 1):
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth, _ = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)


    # Run the graph with this batch of training data.
    train_summary, train_accuracy, train_cross_entropy_value, train_confusion_matrix, predicted_value, learning_rate_value,_, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, confusion_matrix, predicted_indices, learning_rate,
            train_step_conv,increment_global_step
        ],
        feed_dict={
            fingerprint_anchor:   train_fingerprints,
            fingerprint_positive: train_fingerprints,
            fingerprint_negative: train_fingerprints,
            ground_truth_input:   train_ground_truth,
            dropout_prob: 0.5,
            is_training_flag: True
        })

    # Organize the audio samples to train the encoding.

    easy_triplets = input_data.verification_utils_prepare_triplet(train_ground_truth, label_count= model_settings['label_count'], hard_mode=False,
                                                                  num_of_triplets=FLAGS.batch_size)
    hard_triplets = input_data.verification_utils_prepare_triplet(train_ground_truth, label_count= model_settings['label_count'], hard_mode=True,
                                                                  num_of_triplets=FLAGS.batch_size,
                                                                  predicted_label=predicted_value)
    # train only the encoding layer.
    encoding_only_feed_dict = {fingerprint_anchor: train_fingerprints[easy_triplets[:, 0]],
                               fingerprint_positive: train_fingerprints[easy_triplets[:, 1]],
                               fingerprint_negative: train_fingerprints[easy_triplets[:, 2]],
                               dropout_prob: 0.5,
                               is_training_flag: True
                               }
    triplet_loss_easy_value, _,_ = sess.run([loss_triplet, learning_rate_encoding, train_step_encoding_only], encoding_only_feed_dict)
    # train the full network
    encoding_feed_dict = {fingerprint_anchor: train_fingerprints[hard_triplets[:, 0]],
                          fingerprint_positive: train_fingerprints[hard_triplets[:, 1]],
                          fingerprint_negative: train_fingerprints[hard_triplets[:, 2]],
                          dropout_prob: 0.5,
                          is_training_flag: True
                          }
    triplet_loss_hard_value, learning_rate_encoding_value, _ = sess.run([loss_triplet,learning_rate_encoding, train_step_encoding], encoding_feed_dict )

    # record the training process.
    train_summary_jy.update(training_step - 1, train_accuracy, train_cross_entropy_value, train_confusion_matrix,
                            triplet_loss_easy_value, triplet_loss_hard_value, learning_rate_value, learning_rate_encoding_value)
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f, triplet_loss %0.1f, lr %f, lr_en %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     train_cross_entropy_value, triplet_loss_easy_value, learning_rate_value, learning_rate_encoding_value))
    is_last_step = (training_step == training_steps_max)
    

    ## validation. inside of training loop
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_entropy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth, _ = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, validation_cross_entropy_value, validation_confusion_matrix = sess.run(
            [merged_summaries, evaluation_step, cross_entropy_mean,confusion_matrix],
            feed_dict={
                fingerprint_anchor: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_prob: 1.0,
                is_training_flag: False
            })
        # how could you use the model to do testing? test the similarity?

        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        total_entropy  += (validation_cross_entropy_value * batch_size)/set_size
        if total_conf_matrix is None:
          total_conf_matrix = validation_confusion_matrix
        else:
          total_conf_matrix += validation_confusion_matrix
      tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))

      validation_summary_jy.update(training_step - 1, total_accuracy, total_entropy, total_conf_matrix,
                              0,0,0,0)

      # Save the jy_summary as well.
      train_summary_jy.save(jy_summary_train_path)
      validation_summary_jy.save(jy_summary_validation_path)

      # ****** Modified by Yi Hu ******
      yh_log.write('**************** Validation **************** \n')
      yh_log.write('Confusion Matrix:\n %s \n' % (total_conf_matrix))
      yh_log.write(
          'Step %d: Validation accuracy = %.1f%% (N=%d) \n' % (training_step, total_accuracy * 100, set_size))



    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)


  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  total_entropy = 0
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)

    test_summary, test_accuracy, test_cross_entropy_value, test_confusion_matrix = sess.run(
        [merged_summaries, evaluation_step, cross_entropy_mean, confusion_matrix],
        feed_dict={
            fingerprint_anchor: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0,
            is_training_flag: False
        })
    test_writer.add_summary(test_summary, training_step)
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    total_entropy += (test_cross_entropy_value * batch_size) /set_size
    if total_conf_matrix is None:
      total_conf_matrix = test_confusion_matrix
    else:
      total_conf_matrix += test_confusion_matrix

  test_summary_jy.update(0, total_accuracy, total_entropy, total_conf_matrix,
                                 0,0,0,0)
  test_summary_jy.save(jy_summary_test_path)

  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))


  # ****** Modified by Yi Hu ******
  yh_log.write('**************** Test **************** \n')
  yh_log.write('Confusion Matrix:\n %s \n' % (total_conf_matrix))
  yh_log.write('Final test accuracy = %.1f%% (N=%d) \n' % (total_accuracy * 100, set_size))

  yh_log.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='1500,300',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      # default='yes,no,up,down,left,right,on,off,stop,go,zero,one,two,three,four,five,six,seven,eight,nine,bed,marvin,sheila,wow,bird,cat,dog,happy,house',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--mfcc_normalization_flag',
      type=bool,
      default=False,
      help='Whether to normalize the mfcc '
  )
  parser.add_argument(
      '--batch_normalization_flag',
      type=bool,
      default=False,
      help='Whether to do batch normalization '
  )
  parser.add_argument(
      '--similiarity_thresh_training',
      type=float,
      default = 0.5,
      help='threshold in similiarities for verification training'
  )
  parser.add_argument(
      '--encoding length',
      type=int,
      default=64,
      help='len of encoding vector'
  )

  parser.add_argument(
      '--starter_learning_rate',
      type = float,
      default= 0.001,
      help='start learning rate'
  )
  parser.add_argument(
      '--starter_learning_rate_encoding',
      type=float,
      default=0.0001,
      help='start learning rate for encoding layer'
  )
  parser.add_argument(
      '--learning_rate_decay_steps',
      type=int,
      default=150,
      help='every decay_steps, the learning rate is decreased decay_rate'
  )
  parser.add_argument(
      '--learning_rate_decay_rate',
      type=float,
      default=0.6,
      help='every decay_steps, the learning rate is decreased decay_rate'
  )

  # ****** Modified by Yi Hu ******
  parser.add_argument(
      '--yihu_log',
      type = str,
      default = 'E:\Juyue\\tmp\yihu_log.txt',
      help = 'Where to save the summary text logs')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
