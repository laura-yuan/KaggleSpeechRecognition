import sys
import os
import csv
import label_wav_many_wav as lm
import argparse
import sys
import tensorflow as tf
FLAGS = None

def label_test(graph_path, label_path, test_data_path, submission_file_template_path, submission_output_path):
    # get the file name for the test data set.
    submission_field_name = ['fname', 'label']
    file_list = []
    with open(submission_file_template_path) as f:
        data = csv.DictReader(f, delimiter = ';')
        file_list = [row['fname'] for row in data]
    
    ## feed into the provided function label_wav
    input_name  = 'wav_data:0'
    output_name = 'labels_softmax:0'
    
    file_test = file_list
    wav_path_test = [os.path.join(test_data_path, file) for file in file_test ]
    print(label_path)
    print(graph_path)
    label_test = lm.label_wav(wav_path_test, label_path, graph_path, 'wav_data:0','labels_softmax:0',12)
    
    ## save the result in submission file.
    with open(submission_output_path, 'w',newline = '\n') as f:
        data_writer = csv.writer(f, delimiter = ',')
        data_writer.writerow(['fname', 'label'])
    for ii in range(len(file_test)):
        data_writer.writerow([file_test[ii], label_test[ii]])
def main(_):
    label_test(FLAGS.graph_path, FLAGS.label_path, FLAGS.test_data_path, FLAGS.submission_file_template_path, FLAGS.submission_output_path)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--graph_path',type = str, default = 'E:\Juyue\\tmp_conv_deep_norm\my_frozen_graph_conv.pb',
    help = 'Path to the file containing the frozen graph')
    parser.add_argument(
    '--label_path',type = str, default =  'E:\Juyue\kaggle_speech_dataset\conv_labels.txt',
    help = 'Path to file containing labels.'
    )
    parser.add_argument(
    '--submission_file_template_path', type = str, default = 'E:\Juyue\kaggle_speech_dataset\sample_submission.csv',
    help = 'Path to file containing sample submission.'
    )
    parser.add_argument(
    '--submission_output_path', type = str, default = 'E:\Juyue\kaggle_speech_dataset\submission.csv',
    help = 'Path to file storing labeled submission.'
    )
    parser.add_argument(
    '--test_data_path', type = str, default = 'E:\Juyue\kaggle_speech_dataset\\test\\audio',
    help = 'Path to test data'
    )   
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)