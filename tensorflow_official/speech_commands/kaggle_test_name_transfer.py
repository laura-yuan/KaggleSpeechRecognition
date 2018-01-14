# first, load file
import csv
import evaluate_utils
file_original_path = r'E:\Juyue\tmp_conv_replicate_28_1\speech_commands_train\kaggle_test.csv'
file_changed_path = r'E:\Juyue\tmp_conv_replicate_28_1\speech_commands_train\kaggle_test_changed.csv'

fname = []
label = []
with open(csv, 'rb') as fopen:
    reader = csv.DictReader(fopen)

# second, change the item
# third, write it back