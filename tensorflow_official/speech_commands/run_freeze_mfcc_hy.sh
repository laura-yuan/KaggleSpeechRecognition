#!/bin/sh

for MY_DCT in 13 26 32
do
  for MY_WINDOW in 20 30 45 60
  do
     MY_STRIDE=10
     while [ $MY_STRIDE -lt $MY_WINDOW ]
     do
python freeze.py \
--start_checkpoint=/home/yihu/DB/KAGGLE/Tensorflow_Speech/results/tmp_mfcc_${MY_DCT}_${MY_WINDOW}_${MY_STRIDE}/speech_commands_train/conv.ckpt-1800 \
--output_file=/home/yihu/DB/KAGGLE/Tensorflow_Speech/results/tmp_mfcc_${MY_DCT}_${MY_WINDOW}_${MY_STRIDE}/my_frozen_graph.pb \
--dct_coefficient_count=$MY_DCT \
--window_size_ms=$MY_WINDOW \
--window_stride_ms=$MY_STRIDE \


       MY_STRIDE=`expr $MY_STRIDE + 10`
     done
  done
done
