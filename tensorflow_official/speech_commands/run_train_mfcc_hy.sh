#!/bin/sh

for MY_DCT in 13 26 32
do
  for MY_WINDOW in 20 30 45 60
  do
    MY_STRIDE=10
    while [ $MY_STRIDE -lt $MY_WINDOW ]
    do
python train.py \
--data_dir=/home/yihu/DB/KAGGLE/Tensorflow_Speech/speech_dataset \
--how_many_training_steps=800,500,500 \
--learning_rate=0.001,0.0005,0.0001 \
--batch_size=100 \
--dct_coefficient_count=$MY_DCT \
--window_size_ms=$MY_WINDOW \
--window_stride_ms=$MY_STRIDE \
--train_dir=/home/yihu/DB/KAGGLE/Tensorflow_Speech/results/tmp_mfcc_${MY_DCT}_${MY_WINDOW}_${MY_STRIDE}/speech_commands_train \
--summaries_dir=/home/yihu/DB/KAGGLE/Tensorflow_Speech/results/tmp_mfcc_${MY_DCT}_${MY_WINDOW}_${MY_STRIDE}/retrain_logs \
--save_step_interval=50 \
--model_architecture=conv \
--yihu_log=/home/yihu/DB/KAGGLE/Tensorflow_Speech/results/yihu_mfcc_log.txt

       MY_STRIDE=`expr $MY_STRIDE + 10`
    done
  done
done
