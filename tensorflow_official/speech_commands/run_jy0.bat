python train.py --data_url= --data_dir=E:\Juyue\speech_dataset --how_many_training_steps=500,400,300 --learning_rate=0.001,0.0005,0.0001 --train_dir=/tmp_0/speech_commands_train --summaries_dir=/tmp_0/retrain_logs --save_step_interval=50 --model_architecture=conv

python train.py --data_url= --data_dir=E:\Juyue\speech_dataset --how_many_training_steps=500,400,300 --learning_rate=0.001,0.0005,0.0001 --train_dir=/tmp_1/speech_commands_train --summaries_dir=/tmp_0/retrain_logs --save_step_interval=50 --model_architecture=low_latency_conv

python train.py --data_url= --data_dir=E:\Juyue\speech_dataset --how_many_training_steps=500,400,300 --learning_rate=0.001,0.0005,0.0001 --train_dir=/tmp_2/speech_commands_train --summaries_dir=/tmp_0/retrain_logs --save_step_interval=50 --model_architecture=low_latency_svdf
