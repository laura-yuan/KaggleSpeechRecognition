python train.py --data_url= --data_dir=E:\Juyue\speech_dataset --how_many_training_steps=3000,2000,1000 --learning_rate=0.001,0.0005,0.0001 --train_dir=E:/Juyue/tmp_conv_deep/speech_commands_train --summaries_dir=E:/Juyue/tmp_conv_deep/retrain_logs --save_step_interval=500 --model_architecture=conv_deep --eval_step_interval=100