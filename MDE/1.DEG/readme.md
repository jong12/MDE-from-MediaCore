################## Training ################## 

#driving
python monodepth_main_gradient.py --mode train --model_name my_model --data_path /home/nung/data/driving/data/ --filenames_file /home/nung/gradient/filenames.txt --log_directory /home/nung/gradient/tmp50+30/ --checkpoint_path /home/nung/gradient/tmp_4layer50/my_model/model-55000


python monodepth_main_gradient.py --mode train --model_name my_model --data_path /home/nung/data/driving/data/ --filenames_file /home/nung/data/filename/train/train_driving_filenames_4312_edge.txt --log_directory /home/nung/gradient/tmp_sigmoid30/



################## Testing ##################   

#driving
python monodepth_main_gradient.py --mode test --data_path /home/nung/data/driving/data/ --filenames_file /home/nung/data/filename/test/test_driving_filenames_88.txt --test_out_file /home/nung/gradient/test_whole_1/ --log_directory /home/nung/gradient/tmp_whole_1/ --checkpoint_path /home/nung/gradient/tmp_whole_1/my_model/model-86240.meta





#tensorboard
/*
tensorboard --inspect --logdir /home/nung/gradient/tmp_sobel/my_model
tensorboard --logdir /home/nung/aaaMONO/DEG/tmp_0_5/my_model/
*/
