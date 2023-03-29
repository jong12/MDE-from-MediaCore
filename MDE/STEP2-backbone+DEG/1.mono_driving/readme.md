################## Training ################## 

#driving
python main_partial.py --mode train --model_name my_model --data_path /home/nung/data/driving/data/ --filenames_file /home/nung/gradient/filenames.txt --log_directory /home/nung/gradient/tmp50+30/ --checkpoint_path /home/nung/gradient/tmp_4layer50/my_model/model-55000


#driving_edge+kitti_unsupervised
python main_partial.py --mode train --model_name my_model --data_path /home/nung/data/kitti/ --filenames_file /home/nung/original/kitti_train_files.txt --log_directory /home/nung/mono_driving_kittti/tmp_driving_edge+kitti_unsupervised/ --checkpoint_path /home/nung/gradient/tmp_lr_4layer/my_model/model-86240




################## Testing ##################   

#driving
python main_partial.py --mode test --data_path /home/nung/data/driving/data/ --filenames_file /home/nung/gradient/test_filenames.txt --log_directory /home/nung/gradient/tmp_300/ --checkpoint_path /home/nung/gradient/tmp_300/my_model/model-88000

#kitti
python main_partial.py --mode test --data_path /home/nung/data/kitti/data_scene_flow/ --filenames_file /home/nung/monodepth-master/utils/filenames/kitti_stereo_2015_test_files.txt --test_out_file /home/nung/mono_driving_kittti/test_step3/ --log_directory /home/nung/mono_driving_kittti/tmp_kitti_step3/ --checkpoint_path /home/nung/mono_driving_kittti/tmp_kitti_step3/my_model/model-360000.meta

python main_partial.py --mode test --data_path /home/nung/data/kitti/data_scene_flow/ --filenames_file /home/nung/monodepth-master/utils/filenames/kitti_stereo_2015_test_files.txt --test_out_file /home/nung/mono_driving_kittti/test__driving_edge+kitti_unsupervised/ --log_directory /home/nung/mono_driving_kittti/tmp_driving_edge+kitti_unsupervised/ --checkpoint_path /home/nung/mono_driving_kittti/tmp_driving_edge+kitti_unsupervised/my_model/model-362500




#tensorboard
/*
tensorboard --inspect --logdir /home/nung/gradient/tmp/my_model
tensorboard --logdir /home/nung/gradient/tmp_nodisp/my_model
*/
