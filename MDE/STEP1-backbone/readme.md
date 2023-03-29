################ Training ################

#driving
python monodepth_main.py --mode train --encoder resnet50 --model_name my_model --data_path /home/nung/data/kitti/ --filenames_file /home/nung/data/filename/train/kitti_train_files.txt --log_directory /home/nung/original/tmp_1/ 
python monodepth_main.py --mode train --model_name my_model --data_path /home/nung/data/driving/data/ --filenames_file /home/nung/data/filename/train/train_driving_filenames_4320.txt --log_directory /home/nung/original/tmp_driving/

#KITTI
python monodepth_main.py --mode train --model_name my_model --data_path /home/nung/data/kitti/ --filenames_file /home/nung/original/kitti_train_files.txt --log_directory /home/nung/original/tmp_kitti/ --checkpoint_path /home/nung/original/tmp_driving/my_model/model-43600.meta

################ Testing-vgg ################  

#driving
python monodepth_main.py --mode test --data_path /home/nung/data/driving/data/ --filenames_file /home/nung/original/test_filenames.txt --log_directory /home/nung/original/tmp_driving/ --checkpoint_path /home/nung/original/tmp_driving/my_model/model-16500


#kitti
python monodepth_main.py --mode test --data_path /home/nung/data/kitti/data_scene_flow/ --filenames_file /home/nung/monodepth-master/utils/filenames/kitti_stereo_2015_test_files.txt --test_out_file /home/nung/original/test_evaldriving+kitti/ --log_directory  /home/nung/original/tmp_driving+kitti/ --checkpoint_path /home/nung/original/tmp_driving+kitti/my_model/model-224850

#kitti sequence_1096
python monodepth_main.py --mode test --data_path /home/nung/data/kitti/testtttt/color/20110930_0027/ --filenames_file /home/nung/data/filename/test/test_1095frame.txt --test_out_file /home/nung/original/test/1096/ --log_directory /home/nung/original/tmp_driving+kitti/ --checkpoint_path /home/nung/original/tmp_kitti/my_model/model-143000.meta

#kitti sequence_200
python monodepth_main.py --mode test --data_path /home/nung/data/kitti/testtttt/color/20110926_0084/ --filenames_file /home/nung/data/filename/test/test_200frame.txt --test_out_file /home/nung/original/test/200/ --log_directory /home/nung/original/tmp_driving+kitti/ --checkpoint_path /home/nung/original/tmp_kitti/my_model/model-143000.meta

#Poznan_Street
python monodepth_main.py --mode test --data_path /home/nung/data/Poznan_Street/Texture/ --filenames_file /home/nung/data/filename/test/Poznan_Street.txt --test_out_file /home/nung/original/test/Poznan/ --log_directory /home/nung/original/tmp_driving+kitti/ --checkpoint_path /home/nung/aaaMONO/original/tmp_kitti/my_model/model-143000.meta


#cityscape_munich_398
python monodepth_main.py --mode test --data_path /home/nung/data/cityscape_data/leftImg8bit_trainvaltest/leftImg8bit/test/munich/ --filenames_file /home/nung/data/filename/test/cityscape_munich_398.txt --test_out_file /home/nung/aaaMONO/original/test/cityscape_munich/ --log_directory /home/nung/aaaMONO/original/tmp_driving+kitti/ --checkpoint_path /home/nung/aaaMONO/original/tmp_driving+kitti/my_model/model-224850.meta



################ Testing-resnet ################

#kitti
python monodepth_main.py --mode test --encoder resnet50 --data_path /home/nung/data/kitti/data_scene_flow/ --filenames_file /home/nung/data/filename/test/kitti_stereo_2015_test_files.txt --test_out_file /home/nung/aaaMONO/original/test_resnet/kitti_stereo_2015_test/ --log_directory /home/nung/aaaMONO/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_kitti_resnet.meta


#kitti sequence_1096
python monodepth_main.py --mode test --encoder resnet50 --data_path /home/nung/data/kitti/testtttt/color/20110930_0027/ --filenames_file /home/nung/data/filename/test/test_1095frame.txt --test_out_file /home/nung/aaaMONO/original/test_resnet/1096/ --log_directory /home/nung/aaaMONO/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_kitti_resnet.meta

#kitti sequence_200
python monodepth_main.py --mode test --encoder resnet50 --data_path /home/nung/data/kitti/testtttt/color/20110926_0084/ --filenames_file /home/nung/data/filename/test/test_200frame.txt --test_out_file /home/nung/aaaMONO/original/test_resnet/200/ --log_directory /home/nung/aaaMONO/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_kitti_resnet.meta

#Poznan_Street
python monodepth_main.py --mode test --encoder resnet50 --data_path /home/nung/data/Poznan_Street/Texture/ --filenames_file /home/nung/data/filename/test/Poznan_Street.txt --test_out_file /home/nung/aaaMONO/original/test_resnet/Poznan/ --log_directory /home/nung/aaaMONO/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_kitti_resnet.meta



#cityscape_munich_398
python monodepth_main.py --mode test --encoder resnet50 --data_path /home/nung/data/cityscape_data/leftImg8bit_trainvaltest/leftImg8bit/test/munich/ --filenames_file /home/nung/data/filename/test/cityscape_munich_398.txt --test_out_file /home/nung/aaaMONO/original/test_resnet/cityscape_munich/ --log_directory /home/nung/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_kitti_resnet.meta




################ Testing-vgg with vgg pretrain ################  

#kitti sequence_1096
python monodepth_main.py --mode test --data_path /home/nung/data/kitti/testtttt/color/20110930_0027/ --filenames_file /home/nung/data/filename/test/test_1095frame.txt --test_out_file /home/nung/aaaMONO/original/test_ck/1095/ --log_directory /home/nung/aaaMONO/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_city2kitti.meta

#kitti sequence_200
python monodepth_main.py --mode test --data_path /home/nung/data/kitti/testtttt/color/20110926_0084/ --filenames_file /home/nung/data/filename/test/test_200frame.txt --test_out_file /home/nung/aaaMONO/original/test_ck/200/ --log_directory /home/nung/aaaMONO/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_city2kitti.meta

#Poznan_Street
python monodepth_main.py --mode test --data_path /home/nung/data/Poznan_Street/Texture/ --filenames_file /home/nung/data/filename/test/Poznan_Street.txt --test_out_file /home/nung/aaaMONO/original/test_ck/Poznan/ --log_directory /home/nung/aaaMONO/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_city2kitti.meta


#cityscape_munich_398
python monodepth_main.py --mode test --data_path /home/nung/data/cityscape_data/leftImg8bit_trainvaltest/leftImg8bit/test/munich/ --filenames_file /home/nung/data/filename/test/cityscape_munich_398.txt --test_out_file /home/nung/aaaMONO/original/test_ck/cityscape/ --log_directory /home/nung/aaaMONO/original/Gogard_pretrained_model/ --checkpoint_path /home/nung/aaaMONO/original/Gogard_pretrained_model/model_city2kitti.meta



################## Evaluate ##################
cd /home/nung/monodepth-master/utils/
python evaluate_kitti.py --split kitti --predicted_disp_path /home/nung/aaaMONO/original/test_resnet/kitti_stereo_2015_test/disparities.npy --gt_path /home/nung/data/kitti/data_scene_flow/



#tensorboard
/*
tensorboard --inspect --logdir /home/nung/gradient/tmp/my_model
tensorboard --logdir /home/nung/original/tmp_kitti
*/

  abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.1090,     1.0875,      5.558,      0.195,     28.260,      0.857,      0.950,      0.982

