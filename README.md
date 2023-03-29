# MDE-from-MediaCore
This is a refernce code in “Video-based Depth Estimation Autoencoder with Weighted Temporal Feature and Spatial Edge Guided Modules”. Our code is based on Unsupervised Monocular Depth Estimation with Left-Right Consistency (https://github.com/mrharicot/monodepth). If there is any enviorment issues, please refer to the original author's instructions for all related packages.

# Training procedure
We separate different folder to each training step, user can fallow different instruction bellow with suitable dataset. Before training, please make sure your training and testing image folder and run the "filename_to_txt.ipynb" file to build a training/testing list  
Step 1 : training the original UMDE network with stereo data  
Step 2 : training the network for transfer learning after adding DEG. (if you wnat to test on KITTI data set please use 2.mono_driving folder, otherwise, use 1.DEG for ScenFlow dataset)   
Step 3 : Adding TFB and CWB to do temporal data training, there are three kinds of TFB framework in STFB folder.

# General training description
main_xxx_xx.py # main code   
--mode train # training or testing mode
--model_name my_model # your saving model name  
--data_path /your path/ #your image folder   
--filenames_file /your path/kitti_train_files.txt #training data file generate by filename_to_txt.ipynb  
--log_directory /your path/  # save log  
--checkpoint_path /your path/model-xxxxx.meta  # restore training weights   

# General testing description
main_xxx_xx.py # main code   
--mode test # training or testing mode 
--data_path /your path/ #your image folder   
--filenames_file /your path/kitti_train_files.txt #training data file generate by filename_to_txt.ipynb  
--log_directory /your path/  # save log 
--checkpoint_path /your path/model-xxxxx.meta  # restore training weights   

