# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import time
import imageio
from matplotlib.pyplot import imsave
from tensorflow.python import pywrap_tensorflow
from model_nonlocal_fg import *
from dataloader_nonlocal import *
from average_gradients import *

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='test')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--test_out_file',             type=str,   help='path to the testing out file', default='')
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=1)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def get_tensors_in_checkpoint_file(checkpoint_path):
    variables_in_checkpoint = []
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        variables_in_checkpoint.append(key)
    #print('variables_in_checkpoint',variables_in_checkpoint)
    return variables_in_checkpoint


def get_variables_to_restore(variables):
    variables_to_restore = []
    variables_in_checkpoint=get_tensors_in_checkpoint_file(args.checkpoint_path)
    for v in variables:
        #print("v in variables",v)
        if v.name.split(':')[0] in variables_in_checkpoint:
            #print('v.name',v.name)
            variables_to_restore.append(v)
    #print("variables_to_restore",variables_to_restore)
    return variables_to_restore

def train(params):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)
        
        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdamOptimizer(learning_rate)


        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)

        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch

       
        # split for each gpu
        left_splits  = tf.split(left,  args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)
 
        tower_grads  = []
        tower_losses = []

        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):

                    model = MonodepthModel(params, args.mode, left_splits[i], right_splits[i], reuse_variables, i)
                    loss = model.total_loss
                    
                    tower_losses.append(loss)

                    reuse_variables = True
#########################################################################################################
                    train_vars=tf.trainable_variables()

                    #new_trainable_vars=[var for var in train_vars if (var.name.split('/')[2]!='conv_n1')and(var.name.split('/')[2]!='conv_n2')and(var.name.split('/')[2]!='conv_n3')and(var.name.split('/')[2]!='conv_n4')and(var.name.split('/')[2]!='edge_layer1')and(var.name.split('/')[2]!='edge_layer2')and(var.name.split('/')[2]!='edge_layer3')and(var.name.split('/')[2]!='edge_layer4')] # w/ edge
                    new_trainable_vars=[var for var in train_vars] # w/o edge

                    grads = opt_step.compute_gradients(loss, new_trainable_vars)
 
                    #tower_grads.append(grads)

        #grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

########################################################################################################################

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        #tf.summary.scalar('total_loss_edge', total_loss_edge, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options)
        #config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)


        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
     
            # Initilize all variables
            variables = tf.trainable_variables() #recent network
            #print('variables',variables)
            sess.run(tf.variables_initializer(variables, name='init'))
            # Get the trained variables
            variables_to_restore = get_variables_to_restore(variables) #previous network to restore
            #print('variables_to_restore',variables_to_restore)
            train_saver = tf.train.Saver(variables_to_restore)          
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])
            if args.retrain:
                sess.run(global_step.assign(0))

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        training_saver = tf.train.Saver()

        # GO!
        start_step = global_step.eval(session=sess)
        print('start_step',start_step)
        start_time = time.time()
        for step in range(start_step, start_step+num_total_steps):
            before_op_time = time.time()
  
            if (step == start_step) or (step % 100) == 0:
                save34_1 = np.zeros(shape=(5,32,64,128))
                save34_2 = np.zeros(shape=(5,32,64,128))
                save45_1 = np.zeros(shape=(5,16,32,256))
                save45_2 = np.zeros(shape=(5,16,32,256)) 
                save45_3 = np.zeros(shape=(5,16,32,256)) 
                save45_4 = np.zeros(shape=(5,16,32,256))
                
                _, loss_value,save34_tmp,save45_tmp = sess.run([apply_gradient_op, total_loss, model.save34, model.save45], 
                                                               feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                            model.pre_nonlocal_conv34_2:save34_2,
                                                                            model.pre_nonlocal_conv45_1:save45_1, 
                                                                            model.pre_nonlocal_conv45_2:save45_2,
                                                                            model.pre_nonlocal_conv45_3:save45_3, 
                                                                            model.pre_nonlocal_conv45_4:save45_4, model.flag:0})
                save34_1 = save34_tmp
                save45_1 = save45_tmp
                
                duration = time.time() - before_op_time
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / (step-start_step) - 1.0) * time_sofar

                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.4f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op, feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                            model.pre_nonlocal_conv34_2:save34_2,
                                                                            model.pre_nonlocal_conv45_1:save45_1, 
                                                                            model.pre_nonlocal_conv45_2:save45_2,
                                                                            model.pre_nonlocal_conv45_3:save45_3, 
                                                                            model.pre_nonlocal_conv45_4:save45_4, model.flag:0})
                summary_writer.add_summary(summary_str, global_step=step)
                if step and step % 10000 == 0:
                    training_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)
            
            elif((step % 100) == 1):
                _, loss_value,save34_tmp,save45_tmp = sess.run([apply_gradient_op, total_loss, model.save34, model.save45], 
                                                               feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                            model.pre_nonlocal_conv34_2:save34_2,
                                                                            model.pre_nonlocal_conv45_1:save45_1, 
                                                                            model.pre_nonlocal_conv45_2:save45_2,
                                                                            model.pre_nonlocal_conv45_3:save45_3, 
                                                                            model.pre_nonlocal_conv45_4:save45_4, model.flag:1})
                save34_2 = save34_1
                save45_2 = save45_1
                save34_1 = save34_tmp
                save45_1 = save45_tmp
                
            elif((step % 100) == 2):
                _, loss_value,save34_tmp,save45_tmp = sess.run([apply_gradient_op, total_loss, model.save34, model.save45], 
                                                               feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                            model.pre_nonlocal_conv34_2:save34_2,
                                                                            model.pre_nonlocal_conv45_1:save45_1, 
                                                                            model.pre_nonlocal_conv45_2:save45_2,
                                                                            model.pre_nonlocal_conv45_3:save45_3, 
                                                                            model.pre_nonlocal_conv45_4:save45_4, model.flag:2})

                save45_3 = save45_2
                save34_2 = save34_1
                save45_2 = save45_1
                save34_1 = save34_tmp
                save45_1 = save45_tmp
            elif((step % 100) == 3):
                _, loss_value,save34_tmp,save45_tmp = sess.run([apply_gradient_op, total_loss, model.save34, model.save45], 
                                                               feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                            model.pre_nonlocal_conv34_2:save34_2,
                                                                            model.pre_nonlocal_conv45_1:save45_1, 
                                                                            model.pre_nonlocal_conv45_2:save45_2,
                                                                            model.pre_nonlocal_conv45_3:save45_3, 
                                                                            model.pre_nonlocal_conv45_4:save45_4, model.flag:3})

                save45_4 = save45_3
                save45_3 = save45_2
                save34_2 = save34_1
                save45_2 = save45_1
                save34_1 = save34_tmp
                save45_1 = save45_tmp

            else:
                _, loss_value,save34_tmp,save45_tmp = sess.run([apply_gradient_op, total_loss, model.save34, model.save45], 
                                                               feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                            model.pre_nonlocal_conv34_2:save34_2,
                                                                            model.pre_nonlocal_conv45_1:save45_1, 
                                                                            model.pre_nonlocal_conv45_2:save45_2,
                                                                            model.pre_nonlocal_conv45_3:save45_3, 
                                                                            model.pre_nonlocal_conv45_4:save45_4, model.flag:4})
                
                save45_4 = save45_3
                save45_3 = save45_2
                save34_2 = save34_1
                save45_2 = save45_1
                save34_1 = save34_tmp
                save45_1 = save45_tmp
         
        training_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):
    """Test function."""

    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    # demo
    print ('1'+str(left))
    print ('2'+str(right))
    #
    model = MonodepthModel(params, args.mode, left, right)
    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)
    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    print(params.height, ' ', params.width)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    demo = np.zeros((params.height, 2*params.width,3), dtype=np.uint8)
    fps = np.zeros(num_test_samples, dtype=np.float32)
    for step in range(count_text_lines(args.filenames_file)):

        tic=time.clock()
        print('step',step)
        if (step == 0) or (step % 200 == 0):
            save34_1 = np.zeros(shape=(2,32,64,128))
            save34_2 = np.zeros(shape=(2,32,64,128)) 
            save45_1 = np.zeros(shape=(2,16,32,256))
            save45_2 = np.zeros(shape=(2,16,32,256)) 
            save45_3 = np.zeros(shape=(2,16,32,256)) 
            save45_4 = np.zeros(shape=(2,16,32,256))
                
            disp, save34_tmp,save45_tmp = sess.run([model.disp_left_est[0], model.save34, model.save45], 
                                                    feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                 model.pre_nonlocal_conv34_2:save34_2,
                                                                 model.pre_nonlocal_conv45_1:save45_1, 
                                                                 model.pre_nonlocal_conv45_2:save45_2,
                                                                 model.pre_nonlocal_conv45_3:save45_3, 
                                                                 model.pre_nonlocal_conv45_4:save45_4, model.flag:0})
            save34_1 = save34_tmp
            save45_1 = save45_tmp
            disparities[step] = disp[0].squeeze()
            disparities_pp[step] = post_process_disparity(disp.squeeze())
            fps[step] = 1/(time.clock() - tic)
            print('done in {} seconds!'.format(time.clock() - tic))

        elif (step == 1) or (step % 200 == 1):
            disp, save34_tmp,save45_tmp = sess.run([model.disp_left_est[0], model.save34, model.save45], 
                                                    feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                 model.pre_nonlocal_conv34_2:save34_2,
                                                                 model.pre_nonlocal_conv45_1:save45_1, 
                                                                 model.pre_nonlocal_conv45_2:save45_2,
                                                                 model.pre_nonlocal_conv45_3:save45_3, 
                                                                 model.pre_nonlocal_conv45_4:save45_4, model.flag:1})
            save34_2 = save34_1
            save45_2 = save45_1
            save34_1 = save34_tmp
            save45_1 = save45_tmp
            disparities[step] = disp[0].squeeze()
            disparities_pp[step] = post_process_disparity(disp.squeeze())
            fps[step] = 1/(time.clock() - tic)
            print('done in {} seconds!'.format(time.clock() - tic))

        elif (step == 2) or (step % 200 == 2):
            disp, save34_tmp,save45_tmp = sess.run([model.disp_left_est[0], model.save34, model.save45], 
                                                    feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                 model.pre_nonlocal_conv34_2:save34_2,
                                                                 model.pre_nonlocal_conv45_1:save45_1, 
                                                                 model.pre_nonlocal_conv45_2:save45_2,
                                                                 model.pre_nonlocal_conv45_3:save45_3, 
                                                                 model.pre_nonlocal_conv45_4:save45_4, model.flag:2})
            save45_3 = save45_2
            save34_2 = save34_1
            save45_2 = save45_1
            save34_1 = save34_tmp
            save45_1 = save45_tmp
            disparities[step] = disp[0].squeeze()
            disparities_pp[step] = post_process_disparity(disp.squeeze())
            fps[step] = 1/(time.clock() - tic)
            print('done in {} seconds!'.format(time.clock() - tic))

        elif (step == 3) or (step % 200 == 3):
            disp, save34_tmp,save45_tmp = sess.run([model.disp_left_est[0], model.save34, model.save45], 
                                                    feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                 model.pre_nonlocal_conv34_2:save34_2,
                                                                 model.pre_nonlocal_conv45_1:save45_1, 
                                                                 model.pre_nonlocal_conv45_2:save45_2,
                                                                 model.pre_nonlocal_conv45_3:save45_3, 
                                                                 model.pre_nonlocal_conv45_4:save45_4, model.flag:3})
            save45_4 = save45_3
            save45_3 = save45_2
            save34_2 = save34_1
            save45_2 = save45_1
            save34_1 = save34_tmp
            save45_1 = save45_tmp
            disparities[step] = disp[0].squeeze()
            disparities_pp[step] = post_process_disparity(disp.squeeze())
            fps[step] = 1/(time.clock() - tic)
            print('done in {} seconds!'.format(time.clock() - tic))

        else:
            disp, save34_tmp,save45_tmp = sess.run([model.disp_left_est[0], model.save34, model.save45], 
                                                   feed_dict = {model.pre_nonlocal_conv34_1:save34_1, 
                                                                model.pre_nonlocal_conv34_2:save34_2,
                                                                model.pre_nonlocal_conv45_1:save45_1, 
                                                                model.pre_nonlocal_conv45_2:save45_2,
                                                                model.pre_nonlocal_conv45_3:save45_3, 
                                                                model.pre_nonlocal_conv45_4:save45_4, model.flag:4})
            save45_4 = save45_3
            save45_3 = save45_2
            save34_2 = save34_1
            save45_2 = save45_1
            save34_1 = save34_tmp
            save45_1 = save45_tmp
            disparities[step] = disp[0].squeeze()
            disparities_pp[step] = post_process_disparity(disp.squeeze())
            fps[step] = 1/(time.clock() - tic)
            print('done in {} seconds!'.format(time.clock() - tic))
        #inimg = sess.run(left)
        
        #print(inimg.shape)
        #outimg = disparities_pp[step]*255
        #outim = (outimg/outimg.max())*200
        #img = outim.astype(np.uint8)
        #imgC = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        #demo[:,0:512,:]=(inimg[0,:,:,:]*255)[:,:,::-1]
        #demo[:,512:1025,:]=imgC
        #demoup=cv2.resize(demo,(1536,384),interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('disparity',demoup)
        #cv2.waitKey(10) & 0XFF


    print('done.')
    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.test_out_file)
        
    else:
        output_directory = args.output_directory
   
    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)
    
    matrix = np.load(str(output_directory)+'/disparities_pp.npy')
    for i in range(num_test_samples):
        image = matrix[i]  #matric is 3D load 1D
        #image = cv2.putText(image, 'FPS : '+str(fps[i]), (10, 240), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        path = os.path.join(args.test_out_file + str(i).zfill(5)+'.png')
        imsave(path, image)
  
    print('done.')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()
