# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2

from bilinear_sampler import *

monodepth_parameters = namedtuple('parameters', 
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary')

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, mode, left, right,reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right

        self.model_collection = ['model_' + str(model_index)]
        self.reuse_variables = reuse_variables
        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()     

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])
    def downsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h / ratio, w / ratio])
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

        return smoothness_x + smoothness_y

    def get_disp(self, x, scope):
        disp = 0.1 * self.conv(x, 2, 3, 1, scope=scope, activation_fn=tf.nn.sigmoid)
#        disp = 0.3 * self.conv(x, 2, 3, 1, scope=scope, activation_fn=tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, scope, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', scope=scope, activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size, scope):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1, scope+'_1')
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2, scope+'_2')
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale, scope):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1, scope)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale, scope):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME', scope=scope)
        return conv[:,3:-1,3:-1,:]

    def nonlocal_block(self, x, c, scope): #c=channel*1/2
        self.batchsize = 2
        self.flag = tf.placeholder(tf.int32,shape=())
        f0 = lambda:tf.constant(0)
        f1 = lambda:tf.constant(1)
        f2 = lambda:tf.constant(2)
        f3 = lambda:tf.constant(3)
        f4 = lambda:tf.constant(4)
        f5 = lambda:tf.constant(5)
        
        first = tf.case({tf.equal(self.flag, tf.constant(0)):f0, tf.equal(self.flag, tf.constant(1)):f1, 
                         tf.equal(self.flag, tf.constant(2)):f2, tf.equal(self.flag, tf.constant(3)):f3, 
                         tf.equal(self.flag, tf.constant(4)):f4},default=f5)

#        print('***********************x.shape',x.shape)
        nonlocal1 = tf.reshape(x, [self.batchsize, x.shape[1], x.shape[2], x.shape[3]]) #[B,H,W,channel]
#        print('***********************nonlocal1.shape',nonlocal1.shape)
        nonlocal_conv1 = slim.conv2d(nonlocal1, c, 1, 1, 'VALID', scope=scope+'_conv1') #1*1conv [B,H,W,c]
#        print('***********************nonlocal_conv1',nonlocal_conv1.shape)
        
        if scope =='nonlocal_3to4':
            self.pre_nonlocal_conv34_1 = tf.placeholder(tf.float32,shape=(self.batchsize, x.shape[1], x.shape[2], x.shape[3]))#1*1conv [B,H,W,c]
            self.pre_nonlocal_conv34_2 = tf.placeholder(tf.float32,shape=(self.batchsize, x.shape[1], x.shape[2], x.shape[3]))#1*1conv [B,H,W,c]
            
#            print('***********************self.pre_nonlocal_conv34_1',self.pre_nonlocal_conv34_1)
            nonlocal1_t_1 = self.pre_nonlocal_conv34_1 #[B,H,W,c]
            nonlocal1_t_2 = self.pre_nonlocal_conv34_2 #[B,H,W,c]
            
            if first == 0:
                nonlocal1_concat = tf.concat([nonlocal1, nonlocal1, nonlocal1], 2, name=scope+'_concatpre0') #concat [B,H,3W,2c]
            elif first == 1:
                nonlocal1_concat = tf.concat([nonlocal1_t_1, nonlocal1, nonlocal1], 2, name=scope+'_concatpre1') #concat [B,H,3W,2c]
            else:
                nonlocal1_concat = tf.concat([nonlocal1_t_2, nonlocal1_t_1, nonlocal1], 2, name=scope+'_concatpre2') #concat [B,H,3W,2c]
            nonlocal_conv2 = slim.conv2d(nonlocal1_concat, c, 1, 1, 'VALID', scope=scope+'_conv2') #1*1conv [B,H,3W,c]
#            print('***********************nonlocal_conv2',nonlocal_conv2.shape)
            nonlocal_conv3 = slim.conv2d(nonlocal1_concat, c, 1, 1, 'VALID', scope=scope+'_conv3') #1*1conv [B,H,3W,c]


            
        else:
            self.pre_nonlocal_conv45_1 = tf.placeholder(tf.float32,shape=(self.batchsize, x.shape[1], x.shape[2], x.shape[3]))#1*1conv [B,H,W,c]
            self.pre_nonlocal_conv45_2 = tf.placeholder(tf.float32,shape=(self.batchsize, x.shape[1], x.shape[2], x.shape[3]))#1*1conv [B,H,W,c]
            self.pre_nonlocal_conv45_3 = tf.placeholder(tf.float32,shape=(self.batchsize, x.shape[1], x.shape[2], x.shape[3]))#1*1conv [B,H,W,c]
            self.pre_nonlocal_conv45_4 = tf.placeholder(tf.float32,shape=(self.batchsize, x.shape[1], x.shape[2], x.shape[3]))#1*1conv [B,H,W,c]
            print('***********************self.pre_nonlocal_conv45_1',self.pre_nonlocal_conv45_1)
            nonlocal1_t_1 = self.pre_nonlocal_conv45_1 #[B,H,W,c]
            nonlocal1_t_2 = self.pre_nonlocal_conv45_2 #[B,H,W,c]
            nonlocal1_t_3 = self.pre_nonlocal_conv45_3 #[B,H,W,c]
            nonlocal1_t_4 = self.pre_nonlocal_conv45_4 #[B,H,W,c]           
            
            if first == 0:
                nonlocal1_concat = tf.concat([nonlocal1, nonlocal1, nonlocal1, nonlocal1, nonlocal1], 2, name=scope+'_concatpre0') #concat [B,H,5W,2c]
            elif first == 1:
                nonlocal1_concat = tf.concat([nonlocal1_t_1, nonlocal1_t_1, nonlocal1, nonlocal1, nonlocal1], 2, name=scope+'_concatpre1') #concat [B,H,5W,2c]
            elif first == 2:
                nonlocal1_concat = tf.concat([nonlocal1_t_2, nonlocal1_t_1, nonlocal1_t_1, nonlocal1, nonlocal1], 2, name=scope+'_concatpre2') #concat [B,H,5W,2c]
            elif first == 3:
                nonlocal1_concat = tf.concat([nonlocal1_t_3, nonlocal1_t_3, nonlocal1_t_2, nonlocal1_t_1, nonlocal1], 2, name=scope+'_concatpre3')#concat [B,H,5W,2c]
            else:
                nonlocal1_concat = tf.concat([nonlocal1_t_4, nonlocal1_t_3, nonlocal1_t_2, nonlocal1_t_1, nonlocal1], 2, name=scope+'_concatpre4') #concat [B,H,5W,2c]
#            print('***********************nonlocal1_concat',nonlocal1_concat.shape)

            nonlocal_conv2 = slim.conv2d(nonlocal1_concat, c, 1, 1, 'VALID', scope=scope+'_conv2') #1*1conv [B,H,5W,c]
#            print('***********************nonlocal_conv2',nonlocal_conv2.shape)
            nonlocal_conv3 = slim.conv2d(nonlocal1_concat, c, 1, 1, 'VALID', scope=scope+'_conv3') #1*1conv [B,H,5W,c]

  
 
        nonlocal_conv1_flat = tf.reshape(nonlocal_conv1, [nonlocal_conv1.shape[0], -1, nonlocal_conv1.shape[-1]]) #reshape [B,HW,c]
        nonlocal_conv2_flat = tf.reshape(nonlocal_conv2, [nonlocal_conv2.shape[0], -1, nonlocal_conv2.shape[-1]]) #reshape [B,H5W,c]
        nonlocal_conv3_flat = tf.reshape(nonlocal_conv3, [nonlocal_conv3.shape[0], -1, nonlocal_conv3.shape[-1]]) #reshape [B,H5W,c]
#        print('***********************nonlocal_conv1_flat',nonlocal_conv1_flat.shape)       
        dot1 = tf.matmul(nonlocal_conv1_flat, tf.transpose(nonlocal_conv2_flat, [0, 2, 1])) #mul [B,HW,c][c,H5W,B]=[B,HW,H5W]
#        print('***********************dot1',dot1.shape)
        dot2 = tf.matmul(tf.nn.softmax(dot1), nonlocal_conv3_flat) #mul [B,HW,5HW][B,5HW,c]=[B,HW,c]
#        print('***********************dot2',dot2.shape)
        reshape = tf.reshape(dot2,[self.batchsize, x.shape[1], x.shape[2], c]) #[B,H,W,c]
#        print('***********************reshape',reshape.shape)
        nonlocal_conv = slim.conv2d(reshape, x.shape[3], 1, 1, 'VALID', scope=scope+'_conv', weights_initializer=tf.zeros_initializer()) #1*1conv [B,H,W,channel]
##################################selective_kernel_network###############################################
        filterin = tf.concat([nonlocal_conv, nonlocal1], 3, name='filterin')  #[B,H,W,2channel]
        print('***********************filterin.shape',filterin.shape)
        gap = tf.reduce_mean(filterin, axis=[1,2]) #2channel
        print('***********************gap.shape',gap.shape) #[B,channel]      
        fc = slim.fully_connected(gap, int(gap.shape[1]), scope=scope+"fully_c") #[B,2channel]
#        fc1 = slim.fully_connected(gap, int(nonlocal_conv.shape[3]), activation_fn=tf.nn.sigmoid, scope=scope+"fully_c1")
#        fc2 = slim.fully_connected(gap, int(nonlocal_conv.shape[3]), activation_fn=tf.nn.sigmoid, scope=scope+"fully_c2")
        fc1 = fc[:,0:int(nonlocal_conv.shape[3])]
        fc2 = fc[:,int(nonlocal_conv.shape[3]):int(gap.shape[1])]
        concat_fc = tf.concat([tf.expand_dims(fc1, 2), tf.expand_dims(fc2, 2)], 2) #[B,channel,2]
        weighted_filter = tf.nn.softmax(concat_fc, dim=-1) #[B,channel,2]
        weighted_filter_expand = tf.expand_dims(tf.expand_dims(weighted_filter,1),1) #[B,1,1,channel,2]
        print('***********************weighted_filter_expand[:,:,:,:,0].shape',weighted_filter_expand[:,:,:,:,0].shape)
#        filter_now = tf.image.resize_nearest_neighbor(weighted_filter[:,:,0], [filterin[0], filterin[1], filterin[2]])  #[B,channel]
#        filter_previous =  weighted_filter[:,:,1]
        filter_now = tf.image.resize_nearest_neighbor(weighted_filter_expand[:,:,:,:,0], [filterin.shape[1], filterin.shape[2]])
        filter_pre = tf.image.resize_nearest_neighbor(weighted_filter_expand[:,:,:,:,1], [filterin.shape[1], filterin.shape[2]])
        print('***********************filter_previous.shape',filter_pre.shape)

        nonlocal_conv_pre = tf.multiply(nonlocal_conv, filter_pre)
        nonlocal_conv_now = tf.multiply(nonlocal1, filter_now)
        print('***********************nonlocal_conv_now.shape',nonlocal_conv_now.shape)
        nonlocal_out = nonlocal_conv_now+nonlocal_conv_pre

        if scope =='nonlocal_3to4':
            self.save34 = nonlocal_out
#            print('***********************self.save34',self.save34)
        else:
            self.save45 = nonlocal_out
        return nonlocal_out



    def build_vgg(self):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,  32, 7,scope='conv1') # H/2
            conv2 = self.conv_block(conv1,             64, 5,scope='conv2') # H/4
            conv3 = self.conv_block(conv2,            128, 3,scope='conv3') # H/8
################################################################################################
            nonlocal_3 = self.nonlocal_block(conv3,    64,scope='nonlocal_3to4') # H/8
            conv4 = self.conv_block(nonlocal_3,       256, 3,scope='conv4') # H/16
            nonlocal_4 = self.nonlocal_block(conv4,   128,scope='nonlocal_4to5') # H/16
            conv5 = self.conv_block(nonlocal_4,       512, 3,scope='conv5') # H/32
 ################################################################################################           
            conv6 = self.conv_block(conv5,            512, 3,scope='conv6') # H/64
            conv7 = self.conv_block(conv6,            512, 3,scope='conv7') # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7,  512, 3, 2,scope='upconv7') #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1,scope='iconv7')

            upconv6 = upconv(iconv7, 512, 3, 2,scope='upconv6') #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1,scope='iconv6')

            upconv5 = upconv(iconv6, 256, 3, 2,scope='upconv5') #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1,scope='iconv5')

            upconv4 = upconv(iconv5, 128, 3, 2,scope='upconv4') #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1,scope='iconv4')
            self.disp4 = self.get_disp(iconv4,scope='disp4')
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2,scope='upconv3') #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1,scope='iconv3')
            self.disp3 = self.get_disp(iconv3,scope='disp3')
            udisp3  = self.upsample_nn(self.disp3, 2)
#######################################################################################################################################
            conv_n1 = conv(iconv3,           64, 3, 1,scope='conv_n1')
            conv_n2 = conv(conv_n1,          64, 3, 1,scope='conv_n2')
            conv_n3 = conv(conv_n2,          64, 3, 1,scope='conv_n3')
            conv_n4 = conv(conv_n3,          64, 3, 1,scope='conv_n4')


            self.edge_layer1 = conv(conv_n1, 2, 1, 1, scope='edge_layer1')
            self.edge_layer2 = conv(conv_n2, 2, 1, 1, scope='edge_layer2')
            self.edge_layer3 = conv(conv_n3, 2, 1, 1, scope='edge_layer3')
#est_depth_edge=edge_layer4
            self.edge_layer4 = conv(conv_n4, 2, 1, 1, scope='edge_layer4')
            self.est_depth_edgel = tf.expand_dims(self.edge_layer4[:,:,:,0], 3)
            self.est_depth_edger = tf.expand_dims(self.edge_layer4[:,:,:,1], 3)

            iconv_plus = iconv3+conv_n3
            upconv2 = upconv(iconv_plus,  32, 3, 2,scope='upconv2') #H/2
#######################################################################################################################################
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1,scope='iconv2')
            self.disp2 = self.get_disp(iconv2,scope='disp2')
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2,scope='upconv1') #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1,scope='iconv1')
            self.disp1 = self.get_disp(iconv1,scope='disp1')

    def build_resnet50(self):
        #set convenience functions
        conv   = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2) # H/2  -   64D
            pool1 = self.maxpool(conv1,           3) # H/4  -   64D
            conv2 = self.resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = self.resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = self.resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = self.resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid  = self.scale_pyramid(self.left,  4)

                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left
                 
                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

        if self.mode == 'test':
            return

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness  = self.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1

            self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.params.alpha_image_loss * self.ssim_loss_left[i]  + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY

            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss
            
    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
                
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i], collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)

