#! /usr/bin/env python

import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import models
import pdb
import os
import pickle
import flow_vis
import glob

class PreProcessor():
    def __init__(self, input_dir, output_dir):
        INPUT_DIR = "/home/ubuntu/kitti-3d-detection-unzipped/training/"
        OUTPUT_DIR = "/home/ubuntu/kitti-optical_flow/"
        
        # define default input and output directory
        if input_dir is None:
            self.input_dir = INPUT_DIR
        if output_dir is None:
            self.output_dir = OUTPUT_DIR
        else:
            self.input_dir = input_dir
            self.output_dir = output_dir
            
        self.prev_dir = os.path.join(self.input_dir, "prev_2/")
            
        # if input directory not found, return error.    
        if not os.path.isdir(self.input_dir):
            return FileNotFoundError("X incorrect input directory")
        
        # if output directory is not found, create a new directory
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        if not self.output_dir.endswith("/"):
            self.output_dir += "/"
            
    def pwc_fusion(self, im0_fn, im1_fn, im2_fn):
        pwc_model_fn = './pwc_net.pth.tar';

        im_all = [imread(img) for img in [im0_fn, im1_fn, im2_fn]]
        im_all = [im[:, :, :3] for im in im_all]

        # rescale the image size to be multiples of 64
        divisor = 64.
        H = im_all[0].shape[0]
        W = im_all[0].shape[1]

        H_ = int(ceil(H/divisor) * divisor)
        W_ = int(ceil(W/divisor) * divisor)
        for i in range(len(im_all)):
            im_all[i] = cv2.resize(im_all[i], (W_, H_))

        for _i, _inputs in enumerate(im_all):
            im_all[_i] = im_all[_i][:, :, ::-1]
            im_all[_i] = 1.0 * im_all[_i]/255.0

            im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
            im_all[_i] = torch.from_numpy(im_all[_i])
            im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
            im_all[_i] = im_all[_i].float()

        # compute two frame flows
        input_01 = [im_all[0].cuda(), im_all[1].cuda()]
        input_01_var = torch.autograd.Variable(torch.cat(input_01,1), volatile=True)

        input_12 = [im_all[1].cuda(), im_all[2].cuda()]
        input_12_var = torch.autograd.Variable(torch.cat(input_12,1), volatile=True)

        input_10 = [im_all[1].cuda(), im_all[0].cuda()]
        input_10_var = torch.autograd.Variable(torch.cat(input_10,1), volatile=True)


        net = models.pwc_dc_net(pwc_model_fn)
        net = net.cuda()
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

        cur_flow = net(input_12_var) * 20.0
        prev_flow = net(input_01_var) * 20.0
        prev_flow_back = net(input_10_var) * 20.0

        # perfom flow fusion
        net_fusion = models.netfusion_custom(path="/app/PWC-Net/Multi_Frame_Flow/fusion_net.pth.tar",
                                             div_flow=20.0, 
                                             batchNorm=False)
        net_fusion = net_fusion.cuda()
        net_fusion.eval()
        for p in net_fusion.parameters():
            p.requires_grad = False

        upsample_layer = torch.nn.Upsample(scale_factor=4, mode='bilinear')

        cur_flow = upsample_layer(cur_flow)
        prev_flow = upsample_layer(prev_flow)
        prev_flow_back = upsample_layer(prev_flow_back)
        input_var_cat = torch.cat((input_12_var, cur_flow, prev_flow, prev_flow_back), dim=1)
        flo = net_fusion(input_var_cat)

        flo = flo[0] * 20.0
        flo = flo.cpu().data.numpy()

        # scale the flow back to the input size 
        flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
        u_ = cv2.resize(flo[:,:,0],(W,H))
        v_ = cv2.resize(flo[:,:,1],(W,H))
        u_ *= W/ float(W_)
        v_ *= H/ float(H_)
        flo = np.dstack((u_,v_))
        return flo
    
    def pwc_net(self, im1_fn, im2_fn):
        pwc_model_fn = '/app/PWC-Net/Multi_Frame_Flow/pwc_net.pth.tar';

        im_all = [imread(img) for img in [im1_fn, im2_fn]]
        im_all = [im[:, :, :3] for im in im_all]

        # rescale the image size to be multiples of 64
        divisor = 64.
        H = im_all[0].shape[0]
        W = im_all[0].shape[1]

        H_ = int(ceil(H/divisor) * divisor)
        W_ = int(ceil(W/divisor) * divisor)
        for i in range(len(im_all)):
            im_all[i] = cv2.resize(im_all[i], (W_, H_))

        for _i, _inputs in enumerate(im_all):
            im_all[_i] = im_all[_i][:, :, ::-1]
            im_all[_i] = 1.0 * im_all[_i]/255.0

            im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
            im_all[_i] = torch.from_numpy(im_all[_i])
            im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
            im_all[_i] = im_all[_i].float()

        im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

        net = models.pwc_dc_net(pwc_model_fn)
        net = net.cuda()
        net.eval()

        flo = net(im_all)
        flo = flo[0] * 20.0
        flo = flo.cpu().data.numpy()

        # scale the flow back to the input size 
        flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
        u_ = cv2.resize(flo[:,:,0],(W,H))
        v_ = cv2.resize(flo[:,:,1],(W,H))
        u_ *= W/ float(W_)
        v_ *= H/ float(H_)
        flo = np.dstack((u_,v_))
        return flo
    
    def process_all_images(self):
        # get a list of png imgs
        curr_img_list = glob.glob(os.path.join(self.input_dir, "image_2/*.png"))
        curr_img_list = sorted(curr_img_list)
        log_fn = os.path.join(self.output_dir, 'get_optical_flow.log')
        for curr_img_path in curr_img_list:
            im2_fn = curr_img_path
            seq = curr_img_path.split("/")[-1].split(".")[0]  # get serial number of the img
            im1_fn = os.path.join(self.prev_dir, seq+'_01.png') # get path of current-1 image
            im0_fn = os.path.join(self.prev_dir, seq+'_02.png') # get path of current-2 image
            print("processing img #" + seq)
            import time
            start = time.time()
            of = self.pwc_fusion(im0_fn, im1_fn, im2_fn) # select either self.pwc_net(im1_fn, im2_fn) or self.pwc_fusion(im0_fn, im1_fn, im2_fn)
            of_fn = os.path.join(self.output_dir, seq+'_pwc_fusion.npy')
            np.save(of_fn, of)
            flow_color = flow_vis.flow_to_color(of, convert_to_bgr=True) # Apply the coloring (for OpenCV, set convert_to_bgr=True)
            flow_color_fn = os.path.join(self.output_dir, seq+'_pwc_fusion.png')
            cv2.imwrite(flow_color_fn, flow_color)
            end = time.time()
            print("completed in " + str(end-start) + "s")
            with open(log_fn, "a") as f:
                f.write("completed img #" + seq + " in " + str(end-start) + "s")

if len(sys.argv) > 1:
	input_dir = sys.argv[1]
if len(sys.argv) > 2:
    output_dir = sys.argv[2]
    
preprocessor = PreProcessor(input_dir, output_dir)
preprocessor.process_all_images()