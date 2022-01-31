# This code is the demo of the CMFD (Copy-Move Forgery Detection)
#    algorithm described in "Efficient dense-field copy-move forgery detection", 
#    written by  D. Cozzolino, G. Poggi and L. Verdoliva, 
#    IEEE Trans. on Information Forensics and Security, in press, 2015.
#    Please refer to this paper for a more detailed description of
#    the algorithm.
# 
#  
#  Copyright (c) 2015 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#  All rights reserved.
#  This work should only be used for nonprofit purposes.
#  
#  By downloading and/or using any of these files, you implicitly agree to all the
#  terms of the license, as specified in the document LICENSE.txt
#  (included in this package) and online at
#  http://www.grip.unina.it/download/LICENSE_OPEN.txt
# 


import argparse
import numpy as np
import cv2
from patchmatch import cmfd_pm

parser = argparse.ArgumentParser()

# parameters Feature Extraction
parser.add_argument('--type_feat', type=str, default = 'ZM-cart',
                    help='type of feature among :\nZM-cart\nZM-polar\nPCT-cart\nPCT-polar\nFMT')
parser.add_argument('--data', type=str, default='../data/',
                    help='path to data')
parser.add_argument('--file', type=str, default='TP_C02_007_copy.png',
                    help='name of the file')

# parameters for Method
parser.add_argument('--ZM_order', type=int, default=5,
                    help='Zernike Moments order')
parser.add_argument('--PCT_NM', default=np.array([[0,0],[0,1],[0,2],[0,3], [1,0],[1,1],[1,2],[2,0],[2,1],[3,0]]),
                    help='')
parser.add_argument('--FMT_N', type=list, default=[-2, -1, 0, 1, 2],
                    help='')
parser.add_argument('--FMT_M', type=list, default=list(range(5)),
                    help='')
parser.add_argument('--radiusNum', type=int, default=26,
                    help='number of sampling points along the radius')
parser.add_argument('--anglesNum', type=int, default=32,
                    help='number of sampling points along the circumferences')
parser.add_argument('--radiusMin', type=float, default=np.sqrt(2),
                    help='minimun radius for FMT')

# parameters Matching
parser.add_argument('--num_iter', type=int, default=8,
                    help='N_{it} = number of iterations')
parser.add_argument('--th_dist1', type=int, default=8,
                    help='T_{D1} = minimum length of offsets')
parser.add_argument('--num_tile', type=int, default=1,
                    help='number of thread')

# parameters Post Processing
parser.add_argument('--th2_dist2', type=int, default=50*50,
                    help='T^2_{D2} = minimum diatance between clones')
parser.add_argument('--th2_dlf', type=int, default=300,
                    help='T^2_{\epsilon} = threshold on DLF error')
parser.add_argument('--th_sizeA', type=int, default=1200,
                    help='T_{S} = minimum size of clones')
parser.add_argument('--th_sizeB', type=int, default=1200,
                    help='T_{S} = minimum size of clones')
parser.add_argument('--rd_median', type=int, default=4,
                    help='\rho_M = radius of median filter')
parser.add_argument('--rd_dlf', type=int, default=6,
                    help='\rho_N = radius of DLF patch')

args = parser.parse_args()

# add some additionnal parameters
diameter_feat = {
    'ZM-cart' : 16,
    'ZM-polar' : 16,
    'PCT-cart' : 16,
    'PCT-polar' : 16,
    'FMT' : 24
    }
setattr(args, 'diameter', diameter_feat[args.type_feat])
setattr(args, 'rd_dil', args.rd_dlf + args.rd_median)


## load input
# some examples of forged images as shown in the experimental section of
# the paper
filename_img = 'TP_C02_007_copy.png'
filename_gt = 'TP_C02_007_gt.png'
#filename_img = 'TP_C01_011_copy_ln20.png'; filename_gt = 'TP_C01_011_gt_ln20.png';
#filename_img = 'TP_C01_039_copy_r45.png'; filename_gt = 'TP_C01_039_gt_r45.png';
#filename_img = 'TP_C02_011_copy_gj60.png'; filename_gt = 'TP_C02_011_gt_gj60.png';
#filename_img = 'TP_C02_019_copy_s1145.png'; filename_gt = 'TP_C02_019_gt_s1145.png';

img = cv2.imread(args.data + args.file)

## proposed techique
mask,param,data = cmfd_pm(img, args)

## show results
# mpf_y_pre = data.cnn(:,:,2)
# mpf_x_pre = data.cnn(:,:,1)
# mpf_y     = data.cnn_end(:,:,2)
# mpf_x     = data.cnn_end(:,:,1)

# figure();
# subplot(2,2,1);
# imshow(filename_img);
# title('forged image');
# subplot(2,2,2);
# imshow(double(repmat(mask,[1,1,3])));
# if sum(mask(:))>0,
#     title(sprintf('output by #s\n this image is forged', data.feat_name));
# else
#     title(sprintf('output by #s\n this image is pristine', data.feat_name));
# end;
# subplot(2,2,3);
# displayMPF(imread(filename_img),mpf_x,mpf_y,[24,24],data.maskMPF);
# title('selected offsets');
# if exist('filename_gt','var'),
#     maskGT = imread(filename_gt);
#     [FM,measure] = getFmeasure(mask,maskGT); disp(measure);
#     [col,map] = getFalseColoredResult(mask,maskGT);
#     
#     subplot(2,2,4);
#     imshow(col,map);
#     title(sprintf('result, FM = #5.3f', FM));
# end;

# Uncomment the following lines if you want to reproduce figure 4 of the
# original paper
# 
# figure();
# subplot(2,3,1);
# imshow(filename_img);
# title('forged image');
# subplot(2,3,2);
# dist2 = MPFspacedist2(mpf_y_pre,mpf_x_pre);
# max_dist = max(sqrt(dist2(:)));
# imshow(sqrt(dist2),[0, max_dist]); colormap(jet()); #colorbar(); 
# title('magnitude of offsets');
# subplot(2,3,3);
# dist2 = MPFspacedist2(mpf_y,mpf_x);
# imshow(sqrt(dist2),[0, max_dist]); colormap(jet()); #colorbar();
# title('median filtering');
# subplot(2,3,4);
# DLF_db = 10*log10(data.DLFerr); DLF_db(DLF_db<-50) = -50;
# imshow(DLF_db,[]); colormap(jet()); #colorbar();
# title('fitting error (db)');
# subplot(2,3,5);
# imshow(double(repmat(data.maskDLF,[1,1,3])));
# title('thresholding of fitting error');
# subplot(2,3,6);
# imshow(double(repmat(mask,[1,1,3])));
# title('final mask');
