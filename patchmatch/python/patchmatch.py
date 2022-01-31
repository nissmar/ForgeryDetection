# [mask,param] = CMFD_PM(img,param)
#This code is the version 1.0 of the CMFD (Copy-Move Forgery Detection)
#   algorithm described in "Efficient dense-field copy-move forgery detection", 
#   written by  D. Cozzolino, G. Poggi and L. Verdoliva, 
#   IEEE Trans. on Information Forensics and Security, in press, 2015.
#   Please refer to this paper for a more detailed description of
#   the algorithm.
#
##########################################################################
# 
# Copyright (c) 2015 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
# 
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#
##########################################################################


import utils
import numpy as np
import os
from time import time
import matlab.engine
from skimage import morphology
import cv2


def cmfd_pm(img, param):

    outData = {}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print('START')

    # (1) Feature Extraction
    timestamp = time()

    # generation of filters
    outData['feat_name'] = param.type_feat
    if param.type_feat=='ZM-cart':
        bfdata = utils.ZM_bf(param.diameter, param.ZM_order)
    elif param.type_feat=='ZM-polar':
        bfdata = utils.ZMp_bf(param.diameter, param.ZM_order, param.radiusNum, param.anglesNum)
    elif param.type_feat=='PCT-cart':
        bfdata = utils.PCT_bf(param.diameter, param.PCT_NM)
    elif param.type_feat=='PCT-polar':
        bfdata = utils.PCTp_bf(param.diameter, param.PCT_NM, param.radiusNum, param.anglesNum)
    elif param.type_feat=='FMT':
        bfdata = utils.FMTpl_bf(param.diameter, param.FMT_M, param.radiusNum, param.anglesNum, param.FMT_N, param.radiusMin)
    else:
        raise ValueError('type of feature not found')

    # feature generation
    feat = np.abs(utils.bf_filter(img, bfdata))

    # cutting off the borders
    raggioU =  int(np.ceil((param.diameter - 1) / 2))
    raggioL = int(np.floor((param.diameter - 1) / 2))
    feat = feat[raggioU:(-1-raggioL), raggioU:(-1-raggioL), :]
    outData['timeFE'] = time() - timestamp
    print('time FE: {:.3f}'.format(outData['timeFE']))

    ## Matching
    timestamp = time()
    feat  = (feat - np.min(feat.reshape(-1,1))) / (np.max(feat) - np.min(feat)) # mPM requires the features to be in [0,1]

    # run matlab compiled file
    eng = matlab.engine.start_matlab()
    matlab_feat = eng.double(feat.tolist())
    cnn   = eng.vecnnmex_mod(matlab_feat, matlab_feat, 1, param.num_iter, -param.th_dist1, param.num_tile)
    eng.quit()

    mpf_y = cnn[:, :, 1, 0].astype(np.double)
    mpf_x = cnn[:, :, 0, 0].astype(np.double)
    outData['timeMP'] = time() - timestamp
    print('time PM: {:.3f}'.format(outData['timeMP']))
    outData['cnn'] = cnn

    ## Post Processing
    timestamp = time()
    # regularize offsets field by median filtering
    DD_med, NN_med = utils.genDisk(param.rd_median)
    NN_med = (NN_med + 1) / 2
    mpf_y, mpf_x = utils.MPFregularize(mpf_y,mpf_x,DD_med,NN_med)

    # Compute the squared error of dense linear fitting
    DLFerr  =  utils.DLFerror(mpf_y,mpf_x,param.rd_dlf)
    mask    = DLFerr <= param.th2_dlf
    outData['maskDLF'] =  mask
    
    # removal of close couples
    dist2 = utils.MPFspacedist2(mpf_y,mpf_x)
    mask  = np.logical_and(mask, (dist2>=param.th2_dist2))

    # morphological operations
    mask  = morphology.remove_small_objects(mask, param.th_sizeA, 8); 
    outData['maskMPF'] = mask
    mask  = utils.MPFdual(mpf_y, mpf_x, mask) # mirroring of detected regions
    mask  = morphology.remove_small_objects(mask, param.th_sizeB, 8)
    mask  = cv2.dilate(mask, morphology.disk(param.rd_dil))
    
    # put the borders
    mask = utils.padarray_both(mask,[raggioU,raggioU,raggioL,raggioL],0)  #utile? avt false() mais même effet que 0
    DLFerr = utils.padarray_both(DLFerr,[raggioU,raggioU,raggioL,raggioL],0)

    outData['timePP'] = time() - timestamp
    print('time PP: {:.3f}'.format(outData['timePP']))
    outData['cnn_end'] = np.concatenate([mpf_x,mpf_y], axis=2) # order in list?
    outData['DLFerr']  = DLFerr

    ## end
    print('END : {:.3f}'.format(outData['timeFE'] + outData['timeMP'] + outData['timePP']))

    return mask, param, outData