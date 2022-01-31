import os
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle

noise_values = [.02, .04, .06, .08, .1]
angle_values = [4, 10, 20, 30, 60, 90, 180]
scale_values = [.5, .8, .91, 1.05, 1.09, 1.2, 2]
jpeg_values = [90, 80, 70, 60, 50, 40, 30, 20]

os.chdir('patchmatch/matlab')
eng = matlab.engine.start_matlab()

print('Computing noise fscores')
fscores_noise = np.zeros((len(noise_values), 5, 80))
filenames = pickle.load(open('../../CMFDdb_grip/listdirs/listdir_noise','rb'))

run_cmd = "main({}, '../../CMFDdb_grip/noise/{}', '../../CMFDdb_grip/noise/{}', false)"
for i in tqdm(range(len(noise_values)*80)):
    [forged, gt] = filenames[2*i:2*i+2]
    for j in range(5):
        fscores_noise[i//5, j, i%5] = eng.eval(run_cmd.format(j+1,forged,gt))
pickle.dump(fscores_noise, open('../../fscores/noise','rb'))

#############################################################################

print('Computing rotate fscores')
fscores_rotate = np.zeros((len(angle_values), 5, 80))
filenames = pickle.load(open('../../CMFDdb_grip/listdirs/listdir_rotate','rb'))

run_cmd = "main({}, '../../CMFDdb_grip/rotate/{}', '../../CMFDdb_grip/rotate/{}', false)"
for i in tqdm(range(len(angle_values)*80)):
    [forged, gt] = filenames[2*i:2*i+2]
    for j in range(5):
        fscores_rotate[i//5, j, i%5] = eng.eval(run_cmd.format(j+1,forged,gt))
pickle.dump(fscores_rotate, open('../../fscores/rotate','rb'))

#############################################################################

print('Computing scale fscores')
fscores_scale = np.zeros((len(scale_values), 5, 80))
filenames = pickle.load(open('../../CMFDdb_grip/listdirs/listdir_scale','rb'))

run_cmd = "main({}, '../../CMFDdb_grip/scale/{}', '../../CMFDdb_grip/scale/{}', false)"
for i in tqdm(range(len(scale_values)*80)):
    [forged, gt] = filenames[2*i,2*i+2]
    for j in range(5):
        fscores_scale[i//5, j, i%5] = eng.eval(run_cmd.format(j+1,forged,gt))
pickle.dump(fscores_scale, open('../../fscores/scale','rb'))

#############################################################################

print('Computing jpeg fscores')
fscores_jpeg = np.zeros((len(jpeg_values), 5, 80))
filenames = pickle.load(open('../../CMFDdb_grip/listdirs/listdir_jpeg','rb'))

run_cmd = "main({}, '../../CMFDdb_grip/jpeg/{}', '../../CMFDdb_grip/jpeg/{}', false)"
for i in tqdm(range(len(jpeg_values)*80)):
    [forged, gt] = filenames[2*i,2*i+2]
    for j in range(5):
        fscores_jpeg[i//5, j, i%5] = eng.eval(run_cmd.format(j+1,forged,gt))
pickle.dump(fscores_jpeg, open('../../fscores/jpeg','rb'))

#############################################################################