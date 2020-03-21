"""
This file was adapted from Xiao-Ming's program.
https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer
"""

from librosa.core import load, resample
from librosa.util import find_files
import const as C
import os
from tqdm import tqdm
import util

PATH_CCMIXTER = "DataSet/ccmixter_corpus/"
files = os.listdir(PATH_CCMIXTER)
files_dir = [f for f in files if os.path.isdir(os.path.join(PATH_CCMIXTER, f))]

for folder in tqdm(files_dir):
    filelist = find_files(PATH_CCMIXTER+folder, ext="wav")

    print("Processing:" + filelist[0])
    print("Processing:" + filelist[1])
    y_mix, _ = load(filelist[0], sr=None, mono=False)
    y_vocal, sr = load(filelist[2], sr=None, mono=False)
    if y_mix.shape[0] == 2:
        minlength = min([y_mix[0, :].size, y_vocal[0, :].size])
        y_mix = resample(y_mix[:, :minlength], sr, C.SR)
        y_vocal = resample(y_vocal[:, :minlength], sr, C.SR)
        y_inst = y_mix - y_vocal

        fname = folder.split("\\")[-1].split(".")[0]
        util.SaveSpectrogram(y_mix, y_vocal, y_inst, fname)
    else:
        print("This file is MONO")