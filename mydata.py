"""
This file was adapted from Xiao-Ming's program.
https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer
"""

from librosa.core import load, resample
from librosa.util import find_files
import const as C
import util

PATH_MYDATA = "Inst"
filelist = find_files(PATH_MYDATA, ext="wav")

for i in range(2, len(filelist), 2):
    print("Processing:" + filelist[i])
    print("Processing:" + filelist[i+1])
    y_mix, _ = load(filelist[i], sr=None, mono=False)
    y_vocal, sr = load(filelist[i+1], sr=None, mono=False)

    if sr == 44100:
        minlength = min([y_mix[0, :].size, y_vocal[0, :].size])
        y_mix = resample(y_mix[:, :minlength], sr, C.SR)
        y_vocal = resample(y_vocal[:, :minlength], sr, C.SR)
        y_inst = y_mix - y_vocal

        fname = filelist[i].split("\\")[-1].split(".")[0]

        util.SaveSpectrogram(y_mix, y_vocal, y_inst, fname)
