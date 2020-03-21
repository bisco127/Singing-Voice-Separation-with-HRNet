"""
This file was adapted from Xiao-Ming's program.
https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer
"""

from librosa.core import load
import const as C
import util
import os
from tqdm import tqdm

PATH_MUSDB_SOURCE = ["Dataset\\musdb18hq\\train"]

FILE_MIX = "mixture.wav"
FILE_VOCAL = "vocals.wav"

list_source_dir = [os.path.join(PATH_MUSDB_SOURCE[0], f)
                   for f in os.listdir(PATH_MUSDB_SOURCE[0])]
list_source_dir = sorted(list_source_dir)

for source_dir in tqdm(list_source_dir[:49]):
    fname = source_dir.split("\\")[-1]
    y_vocal, _ = load(os.path.join(source_dir, FILE_VOCAL), sr=None, mono=False)
    y_mix, _ = load(os.path.join(source_dir, FILE_MIX), sr=None, mono=False)

    y_inst = y_mix - y_vocal
    util.SaveSpectrogram(y_mix, y_vocal, y_inst, fname)
