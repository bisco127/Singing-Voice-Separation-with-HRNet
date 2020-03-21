import argparse
import torch
from tqdm import tqdm
from torch_stft import STFT
from librosa.core import load
import os.path
import soundfile as sf
import const as C
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stft_2048 = STFT(
                filter_length=1024,
                hop_length=256,
                window=C.WINDOW
            ).to(device)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="test")
args = parser.parse_args()
fname = args.name + ".flac"
path = 'Result'

mix, sr = load(os.path.join(C.TESTDATA_PATH, fname), sr=C.SR, mono=False)
inst, _ = load(os.path.join(path, 'inst', 'inst-'+fname), sr=C.SR, mono=False)
vocal, _ = load(os.path.join(path, 'vocal', 'vocal-'+fname), sr=C.SR, mono=False)

mix = torch.tensor(mix).to(device)
inst = torch.tensor(inst).to(device)
vocal = torch.tensor(vocal).to(device)

_, phase = stft_2048.transform(inst)
inst_mag, _ = stft_2048.transform(inst)
vocal_mag, _ = stft_2048.transform(vocal)

phase_vocal = phase
phase_inst = phase

fname = ["bass-%s" % os.path.splitext(os.path.basename(fname))[0]+".flac",
         "drums-%s" % os.path.splitext(os.path.basename(fname))[0]+".flac",
         "vocal-%s" % os.path.splitext(os.path.basename(fname))[0]+".flac",
         "inst-%s" % os.path.splitext(os.path.basename(fname))[0]+".flac" ]

with tqdm(range(100)) as pbar:
    for i in pbar:
        old_phase_vocal = phase_vocal
        _, phase_vocal = stft_2048.transform(vocal)
        _, phase_inst = stft_2048.transform(inst)

        padTensor = torch.zeros(old_phase_vocal.shape[0],
                           old_phase_vocal.shape[1],
                           -1*(old_phase_vocal.shape[2]-phase_vocal.shape[2])).to(device)
        phase_vocal = torch.cat((phase_vocal, padTensor), dim=2)
        phase_inst = torch.cat((phase_inst, padTensor), dim=2)
        phase_vocal = phase_vocal[..., :old_phase_vocal.shape[2]]
        phase_inst = phase_inst[..., :old_phase_vocal.shape[2]]
        loss = torch.mean((phase_vocal-old_phase_vocal)**2)

        vocal = stft_2048.inverse(vocal_mag, phase_vocal, mix.shape[1])
        inst = stft_2048.inverse(inst_mag, phase_inst, mix.shape[1])

        #print(str(i) + " : ")
        i += 1
        pbar.set_postfix(loss=loss.data)
        #if loss <= 1.5*10**-5:

vocal[vocal > 1] = 1
vocal[vocal < -1] = -1
vocal *= 32767 / torch.max(torch.abs(vocal))
vocal = vocal / torch.max(vocal) * 32767
temp_vocal = vocal.clone()
vocal = torch.transpose(vocal, 0, 1)
vocal = vocal.cpu().numpy()

inst[inst > 1] = 1
inst[inst < -1] = -1
inst *= 32767 / torch.max(torch.abs(inst))
inst = inst / torch.max(inst) * 32767
inst = torch.transpose(inst, 0, 1)
inst = inst.cpu().numpy()
sf.write(os.path.join(C.PATH_VAL_DATA, "inst", fname[3]),
         inst.astype(np.int16), C.SR, format='flac', subtype='PCM_16')

sf.write(os.path.join(C.PATH_VAL_DATA, "vocal", fname[2]),
         vocal.astype(np.int16), C.SR, format='flac', subtype='PCM_16')