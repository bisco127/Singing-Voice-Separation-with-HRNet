import argparse
import torch
import util
from librosa.util import find_files
from tqdm import tqdm
from torch_stft import STFT
import SpecHRNet
import const as C
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cuda'

f = open("config/default.yaml", "r+")
config = yaml.safe_load(f)
SpecHRNet = SpecHRNet.HighResolutionNet(config)
f.close()

SpecHRNet.to(device)
torch.manual_seed(32)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="test.model")
parser.add_argument('--name', type=str, default="test")
parser.add_argument('--train', action='store_true')
parser.add_argument('--all', action='store_true')
parser.add_argument('--hard', action='store_true')
parser.add_argument('--summary', action='store_true')
args = parser.parse_args()


stft_4096 = STFT(
            filter_length=4096,
            hop_length=1024,
            window=C.WINDOW
        ).to(device)
stft_2048 = STFT(
            filter_length=2048,
            hop_length=512,
            window=C.WINDOW
        ).to(device)
stft_1024 = STFT(
            filter_length=1024,
            hop_length=256,
            window=C.WINDOW
        ).to(device)


def compute(fname):
    y, mag, norm =\
            util.LoadAudioStereo(fname, stft_4096, stft_2048, stft_1024, device)

    pad_tensor = torch.zeros(2, 2048,
                             C.PATCH_SIZE-mag.shape[2]%C.PATCH_SIZE).to(device)
    mag = torch.cat([mag, pad_tensor], dim=2)

    #print("Computing mask...")
    mask = util.ComputeMask(mag, model_name=args.model, hard=args.hard,
                            SpecHRNet=SpecHRNet, device=device)

    #print("Saving track...")
    util.SaveAudioStereo(
        fname, y, mag, mask, norm, stft_4096, stft_2048, stft_1024, device)

with torch.no_grad():
    if args.all == True:
        filelist = find_files(C.MYDATA_PATH, ext="flac")
        with tqdm(filelist) as pbar:
            for fname in pbar:
                fname = os.path.basename(fname)
                pbar.set_postfix(Processing=fname)
                compute(fname)
    else:
        fname = args.name + ".flac"
        compute(fname)

print("Done")