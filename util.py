import torch

from tqdm import tqdm
import const as C
import numpy as np
import os.path
import soundfile as sf
from librosa.core import stft, load, istft, resample
from sklearn.metrics import mean_squared_error
import SpecHRNet
import random
import torch.nn.functional as F


def SaveSpectrogram(y_mix, y_vocal, y_inst, fname):
    ite = int(np.floor(y_mix[0, :].shape[0]/(np.power(2, 18)-1)))
    y_mix = torch.FloatTensor(y_mix).to(device)
    y_inst = torch.FloatTensor(y_inst).to(device)
    y_vocal = torch.FloatTensor(y_vocal).to(device)

    s_mix, _ = stft.transform(y_mix)
    norm = torch.max(s_mix)

    path = "DataSet/jyogai.txt"
    with open(path) as f:
        exclutionList = [s.strip() for s in f.readlines()]

    for i in tqdm(range(int(np.floor(y_mix[0, :].shape[0]/(np.power(2, 18)-1)) ))):
        if fname not in exclutionList:
            a = (np.power(2,18))*i
            b = (np.power(2,18))*(i+1)-1

            s_mix, _ = stft.transform(y_mix[:, a:b])
            s_inst, _ = stft.transform(y_inst[:, a:b])
            s_vocal, _ = stft.transform(y_vocal[:, a:b])

            s_mix /= norm
            s_inst /= norm
            s_vocal /= norm

            s_mix = s_mix.cpu().numpy()
            s_inst = s_inst.cpu().numpy()
            s_vocal = s_vocal.cpu().numpy()
            #phase_mix = phase_mix.cpu().numpy()
            #wav_inst = y_inst[:, a:b].cpu().numpy()
            #wav_vocal = y_vocal[:, a:b].cpu().numpy()

            np.savez(os.path.join(C.PATH_PHASE, fname + "[" + str(i) +"]" + ".npz"),
                s_mix=s_mix, s_vocal=s_vocal, s_inst=s_inst)
                #phase_mix=phase_mix)#, wav_vocal=wav_vocal, wav_inst=wav_inst)


def LoadDataset(dataPath=None, DataAug=False):
    s_mix = []
    s_inst = []
    s_vocal = []

    for file_fft in dataPath:
        dat = np.load(file_fft)
        assert(dat["s_mix"].shape == dat["s_vocal"].shape == dat["s_inst"].shape)
        s_mix.append(dat["s_mix"])
        s_inst.append(dat["s_inst"])
        s_vocal.append(dat["s_vocal"])

    if DataAug == True:
        #DataAug Channe Swap
        choice3 = np.random.randint(0, len(s_mix))
        s_mix.append(s_mix[choice3][[1,0]])
        s_inst.append(s_inst[choice3][[1,0]])
        s_vocal.append(s_vocal[choice3][[1,0]])

    s_mix = np.asarray(s_mix)
    s_inst = np.asarray(s_inst)
    s_vocal = np.asarray(s_vocal)

    return s_mix, s_inst, s_vocal

def tensorInterpolate(x, n):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(int(C.FFT_SIZE/2), x.shape[3]*n),
                                mode='nearest')
    x = x.squeeze(0)
    return x

def LoadAudioStereo(fname, stft_4096, stft_2048, stft_1024, device):

    y, sr = load(os.path.join(C.TESTDATA_PATH, fname), sr=C.SR, mono=False)
    y = y[:65535, :]
    y = torch.tensor(y).to(device)
    sample = int(np.floor(y.shape[1]/1024))
    sample = sample*1024-1
    y = y[:, :sample]

    mag_4096, _ = stft_4096.transform(y)
    mag_2048, _ = stft_2048.transform(y)
    mag_1024, _ = stft_1024.transform(y)
    norm = torch.max(mag_4096)
    mag_4096 /= norm
    mag_2048 /= norm
    mag_1024 /= norm
    mag_4096 = tensorInterpolate(mag_4096[:, 1:], 4)
    mag_2048 = tensorInterpolate(mag_2048[:, 1:], 2)
    mag_1024 = tensorInterpolate(mag_1024[:, 1:], 1)

    mag = torch.cat([mag_4096[:, :128], mag_2048[:, 128:1024],
                     mag_1024[:, 1024:]], 1)

    mag_4096 = 0
    mag_2048 = 0
    mag_1024 = 0

    torch.cuda.empty_cache()

    return y, mag, norm


def SaveAudioStereo(fname, y_mix, mag, mask, norm, Griffin, stft_4096, stft_2048, stft_1024, device):
    device = 'cpu'
    y_mix = y_mix.cpu()
    mag = mag.cpu()
    mask = mask.cpu()
    norm = norm.cpu()
    stft_4096 = stft_4096.cpu()
    stft_2048 = stft_2048.cpu()
    stft_1024 = stft_1024.cpu()

    vocal_mag = mag*mask
    vocal_mag *=  norm
    inst_mag = mag*(1-mask)
    inst_mag *= norm

    vocal = torch.zeros(mask.shape[0], y_mix.shape[1]).to(device)
    inst = torch.zeros(mask.shape[0], y_mix.shape[1]).to(device)
    mag = 0
    mask = 0

    fname = ["bass-%s" % os.path.splitext(os.path.basename(fname))[0]+".flac",
             "drums-%s" % os.path.splitext(os.path.basename(fname))[0]+".flac",
             "vocal-%s" % os.path.splitext(os.path.basename(fname))[0]+".flac",
             "inst-%s" % os.path.splitext(os.path.basename(fname))[0]+".flac" ]

    vocal_mag = vocal_mag.unsqueeze(0)
    inst_mag = inst_mag.unsqueeze(0)
    mag_temp = F.interpolate(vocal_mag,
                             size=(2048, int(vocal_mag.shape[3]/4)),
                             mode='nearest')
    mag_temp[:, :, 128:] = 0
    pad_temp = torch.zeros([1, 2, 1, mag_temp.shape[3]]).to(device)
    mag_temp = torch.cat([pad_temp, mag_temp], 2)
    mag_temp = mag_temp.squeeze(0)
    _, phase = stft_4096.transform(y_mix)
    mag_temp = mag_temp[:, :, :phase.shape[2]]
    vocal += stft_4096.inverse(mag_temp, phase, y_mix.shape[1])

    mag_temp = F.interpolate(vocal_mag,
                             size=(1024, int(vocal_mag.shape[3]/2)),
                             mode='nearest')
    mag_temp[:, :, :64] = 0
    mag_temp[:, :, 512:] = 0
    pad_temp = torch.zeros([1, 2, 1, mag_temp.shape[3]]).to(device)
    mag_temp = torch.cat([pad_temp, mag_temp], 2)
    mag_temp = mag_temp.squeeze(0)
    _, phase = stft_2048.transform(y_mix)
    mag_temp = mag_temp[:, :, :phase.shape[2]]
    vocal += stft_2048.inverse(mag_temp, phase, y_mix.shape[1])

    mag_temp = F.interpolate(vocal_mag,
                             size=(512, int(vocal_mag.shape[3])),
                             mode='nearest')
    mag_temp[:, :, :256] = 0
    pad_temp = torch.zeros([1, 2, 1, mag_temp.shape[3]]).to(device)
    mag_temp = torch.cat([pad_temp, mag_temp], 2)
    mag_temp = mag_temp.squeeze(0)
    _, phase = stft_1024.transform(y_mix)
    mag_temp = mag_temp[:, :, :phase.shape[2]]
    vocal += stft_1024.inverse(mag_temp, phase, y_mix.shape[1])


    mag_temp = F.interpolate(inst_mag,
                             size=(2048, int(inst_mag.shape[3]/4)),
                             mode='nearest')
    mag_temp[:, :, 128:] = 0
    pad_tmep = torch.zeros([1, 2, 1, mag_temp.shape[3]]).to(device)
    mag_temp = torch.cat([pad_tmep, mag_temp], 2)
    mag_temp = mag_temp.squeeze(0)
    _, phase = stft_4096.transform(y_mix)
    mag_temp = mag_temp[:, :, :phase.shape[2]]
    inst += stft_4096.inverse(mag_temp, phase, y_mix.shape[1])

    mag_temp = F.interpolate(inst_mag,
                             size=(1024, int(inst_mag.shape[3]/2)),
                             mode='nearest')
    mag_temp[:, :, :64] = 0
    mag_temp[:, :, 512:] = 0
    pad_temp = torch.zeros([1, 2, 1, mag_temp.shape[3]]).to(device)
    mag_temp = torch.cat([pad_temp, mag_temp], 2)
    mag_temp = mag_temp.squeeze(0)
    _, phase = stft_2048.transform(y_mix)
    mag_temp = mag_temp[:, :, :phase.shape[2]]
    inst += stft_2048.inverse(mag_temp, phase, y_mix.shape[1])

    mag_temp = F.interpolate(inst_mag,
                             size=(512, int(inst_mag.shape[3])),
                             mode='nearest')
    mag_temp[:, :, :256] = 0
    pad_tmep = torch.zeros([1, 2, 1, mag_temp.shape[3]]).to(device)
    mag_temp = torch.cat([pad_tmep, mag_temp], 2)
    mag_temp = mag_temp.squeeze(0)
    _, phase = stft_1024.transform(y_mix)
    mag_temp = mag_temp[:, :, :phase.shape[2]]
    inst += stft_1024.inverse(mag_temp, phase, y_mix.shape[1])

    if Griffin == True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        stft_4096 = stft_4096.to(device)
        stft_2048 = stft_2048.to(device)
        stft_1024 = stft_1024.to(device)
        y_mix = y_mix.to(device)
        vocal = vocal.to(device)
        inst = inst.to(device)
        _, phase = stft_2048.transform(y_mix)
        vocal_mag, _ = stft_2048.transform(vocal)
        inst_mag, _ = stft_2048.transform(inst)

        print("Starting Giffin-Lim Phase Estimate")
        i=0
        phase_vocal = phase
        phase_inst = phase
        tensorSize = phase_vocal.shape[0] *phase_vocal.shape[1]*phase_vocal.shape[2]
        while True:
            with tqdm(range(100)) as pbar:
                for i in pbar:
                    old_phase_vocal = phase_vocal
                    _, phase_vocal = stft_2048.transform(vocal)
                    _, phase_inst = stft_2048.transform(inst)

                    padTensor = torch.zeros(old_phase_vocal.shape[0],
                                       old_phase_vocal.shape[1],
                                       old_phase_vocal.shape[2]-phase_vocal.shape[2]).to(device)
                    phase_vocal = torch.cat((phase_vocal, padTensor), dim=2)
                    phase_inst = torch.cat((phase_inst, padTensor), dim=2)
                    loss = torch.mean((phase_vocal-old_phase_vocal)**2)

                    vocal = stft_2048.inverse(vocal_mag, phase_vocal, y_mix.shape[1])
                    inst = stft_2048.inverse(inst_mag, phase_inst, y_mix.shape[1])

                    #print(str(i) + " : ")
                    i += 1
                    pbar.set_postfix(loss=loss.data)
                    #if loss <= 1.5*10**-5:
                break

        vocal = stft_2048.inverse(vocal_mag, phase_vocal, y_mix.shape[1])
        inst = stft_2048.inverse(inst_mag, phase_inst, y_mix.shape[1])

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


def ComputeMask(input_mag, model_name, hard, SpecHRNet, device):
    SpecHRNet.load_state_dict(torch.load(C.PATH_MODEL+model_name))
    SpecHRNet.eval()

    mask = torch.zeros(2, input_mag.shape[1], input_mag.shape[2]).to(device)

    patch = 128
    ite = int(input_mag.shape[2]/patch)
    print(input_mag.shape)
    input_mag = input_mag.unsqueeze(0)

    for i in tqdm(range(ite)):
        mask[:, :, i*patch:(i+1)*patch] = \
            SpecHRNet(input_mag[:, :, :, i*patch:(i+1)*patch]).unsqueeze(1)
    input_mag = 0
    torch.cuda.empty_cache()


    if hard:
        hard_mask = torch.zeros(mask.shape).to(device)
        hard_mask[mask > 0.075] = mask[mask > 0.075]
        return hard_mask
    else:
        return mask
