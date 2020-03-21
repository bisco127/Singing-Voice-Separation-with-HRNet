import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import detect_anomaly

from tqdm import tqdm
import numpy as np
import const as C
import util
import random
from torchsummary import summary


def Train(epoch, SpecHRNet, device, stft):
    #SpecHRNet.load_state_dict(torch.load(C.PATH_MODEL+"a.model"))
    criterion = nn.L1Loss()
    optimizer = optim.Adam(SpecHRNet.parameters(), lr=C.lr)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 150], gamma=0.5)

    loss = 10000

    filePathlist = find_files(C.PATH_FFT, ext="npz")
    allDataset = np.asarray(filePathlist)

    random.seed(0) #init seed

    train_size = int(allDataset.shape[0]-allDataset.shape[0]*0.1)
    val_size = allDataset.shape[0]-train_size

    train_data_path, val_data_path = \
        torch.utils.data.random_split(allDataset, [train_size, val_size])


    batch_size = C.BATCH_SIZE-1

    for ep in range(epoch):
        SpecHRNet.train()
        sum_loss = 0.0
        train_ep = 0

        dataloader = torch.utils.data.DataLoader(train_data_path,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        with tqdm(dataloader) as pbar:
            for data in pbar:
                with detect_anomaly():
                    X, Y_inst, Y = util.LoadDataset(dataPath=data, DataAug=True)

                    X = torch.tensor(X[:, :, 1:, :]).to(device)
                    Y_inst = torch.tensor(Y_inst[:, :, 1:, :]).to(device)
                    Y = torch.tensor(Y[:, :, 1:, :]).to(device)

                    optimizer.zero_grad()
                    O = SpecHRNet(X)

                    loss = criterion(X * O, Y) + criterion(X * (1-O), Y_inst)

                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
                    train_ep += 1

        torch.save(SpecHRNet.state_dict(),
                   "./checkpoint/ckpt-"+ str(ep+1) + ".model")

        #check validate
        with torch.no_grad():
            SpecHRNet.eval()
            dataloader = torch.utils.data.DataLoader(val_data_path,
                                                     batch_size=batch_size,
                                                     shuffle=True)
            val_loss = 0.0
            val_ep = 0
            with tqdm(dataloader) as pbar:
                for data in pbar:
                    X, Y_inst, Y = util.LoadDataset(dataPath=data, DataAug=True)
                    X = torch.tensor(X[:, :, 1:, :]).to(device)
                    Y_inst = torch.tensor(Y_inst[:, :, 1:, :]).to(device)
                    Y = torch.tensor(Y[:, :, 1:, :]).to(device)

                    O = SpecHRNet(X)

                    loss = criterion(X * O, Y) + criterion(X * (1-O), Y_inst)
                    val_loss += loss.item()
                    val_ep += 1
                    pbar.set_postfix(val_loss=loss.item())

        tqdm.write("epoch: %d/%d  train_loss=%.5f val_loss=%.5f" %
                                    (ep + 1, epoch, sum_loss/train_ep, val_loss/val_ep))
        #scheduler.step()
        loss = sum_loss