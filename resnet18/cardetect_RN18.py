import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import time
import datetime

import sklearn.model_selection as skms


from tqdm import tqdm

import math, random
from IPython.display import Audio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch.utils.data import DataLoader, Dataset

from torchmetrics.classification import BinaryAccuracy, BinaryRecall,BinaryPrecision

import torchvision.utils as vutils
import torchvision.models as models
from torchvision.models import ResNet18_Weights

import torchaudio
from torchaudio import transforms


from watermark import watermark

from typing import Tuple
from typing import Optional
import warnings

import multiprocessing as mp

class MyDataset(Dataset):
    def __init__(self, 
                 csv_path: str, 
                 aud_dir: str,
                 sampling_rate: int
                 ):
        df = pd.read_csv(csv_path, sep = ',')
        self.aud_names = df['filename']
        self.labels = df['Engine']

        self.csv_path = csv_path
        self.aud_dir = aud_dir
        self.sampling_rate = sampling_rate
        self.duration = 3000
        self.channel = 1
        self.shift_pct = 0.4

        # based on DataFrame columns



    def __getitem__(self, index):
        audio_file = os.path.join(self.aud_dir,self.aud_names[index])
        label = self.labels[index]

        
        try:
            aud = AudioUtil.open(audio_file)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            raise

        reaud = AudioUtil.resample(aud, self.sampling_rate)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        # dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        # shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        # sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        
        # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct = 0.1, n_freq_masks = 2, n_time_masks = 2)
        # return aug_sgram, label 

        sgram = AudioUtil.spectro_gram(rechan, n_mels=224, n_fft=16384, hop_len=441, f_max=400)
        return sgram, label 
    
    def __len__(self):
        return self.labels.shape[0]


class MyUS8K(Dataset):
    def __init__(self, 
                 csv_path: str, 
                 aud_dir: str,
                 sampling_rate: int, 
                 train: bool = True,
                 fold: Optional[int] = None,
                 random_split_seed: Optional[int]= None,
                 mono: bool = True,
                 ):
        
        super(MyUS8K, self).__init__()

        self.csv_path = csv_path
        self.aud_dir = aud_dir
        self.sampling_rate = sampling_rate
        self.duration = 3000
        self.channel = 1
        self.shift_pct = 0.4
        self.train = train

        if fold is None:
            fold = 1
        if not (1<= fold <= 10):
            raise ValueError(f'Expected fold in range [1, 10], got {fold}')

        self.fold = fold
        self.folds_to_load = set(range(1,11))

        if self.train:
            self.folds_to_load -= {self.fold}
        else:
            self.folds_to_load -= self.folds_to_load - {self.fold}

        self.random_split_seed = random_split_seed
        self.mono = mono

        self.data = dict()
        self.indices = dict()
        self.load_data()

    @staticmethod
    def _load_worker(fn: str, path_to_file: str, sample_rate: int, mono: bool = False) -> Tuple[str, int, np.ndarray]:
        wav, sample_rate = librosa.load(path_to_file, sr=sample_rate, mono=mono)

        if wav.ndim == 1:
            wav = wav[np.newaxis, :]

            if not mono:
                wav = np.concatenate((wav, wav), axis=0)

        wav = wav.T
        wav = wav[:sample_rate * 3] #modif ici car longueur de fichier de 3 secondes

        if np.abs(wav.max()) > 1.0:
            wav = AudioUtil.scale(wav, wav.min(), wav.max(), -1.0, 1.0)

        wav = AudioUtil.scale(wav, wav.min(), wav.max(), -32768.0, 32767.0).T

        return fn, sample_rate, wav.astype(np.float32)
    
    def load_data(self):
        meta = pd.read_csv(self.csv_path, sep = ';', index_col = 'filename')
        for row_idx, (fn,row) in enumerate(meta.iterrows()):
            path = os.path.join(self.aud_dir,fn)
            self.data[fn]=path,self.sampling_rate, self.mono
        files_to_load = list()
        if self.random_split_seed is not None:
            skf = skms.StratifiedKFold(n_splits = 10, shuffle = True, random_state = self.random_split_seed)

            for fold_idx, (train_ids, test_ids) in enumerate(skf.split(
                np.zeros(len(meta)),meta['Engine'].values.astype(int)
            ),1):
                if fold_idx == self.fold:
                    ids = train_ids if self.train else test_ids
                    filenames = meta.iloc[ids].index
                    files_to_load.extend(filenames)
                    break
        else:
            for fn, row in meta.iterrows():
                if int(row['fold']) in self.folds_to_load:
                    files_to_load.append(fn)
        
        self.data = {fn: vals for fn, vals in self.data.items() if fn in files_to_load}
        self.indices = {idx: fn for idx,fn in enumerate(self.data)}

        num_processes = mp.cpu_count()
        warnings.filterwarnings('ignore')
        with mp.Pool(processes=num_processes) as pool:
            chunksize = int(np.ceil(len(meta) / num_processes)) or 1

            tqdm.write(f'Loading {self.__class__.__name__} (train={self.train})')
            # print(self.data.items())
            for fn, sample_rate, wav in pool.starmap(
                func=self._load_worker,
                iterable=[(fn, path, sr, mono) for fn, (path, sr, mono) in self.data.items()],
                chunksize=chunksize
            ):
                self.data[fn] = {
                    'filename':fn,
                    'audio': wav,
                    'sample_rate': sample_rate,
                    'target': meta.loc[fn, 'Engine']
                }
        print(self.data.items())
        

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        if not (0 <= index < len(self)):
            raise IndexError

        audio: np.ndarray = self.data[self.indices[index]]['audio']
        target: int = self.data[self.indices[index]]['target']
        sgram = AudioUtil.spectro_gram((audio,self.sampling_rate), n_mels=224, n_fft=8192*2, hop_len=441, f_max=400)

        return sgram, target

    def __len__(self) -> int:
        return len(self.data)

class AudioClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        conv_layers = []

        self.conv1 = torch.nn.Conv2d(1,8, kernel_size= (5,5),stride = (2,2), padding = (2,2))
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a = 0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        self.conv2 = torch.nn.Conv2d(8,16, kernel_size= (3,3),stride = (2,2), padding = (1,1))
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a = 0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        self.conv3 = torch.nn.Conv2d(16,32, kernel_size= (3,3),stride = (2,2), padding = (1,1))
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a = 0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        
        self.conv4 = torch.nn.Conv2d(32,64, kernel_size= (3,3),stride = (2,2), padding = (1,1))
        self.relu4 = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a = 0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        self.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = torch.nn.Linear(in_features=64, out_features= 256)
        self.act = torch.nn.ReLU()
        self.lin = torch.nn.Linear(in_features=256,out_features = 1)

        self.conv = torch.nn.Sequential(*conv_layers)
    def forward(self,x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        x = self.act(x)
        x = self.lin(x)

        return x
    
class AudioResNet18(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, pretrained=False):
        super(AudioResNet18, self).__init__()
        
        # Charger un ResNet18
        # self.resnet = models.resnet18(pretrained=pretrained)
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        # Adapter la premi�re couche pour le nombre de canaux (1 pour spectrogramme)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Adapter la derni�re couche fully connected pour le nombre de classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class AudioUtil():
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        # sig,sr = librosa.load(audio_file)
        return (sig,sr)
    
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0]== new_channel):
            return aud
        
        if (new_channel == 1):
            resig = sig[:1,:]
        else:
            resig = torch.cat([sig, sig])

        return((resig,sr))
    
    @staticmethod
    def resample(aud,newsr):
        sig, sr = aud

        if (sr == newsr):
            return aud

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr,newsr)(sig[:1,:])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr,newsr)(sig[1:,:])
            resig = torch.cat([resig,retwo])

        return ((resig,newsr))
    
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig,sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000* max_ms

        if(sig_len > max_len):
            sig = sig[:,:max_len]
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0,max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        
        return (sig,sr)
    
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt),sr)
    
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None, f_max = None):
        sig,sr = aud
        top_db = 80
        
        if isinstance(sig, np.ndarray):
            sig = torch.from_numpy(sig)
        if sig.ndim == 1:
            sig = sig.unsqueeze(0)
        sig = sig.float()

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels, f_max=f_max)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
    

    
    @staticmethod
    def spectro_augment(spec, max_mask_pct = 0.1, n_freq_masks =1, n_time_masks =1):
        _,n_mels,n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        
        return aug_spec
    @staticmethod
    def scale(old_value, old_min, old_max, new_min, new_max):
        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

        return new_value
    

def training(model, train_dl, val_dl, num_epochs,device):
    pos_weight = torch.tensor([12.09]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
    #                                                 max_lr= 0.005,
    #                                                 steps_per_epoch = int(len(train_dl)),
    #                                                 epochs = num_epochs,
    #                                                 anneal_strategy= 'linear'
    #                                                 )
    
    patience = 10
    counter = 0
    best_val_loss = float('inf')

    accuracy = BinaryAccuracy().to(device)
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        print(f'Epoch: {epoch+1}')
        for data in tqdm(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m)/inputs_s

            optimizer.zero_grad()
            # inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # scheduler.step()

            running_loss += loss.item() * inputs.size(0)

            accuracy.update(outputs,labels)
            recall.update(outputs,labels)
            precision.update(outputs,labels)

            # if i % 10 == 0:
            #     print('[%d,%5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 10))

        train_loss = running_loss/len(train_dl.dataset)
        train_acc = accuracy.compute()
        train_tpr = recall.compute()
        train_ppv = precision.compute()

        print(f'(TRAIN) Loss: {train_loss:.2f}, Accuracy: {train_acc:.2f}, TPR: {train_tpr:.4f}, PPV: {train_ppv:.4f}')
        # --- Validation phase

        model.eval()

        accuracy.reset()
        recall.reset()
        precision.reset()

        val_loss = 0.0

        with torch.no_grad():
            for data in tqdm(val_dl):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m)/inputs_s

                # inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                labels = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)

                accuracy.update(outputs, labels)
                recall.update(outputs, labels)
                precision.update(outputs, labels)

        val_loss = val_loss / len(val_dl.dataset)
        val_acc = accuracy.compute()
        val_tpr = recall.compute()
        val_ppv = precision.compute()

        print(f'(EVAL) Loss: {val_loss:.2f}, Accuracy: {val_acc:.2f}, TPR: {val_tpr:.4f}, PPV: {val_ppv:.4f}')

         # --- Early stopping & saving model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_ResNet18v3_model-2026(pretrained).pth')
            # torch.save(model.state_dict(), 'best_ResNet18_model.pth')
            print(f"Best model saved at epoch {epoch+1} with val_loss={val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch +1}.")
                break
    

    print('Finished Training')


    
if __name__ == "__main__":
    print(watermark(packages="torch", python =True))  
    start_time = time.time()
    aud_dir=r'/bettik/PROJECTS/pr-orchampvision/phano/cardetect/Audio-3s-32000'
    # train_dataset = MyDataset(
    #     csv_path = '/bettik/PROJECTS/pr-orchampvision/phano/cardetect/custom_NN/oc22_ech-train.csv',
    #     aud_dir = aud_dir,
    #     sampling_rate= 32000
    #     )

    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size= 128,
    #     shuffle=True,
    #     num_workers = 8
    # )

    # val_dataset = MyDataset(
    #     csv_path = '/bettik/PROJECTS/pr-orchampvision/phano/cardetect/custom_NN/oc22_ech-val.csv',
    #     aud_dir =aud_dir,
    #     sampling_rate= 32000
    #     )

    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size= 128,
    #     shuffle=False,
    #     num_workers = 8
    # )

    train_dataset = MyUS8K(
        csv_path= '/bettik/PROJECTS/pr-orchampvision/phano/cardetect/custom_NN/annotations-training-placette.csv',
        aud_dir= aud_dir,
        sampling_rate= 32000,
        train = True      
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size= 128,
        shuffle=True,
        num_workers = 8
    )

    val_dataset = MyUS8K(
        csv_path= '/bettik/PROJECTS/pr-orchampvision/phano/cardetect/custom_NN/annotations-training-placette.csv',
        aud_dir= aud_dir,
        sampling_rate= 32000,
        train = False
        
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size= 128,
        shuffle=False,
        num_workers = 8
    )



    # model = AudioClassifier()
    model = AudioResNet18(num_classes=1,in_channels=1,pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    next(model.parameters()).device

    
    num_epochs = 300
    
    training(model,train_loader,val_loader,num_epochs,device)
    end_time = time.time()
    elapsed_time = end_time - start_time  # In seconds
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
    print(f"Training completed in {elapsed_str} (hh:mm:ss)")



    
    
    