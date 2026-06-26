import torch
import torch.nn as nn
import torchaudio

class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        # Initialize the torchaudio MelSpectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        # Use L1 or MSE loss to compare the spectrograms
        self.criterion = nn.L1Loss()

    def forward(self, x_pred, x_target):

        _, _, t1 = x_pred.shape
        _, _, t2 = x_target.shape
        min_length = min(t1, t2)

        x_pred = x_pred.narrow(3, start=0, min_length=min_length)
        x_target = x_target.narrow(3, start=0, min_length=min_length)


        # 1. Compute Mel-spectrograms for both target and prediction
        spec_pred = self.mel_spectrogram(x_pred)
        spec_target = self.mel_spectrogram(x_target)
        
        
        # 2. Convert to Decibel (log) scale to mimic human hearing 
        # (Alternatively, you can use torchaudio.transforms.AmplitudeToDB)
        spec_pred_db = torch.log(torch.clamp(spec_pred, min=1e-8))
        spec_target_db = torch.log(torch.clamp(spec_target, min=1e-8))
        
        # 3. Calculate Loss
        loss = self.criterion(spec_pred_db, spec_target_db)
        return loss
