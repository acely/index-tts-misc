import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class MelSpectrogramLoss(nn.Module):
    """
    Loss module that calculates L1 loss between log mel spectrograms of input and target.
    """
    def __init__(self,
                 sample_rate=44100,
                 n_fft=2048,
                 hop_length=512,
                 win_length=None,
                 n_mels=128,
                 mel_fmin=0.0,
                 mel_fmax=None,
                 power=1.0, # For torchaudio.transforms.MelSpectrogram, 1.0 for energy, 2.0 for power
                 normalized=False,
                 center=True,
                 pad_mode="reflect"):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode

        # Mel transform is created on the fly or needs device argument if not scripting
        # For nn.Module, it's better to register it if it has parameters,
        # but torchaudio transforms are often fine to create like this.
        # However, to handle device placement correctly without JIT scripting,
        # it's safer to recreate or move it in the forward pass if its device differs.
        self.mel_transform_params = {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "f_min": self.mel_fmin,
            "f_max": self.mel_fmax,
            "n_mels": self.n_mels,
            "power": self.power,
            "normalized": self.normalized,
            "center": self.center,
            "pad_mode": self.pad_mode,
        }
        # self.mel_transform = torchaudio.transforms.MelSpectrogram(**self.mel_transform_params)
        self.loss = nn.L1Loss()

    def _get_mel_transform(self, device):
        # Helper to ensure mel_transform is on the correct device
        # This is a workaround if self.mel_transform itself isn't an nn.Module registered to the parent
        # or if it doesn't automatically move with .to(device)
        # A simpler way if not using JIT is to just create it in forward or ensure it's moved.
        # For this implementation, let's assume we create it freshly or it's already on device.
        # The provided code snippet in the prompt has a check in forward, which is good.
        # The prompt's original self.mel_transform init is fine.
        return torchaudio.transforms.MelSpectrogram(**self.mel_transform_params).to(device)


    def forward(self, y_hat, y):
        """
        Args:
            y_hat (Tensor): Predicted waveform (B, T).
            y (Tensor): Ground truth waveform (B, T).
        Returns:
            Tensor: L1 loss between log mel spectrograms.
        """
        # Create MelSpectrogram on the correct device if not already done or if it's not an nn.Module
        # This ensures that the transform's internal buffers (like windows) are on the correct device.
        mel_transform_op = torchaudio.transforms.MelSpectrogram(**self.mel_transform_params).to(y_hat.device)

        mel_y_hat = mel_transform_op(y_hat)
        mel_y = mel_transform_op(y)

        # Using log mel spectrograms for loss calculation is common
        log_mel_y_hat = torch.log(torch.clamp(mel_y_hat, min=1e-5))
        log_mel_y = torch.log(torch.clamp(mel_y, min=1e-5))

        return self.loss(log_mel_y_hat, log_mel_y)

class STFTLoss(nn.Module):
    """
    Loss module that calculates STFT-based losses:
    1. Spectral Convergance Loss
    2. Log STFT Magnitude Loss
    """
    def __init__(self,
                 n_fft=2048,
                 hop_length=512,
                 win_length=None,
                 window_fn_constructor=torch.hann_window, # Pass constructor, not instance
                 center=True,
                 pad_mode="reflect"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window_fn_constructor = window_fn_constructor
        self.center = center
        self.pad_mode = pad_mode
        self.l1_loss = nn.L1Loss()

        # Window is created on the fly in _stft based on device

    def _stft(self, x):
        # Window needs to be created on the same device as input x
        window = self.window_fn_constructor(self.win_length, device=x.device)
        return torch.stft(x,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=window,
                          center=self.center,
                          pad_mode=self.pad_mode,
                          return_complex=True)

    def forward(self, y_hat, y):
        """
        Args:
            y_hat (Tensor): Predicted waveform (B, T).
            y (Tensor): Ground truth waveform (B, T).
        Returns:
            Tensor: Spectral convergence loss.
            Tensor: Log STFT magnitude loss.
        """
        stft_y_hat_complex = self._stft(y_hat)
        stft_y_complex = self._stft(y)

        mag_y_hat = torch.abs(stft_y_hat_complex)
        mag_y = torch.abs(stft_y_complex)

        # Spectral Convergence Loss
        # Frobenius norm is sqrt(sum of squares of elements)
        # torch.norm(tensor, p='fro')
        sc_loss = torch.norm(mag_y - mag_y_hat, p="fro") / (torch.norm(mag_y, p="fro") + 1e-9) # Add epsilon for stability

        # Log STFT Magnitude Loss
        log_mag_y_hat = torch.log(torch.clamp(mag_y_hat, min=1e-5))
        log_mag_y = torch.log(torch.clamp(mag_y, min=1e-5))
        log_mag_loss = self.l1_loss(log_mag_y_hat, log_mag_y)

        return sc_loss, log_mag_loss
