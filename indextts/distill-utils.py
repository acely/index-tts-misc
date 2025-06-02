import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import random
import os
from torch.utils.data import Dataset

# ==== 1. Generator44kHzWithSpeaker 定义 ====
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = self.conv2(x)
        return x + residual

class Generator44kHzWithSpeaker(nn.Module):
    def __init__(self, mel_dim=100, speaker_embed_dim=512):
        super().__init__()
        self.mel_dim = mel_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.upsample_initial_channel = 512

        self.pre_conv = nn.Conv1d(mel_dim, self.upsample_initial_channel, 7, 1, padding=3)

        self.upsample_layers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.spk_proj = nn.ModuleList()

        channels = self.upsample_initial_channel
        for i, (k, s) in enumerate(zip([16, 8, 4, 4, 4, 4], [8, 4, 2, 2, 2, 2])):
            self.upsample_layers.append(nn.ConvTranspose1d(channels, channels // 2, k, s, padding=(k - s)//2))
            self.spk_proj.append(nn.Linear(speaker_embed_dim, channels // 2))
            self.resblocks.append(ResBlock(channels // 2))
            channels = channels // 2

        self.post_conv = nn.Conv1d(channels, 1, 7, 1, padding=3)

    def forward(self, mel, d_vector):
        x = self.pre_conv(mel)
        for i in range(len(self.upsample_layers)):
            x = F.leaky_relu(x, 0.1)
            x = self.upsample_layers[i](x)

            spk = self.spk_proj[i](d_vector).unsqueeze(-1)
            x = x + spk

            x = self.resblocks[i](x)

        x = F.tanh(self.post_conv(x))
        return x


# ==== 2. AudioMelDataset 定义 ====
class AudioMelDataset(Dataset):
    def __init__(self, wav_dir, mel_dir, segment_size, hop_length, sampling_rate, speaker_embed_dim):
        self.wav_dir = wav_dir
        self.mel_dir = mel_dir
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.speaker_embed_dim = speaker_embed_dim

        self.wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])

        self.embed_generator = torch.nn.Linear(1, speaker_embed_dim)  # mock embedding (for demo only)

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.wav_dir, self.wav_files[idx])
        audio, sr = torchaudio.load(wav_path)
        audio = torchaudio.functional.resample(audio, sr, self.sampling_rate)
        audio = audio[0]  # mono

        if audio.size(0) >= self.segment_size:
            start = random.randint(0, audio.size(0) - self.segment_size)
            audio = audio[start:start+self.segment_size]
        else:
            audio = F.pad(audio, (0, self.segment_size - audio.size(0)), "constant")

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate, n_fft=1024, hop_length=self.hop_length, n_mels=100
        )(audio.unsqueeze(0)).squeeze(0)

        d_vector = self.embed_generator(torch.ones(1))  # mock embedding vector (for demo)

        return {
            "mel": mel,
            "audio": audio,
            "d_vector": d_vector.squeeze(0)
        }
