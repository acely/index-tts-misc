import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import random
import os
from torch.utils.data import Dataset
import numpy as np # For saving .npy if needed, though .pt is fine

from indextts.BigVGAN.models import BigVGAN as BigVGANGenerator
from indextts.BigVGAN.ECAPA_TDNN import ECAPA_TDNN # Import ECAPA_TDNN

# ==== Helper HParams Class ====
# (HParams class remains unchanged)
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if 'use_cuda_kernel' not in kwargs:
            self.use_cuda_kernel = False
        if 'snake_logscale' not in kwargs:
            self.snake_logscale = True
        if 'feat_upsample' not in kwargs:
            self.feat_upsample = False
        if 'cond_d_vector_in_each_upsampling_layer' not in kwargs:
            self.cond_d_vector_in_each_upsampling_layer = True

    def get(self, key, default=None):
        return getattr(self, key, default)

# ==== 1. Generator44kHzWithSpeaker 定义 ====
# (Generator44kHzWithSpeaker class remains unchanged)
class Generator44kHzWithSpeaker(nn.Module):
    def __init__(self, mel_dim=128, speaker_embed_dim=512):
        super().__init__()

        self.h_bigvgan = HParams(
            upsample_rates=[8,8,2,2,2],
            upsample_kernel_sizes=[16,16,4,4,4],
            upsample_initial_channel=512,
            resblock_kernel_sizes=[3,7,11],
            resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
            gpt_dim=mel_dim,
            speaker_embedding_dim=speaker_embed_dim,
            num_mels=128,
            activation="snake",
            resblock="1",
        )
        self.generator = BigVGANGenerator(h=self.h_bigvgan)

    def forward(self, mel, d_vector):
        wav_out, _ = self.generator(x=mel, d_vector=d_vector.unsqueeze(-1))
        return wav_out

# ==== 2. AudioMelDataset 定义 ====
class AudioMelDataset(Dataset):
    def __init__(self,
                 wav_dir,
                 # mel_dir is not used for student mels if generated on the fly
                 segment_size,
                 # Student mel params
                 student_sampling_rate,
                 student_hop_length,
                 student_n_fft,
                 student_win_length,
                 student_n_mels,
                 # Speaker embedding params
                 speaker_embed_dim,
                 speaker_embed_dir, # Directory for pre-computed speaker embeddings
                 ecapa_model_path,  # Path to pre-trained ECAPA_TDNN model
                 ecapa_input_mels,
                 ecapa_sampling_rate,
                 ecapa_n_fft,
                 ecapa_hop_length,
                 ecapa_win_length,
                 device='cpu' # Device for ECAPA model if used on-the-fly
                ):
        self.wav_dir = wav_dir
        self.segment_size = segment_size

        # Student mel params
        self.student_sampling_rate = student_sampling_rate
        self.student_hop_length = student_hop_length
        self.student_n_fft = student_n_fft
        self.student_win_length = student_win_length if student_win_length is not None else student_n_fft
        self.student_n_mels = student_n_mels

        # Speaker embedding params
        self.speaker_embed_dim = speaker_embed_dim
        self.speaker_embed_dir = speaker_embed_dir
        self.ecapa_model_path = ecapa_model_path
        self.ecapa_input_mels = ecapa_input_mels
        self.ecapa_sampling_rate = ecapa_sampling_rate
        self.ecapa_n_fft = ecapa_n_fft
        self.ecapa_hop_length = ecapa_hop_length
        self.ecapa_win_length = ecapa_win_length if ecapa_win_length is not None else ecapa_n_fft
        self.device = device

        self.wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])

        # Mel transform for student input
        self.student_mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.student_sampling_rate,
            n_fft=self.student_n_fft,
            hop_length=self.student_hop_length,
            win_length=self.student_win_length,
            n_mels=self.student_n_mels
        ).to(self.device)

        # ECAPA_TDNN for speaker embeddings
        self.speaker_encoder = ECAPA_TDNN(input_size=self.ecapa_input_mels, lin_neurons=self.speaker_embed_dim).to(self.device)
        if self.ecapa_model_path and os.path.exists(self.ecapa_model_path):
            try:
                self.speaker_encoder.load_state_dict(torch.load(self.ecapa_model_path, map_location=self.device))
                print(f"Loaded pre-trained ECAPA_TDNN from {self.ecapa_model_path}")
            except Exception as e:
                print(f"Error loading ECAPA_TDNN weights from {self.ecapa_model_path}: {e}. Using random init.")
        else:
            print("ECAPA_TDNN model path not found or not specified. Using random init for speaker encoder.")
        self.speaker_encoder.eval()

        # Mel transform for ECAPA input
        self.ecapa_mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.ecapa_sampling_rate,
            n_fft=self.ecapa_n_fft,
            hop_length=self.ecapa_hop_length,
            win_length=self.ecapa_win_length,
            n_mels=self.ecapa_input_mels
        ).to(self.device)

        # Mock speaker embedding generator (fallback)
        self.mock_embed_generator = torch.nn.Linear(1, speaker_embed_dim).to(self.device)


    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_filename = self.wav_files[idx]
        wav_path = os.path.join(self.wav_dir, wav_filename)

        d_vector = None

        # Priority 1: Load pre-computed speaker embedding
        if self.speaker_embed_dir:
            embed_filename = wav_filename.replace(".wav", ".pt") # Or .speaker.pt, .npy etc.
            embed_path = os.path.join(self.speaker_embed_dir, embed_filename)
            if os.path.exists(embed_path):
                try:
                    d_vector = torch.load(embed_path, map_location=torch.device('cpu')) # Load to CPU first
                    if d_vector.shape[0] != self.speaker_embed_dim:
                        print(f"Warning: Loaded speaker embedding {embed_path} has incorrect dimension {d_vector.shape}. Expected {self.speaker_embed_dim}. Ignoring.")
                        d_vector = None
                except Exception as e:
                    print(f"Error loading speaker embedding {embed_path}: {e}")
                    d_vector = None

        # Load audio
        try:
            audio_full, sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"Error loading wav file {wav_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.wav_files)) # Fallback

        # Resample to student_sampling_rate for student mels and main audio segment
        if sr != self.student_sampling_rate:
            audio_full = torchaudio.functional.resample(audio_full, sr, self.student_sampling_rate)
        if audio_full.shape[0] > 1: # Ensure mono
            audio_full = torch.mean(audio_full, dim=0)
        audio_full = audio_full.squeeze()

        # Segment or pad audio for student processing
        if audio_full.size(0) >= self.segment_size:
            start = random.randint(0, audio_full.size(0) - self.segment_size)
            audio_segment_student = audio_full[start:start+self.segment_size]
        else:
            audio_segment_student = F.pad(audio_full, (0, self.segment_size - audio_full.size(0)), "constant")

        # Generate mel spectrogram for student input
        mel_student = self.student_mel_transform(audio_segment_student.unsqueeze(0).to(self.device)).squeeze(0)

        # Priority 2: Generate speaker embedding on-the-fly if not loaded
        if d_vector is None:
            try:
                # Prepare audio for ECAPA: use the *full* audio for better embedding, or segment if needed
                # Here, using the full audio resampled to ECAPA's expected sample rate
                audio_for_ecapa = audio_full # Already at student_sampling_rate
                if self.student_sampling_rate != self.ecapa_sampling_rate:
                    audio_for_ecapa = torchaudio.functional.resample(audio_for_ecapa, self.student_sampling_rate, self.ecapa_sampling_rate)

                # Ensure audio_for_ecapa is on the correct device for transformation
                mel_for_ecapa = self.ecapa_mel_transform(audio_for_ecapa.unsqueeze(0).to(self.device)) # (1, n_mels_ecapa, time)

                with torch.no_grad(): # Ensure no gradients for speaker encoder
                    d_vector = self.speaker_encoder(mel_for_ecapa).squeeze(0).squeeze(-1).cpu() # (speaker_embed_dim)
            except Exception as e:
                print(f"Error generating speaker embedding for {wav_filename} with ECAPA_TDNN: {e}")
                d_vector = None

        # Priority 3: Mock embedding (fallback)
        if d_vector is None:
            print(f"Warning: Using mock speaker embedding for {wav_filename}.")
            d_vector = self.mock_embed_generator(torch.ones(1).to(self.device)).squeeze(0).detach().cpu()

        return {
            "mel": mel_student.cpu(),
            "audio": audio_segment_student.cpu(),
            "d_vector": d_vector.cpu() # Ensure all outputs are on CPU
        }
