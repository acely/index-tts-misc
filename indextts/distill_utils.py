import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import random
import os
from torch.utils.data import Dataset
import numpy as np
from attrdict import AttrDict

from indextts.BigVGAN.models import BigVGAN
from indextts.BigVGAN.ECAPA_TDNN import ECAPA_TDNN


# ==== 1. Generator44kHzWithSpeaker 定义 ====
class Generator44kHzWithSpeaker(nn.Module):
    def __init__(self, mel_dim_input, speaker_embed_dim, bigvgan_config_dict):
        super().__init__()
        hparams_values = {
            "upsample_rates": bigvgan_config_dict.get("upsample_rates", [8,8,2,2,2]),
            "upsample_kernel_sizes": bigvgan_config_dict.get("upsample_kernel_sizes", [16,16,4,4,4]),
            "upsample_initial_channel": bigvgan_config_dict.get("upsample_initial_channel", 512),
            "resblock_kernel_sizes": bigvgan_config_dict.get("resblock_kernel_sizes", [3,7,11]),
            "resblock_dilation_sizes": bigvgan_config_dict.get("resblock_dilation_sizes", [[1,3,5],[1,3,5],[1,3,5]]),
            "gpt_dim": mel_dim_input,
            "speaker_embedding_dim": speaker_embed_dim,
            "num_mels": mel_dim_input,
            "activation": bigvgan_config_dict.get("activation", "snake"),
            "resblock": bigvgan_config_dict.get("resblock", "1"),
            "feat_upsample": bigvgan_config_dict.get("feat_upsample", False),
            "cond_d_vector_in_each_upsampling_layer": bigvgan_config_dict.get("cond_d_vector_in_each_upsampling_layer", True),
            "use_cuda_kernel": bigvgan_config_dict.get("use_cuda_kernel", False),
            "snake_logscale": bigvgan_config_dict.get("snake_logscale", True)
        }
        internal_hparams = AttrDict(hparams_values)
        self.speaker_embed_dim = speaker_embed_dim
        self.bigvgan = BigVGAN(h=internal_hparams)

    def forward(self, mel_spec, d_vector):
        output_waveform, _ = self.bigvgan(x=mel_spec, d_vector=d_vector)
        return output_waveform

# ==== 2. AudioMelDataset 定义 ====
class AudioMelDataset(Dataset):
    def __init__(self,
                 wav_dir,
                 segment_size,
                 # Student mel params
                 student_sampling_rate,
                 student_hop_length,
                 student_n_fft,
                 student_win_length,
                 student_n_mels,
                 # Speaker embedding params
                 speaker_embed_dim,
                 speaker_embed_dir=None, # Default to None
                 ecapa_model_path=None,  # Default to None
                 ecapa_input_mels=80,    # Default as per subtask
                 ecapa_sampling_rate=24000,# Default as per subtask
                 ecapa_n_fft=1024,       # Default as per subtask
                 ecapa_hop_length=256,   # Default as per subtask
                 ecapa_win_length=1024,  # Default as per subtask
                 ecapa_mel_fmin=0.0,     # Default as per subtask
                 ecapa_mel_fmax=None,    # Default as per subtask
                 device='cpu',
                 **kwargs # To catch any other student_mel_ params if passed by old config
                ):
        self.wav_dir = wav_dir
        self.segment_size = segment_size

        self.student_sampling_rate = student_sampling_rate
        self.student_hop_length = student_hop_length
        self.student_n_fft = student_n_fft
        self.student_win_length = student_win_length if student_win_length is not None else student_n_fft
        self.student_n_mels = student_n_mels

        self.speaker_embed_dim = speaker_embed_dim
        self.speaker_embed_dir = speaker_embed_dir
        self.ecapa_model_path = ecapa_model_path
        self.ecapa_input_mels = ecapa_input_mels
        self.ecapa_sampling_rate = ecapa_sampling_rate
        self.ecapa_n_fft = ecapa_n_fft
        self.ecapa_hop_length = ecapa_hop_length
        self.ecapa_win_length = ecapa_win_length if ecapa_win_length is not None else ecapa_n_fft
        self.ecapa_mel_fmin = ecapa_mel_fmin
        self.ecapa_mel_fmax = ecapa_mel_fmax
        self.device = device

        self.wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])

        self.student_mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.student_sampling_rate,
            n_fft=self.student_n_fft,
            hop_length=self.student_hop_length,
            win_length=self.student_win_length,
            n_mels=self.student_n_mels
        ).to(self.device)

        self.speaker_encoder = ECAPA_TDNN(input_size=self.ecapa_input_mels, lin_neurons=self.speaker_embed_dim).to(self.device)
        if self.ecapa_model_path and os.path.exists(self.ecapa_model_path):
            try:
                self.speaker_encoder.load_state_dict(torch.load(self.ecapa_model_path, map_location=self.device))
                print(f"INFO: Loaded pre-trained ECAPA_TDNN from {self.ecapa_model_path}")
            except Exception as e:
                print(f"WARN: Error loading ECAPA_TDNN weights from {self.ecapa_model_path}: {e}. Using random init.")
        else:
            if self.ecapa_model_path: # Only warn if a path was given but not found
                 print(f"WARN: ECAPA_TDNN model path {self.ecapa_model_path} not found. Using random init for speaker encoder.")
            else:
                 print("INFO: No ECAPA_TDNN model path specified. Using random init for speaker encoder.")
        self.speaker_encoder.eval()

        self.ecapa_mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.ecapa_sampling_rate,
            n_fft=self.ecapa_n_fft,
            hop_length=self.ecapa_hop_length,
            win_length=self.ecapa_win_length,
            n_mels=self.ecapa_input_mels,
            f_min=self.ecapa_mel_fmin,
            f_max=self.ecapa_mel_fmax
        ).to(self.device)

        self.mock_embed_generator = torch.nn.Linear(1, speaker_embed_dim).to(self.device)


    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_filename = self.wav_files[idx]
        wav_path = os.path.join(self.wav_dir, wav_filename)

        d_vector = None

        # Priority 1: Load pre-computed speaker embedding
        if self.speaker_embed_dir:
            # Use splitext for robustness with filenames containing dots
            base_filename = os.path.splitext(wav_filename)[0]
            embed_path = os.path.join(self.speaker_embed_dir, f"{base_filename}.pt")
            if os.path.exists(embed_path):
                try:
                    d_vector = torch.load(embed_path, map_location=torch.device('cpu'))
                    if not isinstance(d_vector, torch.Tensor) or d_vector.ndim != 1 or d_vector.shape[0] != self.speaker_embed_dim:
                        print(f"WARN: Loaded speaker embedding {embed_path} has incorrect dimension or type. Expected ({self.speaker_embed_dim},). Got {d_vector.shape if isinstance(d_vector, torch.Tensor) else type(d_vector)}. Ignoring.")
                        d_vector = None
                except Exception as e:
                    print(f"WARN: Error loading speaker embedding {embed_path}: {e}")
                    d_vector = None

        # Load audio (original_audio_tensor)
        try:
            original_audio_tensor, current_sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"ERROR: Error loading wav file {wav_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.wav_files))

        # Ensure audio is on device for transforms
        original_audio_tensor_on_device = original_audio_tensor.to(self.device)
        if original_audio_tensor_on_device.shape[0] > 1: # Ensure mono for ECAPA and student processing
            original_audio_tensor_on_device = torch.mean(original_audio_tensor_on_device, dim=0, keepdim=True)

        # Prepare audio for student (segmentation, resampling if needed)
        audio_for_student = original_audio_tensor_on_device
        if current_sr != self.student_sampling_rate:
            audio_for_student = torchaudio.functional.resample(audio_for_student, current_sr, self.student_sampling_rate)

        audio_segment_student = audio_for_student.squeeze(0) # Remove channel dim for 1D operations
        if audio_segment_student.size(0) >= self.segment_size:
            start = random.randint(0, audio_segment_student.size(0) - self.segment_size)
            audio_segment_student = audio_segment_student[start:start+self.segment_size]
        else:
            audio_segment_student = F.pad(audio_segment_student, (0, self.segment_size - audio_segment_student.size(0)), "constant")

        mel_student = self.student_mel_transform(audio_segment_student.unsqueeze(0)).squeeze(0) # Add batch dim for transform, then remove

        # Priority 2: Generate speaker embedding on-the-fly if not loaded
        if d_vector is None and self.speaker_encoder:
            try:
                audio_for_ecapa = original_audio_tensor_on_device # Start with mono audio on device
                if current_sr != self.ecapa_sampling_rate:
                    # Resample expects (..., time), so if (1, time), it's fine.
                    audio_for_ecapa = torchaudio.functional.resample(audio_for_ecapa, current_sr, self.ecapa_sampling_rate)

                min_len_for_ecapa_mel = self.ecapa_n_fft
                if audio_for_ecapa.shape[-1] < min_len_for_ecapa_mel: # input to STFT must be at least n_fft
                     audio_for_ecapa = F.pad(audio_for_ecapa, (0, min_len_for_ecapa_mel - audio_for_ecapa.shape[-1]), "reflect")

                mel_for_ecapa = self.ecapa_mel_transform(audio_for_ecapa) # Expects (B, T) or (T) -> (B, n_mels, time)

                with torch.no_grad():
                    d_vector = self.speaker_encoder(mel_for_ecapa).squeeze(0).squeeze(-1).cpu() # ECAPA outputs (B, C, 1) -> (C)
            except Exception as e:
                print(f"WARN: ECAPA_TDNN failed to generate speaker embedding for {wav_path}: {e}")
                d_vector = None

        # Priority 3: Mock embedding (fallback)
        if d_vector is None:
            print(f"WARN: Using mock speaker embedding for {wav_path}")
            with torch.no_grad():
                d_vector = self.mock_embed_generator(torch.ones(1, device=self.device)).squeeze(0).cpu()

        return {
            "mel": mel_student.cpu(),
            "audio": audio_segment_student.cpu(),
            "d_vector": d_vector.cpu() # Ensure (speaker_embed_dim,)
        }
