#!/usr/bin/env python
# coding: utf-8

"""
BigVGAN 24kHz -> 44kHz 蒸馏训练脚本
本脚本实现基于模型A（24kHz多音色）蒸馏模型B（44kHz高保真），迁移音色能力。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm, remove_weight_norm

from indextts.distill_utils import Generator44kHzWithSpeaker, AudioMelDataset # Combined and corrected import
from models.discriminator import DiscriminatorMultiRes
from indextts.custom_audio_utils import MelSpectrogramLoss, STFTLoss
from utils.vocoder import PQMF
# AudioMelDataset also imported above from indextts.distill_utils
import logging

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

# 配置参数（建议移至配置文件）
config = {
    "batch_size": 16,
    "lr": 2e-4,
    "epochs": 300,
    "segment_size": 65536, # For student, 44.1kHz

    "speaker_embed_dim": 512,

    # Student Model & Data Parameters (44.1kHz)
    "sampling_rate": 44100,
    "student_mel_bands": 128,    # Number of mel bands for student input/output
    "student_n_fft": 2048,       # FFT size for student mel spectrograms
    "student_hop_length": 512,   # Hop length for student mel spectrograms (config["student_mel_hop"])
    "student_win_length": 2048,  # Window length for student mel spectrograms
    "student_mel_fmin": 0.0,
    "student_mel_fmax": None, # Can be sampling_rate / 2
    "student_mel_power": 1.0, # Energy
    "student_mel_normalized": False, # Typically False for torchaudio default, but can be True
    "student_mel_center": True,
    "student_mel_pad_mode": "reflect",

    # STFT Loss parameters (for student)
    "stft_window_fn": "torch.hann_window", # As string, to be resolved
    "stft_center": True,
    "stft_pad_mode": "reflect",

    # Teacher Model & Data Parameters (24kHz)
    "sampling_rate_teacher": 24000,
    "teacher_mel_bands": 100,    # (config["mel_dim"])
    "teacher_n_fft": 1024,       # FFT size for teacher mel spectrograms
    "teacher_hop_length": 256,   # (config["teacher_mel_hop"])
    "teacher_win_length": 1024,  # Window length for teacher mel spectrograms

    "log_interval": 100,
    "save_interval": 1000,
    "out_dir": "checkpoints_distill",
    "teacher_ckpt": "bigvgan_generator.pth",
    "student_ckpt": "bigvgan_generator-44k.pth", # This is the old student (not BigVGAN based)

    "loss_weights": {
        "teacher": 1.0,     # L1 loss between student (downsampled) and teacher output
        "mel": 45.0,        # MelSpectrogram loss for student's 44.1kHz output (BigVGAN uses 45)
        "stft_sc": 0.5,     # STFT Spectral Convergence loss weight
        "stft_mag": 0.5,    # STFT Log Magnitude loss weight
        "adv_g": 1.0,       # Adversarial loss for the generator (placeholder, as old D is used)
        "fm": 2.0           # Feature Matching loss (placeholder)
    }
}

# logger = setup_logger("distill", config["out_dir"]) # Removed custom logger

torch.manual_seed(1234)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 1. 加载模型 ====
def load_teacher_model(path):
    model = torch.load(path, map_location=device)["generator"]
    return model.eval().to(device)

teacher = load_teacher_model(config["teacher_ckpt"])

student = Generator44kHzWithSpeaker(
    mel_dim=config["mel_dim"],
    speaker_embed_dim=config["speaker_embed_dim"],
    # The Generator44kHzWithSpeaker in this old version does not take BigVGAN hparams.
    # It expects mel_dim (which is teacher_mel_bands for its pre_conv)
    mel_dim=config["teacher_mel_bands"]
).to(device)

if os.path.exists(config["student_ckpt"]):
    print(f"Loading student checkpoint from {config['student_ckpt']}")
    student.load_state_dict(torch.load(config["student_ckpt"], map_location=device))

# 判别器 (Old discriminator, not BigVGAN based)
discriminator = DiscriminatorMultiRes().to(device)

# ==== 2. 准备数据集 ====
# AudioMelDataset in this old version has a different signature
train_dataset = AudioMelDataset(
    wav_dir="data/wavs",
    mel_dir="data/mels", # This implies loading mels, not on-the-fly generation for student
    segment_size=config["segment_size"],
    hop_length=config["student_hop_length"], # Used by dataset if it generates mels
    sampling_rate=config["sampling_rate"],
    speaker_embed_dim=config["speaker_embed_dim"]
    # Add other params if the old AudioMelDataset was updated to accept them for student mels
    # For now, assuming it uses its internal hardcoded mel params or loads them.
    # This part highlights the inconsistency due to file reversion.
)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

# ==== 3. 损失函数 ====
# Resolve window function for STFTLoss
stft_window_fn_str = config.get("stft_window_fn", "torch.hann_window")
if hasattr(torch, stft_window_fn_str.split('.')[-1]): # Simplistic check
    stft_window_constructor = getattr(torch, stft_window_fn_str.split('.')[-1])
else:
    stft_window_constructor = torch.hann_window # Default
    logging.warning(f"STFT window function {stft_window_fn_str} not found. Using torch.hann_window.")

stft_loss_fn = STFTLoss(
    n_fft=config["student_n_fft"],
    hop_length=config["student_hop_length"],
    win_length=config["student_win_length"],
    window_fn_constructor=stft_window_constructor,
    center=config.get("stft_center", True),
    pad_mode=config.get("stft_pad_mode", "reflect")
).to(device)

mel_loss_fn = MelSpectrogramLoss(
    sample_rate=config["sampling_rate"],
    n_fft=config["student_n_fft"],
    hop_length=config["student_hop_length"],
    win_length=config["student_win_length"],
    n_mels=config["student_mel_bands"],
    mel_fmin=config.get("student_mel_fmin", 0.0),
    mel_fmax=config.get("student_mel_fmax", None),
    power=config.get("student_mel_power", 1.0),
    normalized=config.get("student_mel_normalized", False),
    center=config.get("student_mel_center", True),
    pad_mode=config.get("student_mel_pad_mode", "reflect")
).to(device)

mse = nn.MSELoss() # For old discriminator loss
l1_loss = nn.L1Loss() # For teacher L1 loss

mel_teacher_extractor = torchaudio.transforms.MelSpectrogram(
    sample_rate=config["sampling_rate_teacher"],
    n_fft=config["teacher_n_fft"],
    hop_length=config["teacher_hop_length"],
    n_mels=config["teacher_mel_bands"]
).to(device)

# ==== 4. 优化器 ====
optim_g = torch.optim.Adam(student.parameters(), lr=config["lr"], betas=(0.8, 0.99))
optim_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(0.8, 0.99))

# ==== 5. 蒸馏训练 ====
step = 0
for epoch in range(config["epochs"]):
    for batch in train_loader:
        mel, audio, d_vector = batch["mel"].to(device), batch["audio"].to(device), batch["d_vector"].to(device)

        # ==== Generator ====
        student.train()
        optim_g.zero_grad()

        y_hat = student(mel, d_vector)

        # === Downsample student output to 24kHz ===
        y_hat_down = torchaudio.functional.resample(y_hat, config["sampling_rate"], config["sampling_rate_teacher"])

        with torch.no_grad():
            y_teacher = teacher(mel_teacher_extractor(audio), d_vector)

        # === Losses ===
        loss_teacher = l1_loss(y_hat_down, y_teacher.detach()) # Use l1_loss

        # Student spectral losses (using new loss functions)
        # Ensure y_hat and audio are (B, T) for these losses
        loss_mel_student = mel_loss_fn(y_hat.squeeze(1), audio.squeeze(1))
        loss_stft_sc_student, loss_stft_mag_student = stft_loss_fn(y_hat.squeeze(1), audio.squeeze(1))

        # === Adversarial Loss (using old discriminator) ===
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(audio, y_hat) # audio is (B,1,T), y_hat is (B,1,T)
        loss_g_adv = sum([mse(y_g, torch.ones_like(y_g)) for y_g in y_d_hat_g]) # mse already instantiated

        # === Feature Matching (using old discriminator) ===
        loss_fm = 0
        for dr_list, dg_list in zip(fmap_r, fmap_g): # fmap_r/g are lists of lists of tensors
            for dr, dg in zip(dr_list, dg_list):
                loss_fm += l1_loss(dg, dr) # Use l1_loss; order for FM is typically G vs R

        lw = config["loss_weights"]
        # Note: lw["adv"] and lw["fm"] are now placeholders if using BigVGAN D losses
        # but here we use the old discriminator's output with new G losses.
        # The keys in loss_weights should match: "mel", "stft_sc", "stft_mag", "adv_g", "fm"
        loss_g = (lw.get("teacher", 1.0) * loss_teacher +
                  lw.get("mel", 1.0) * loss_mel_student +
                  lw.get("stft_sc", 1.0) * loss_stft_sc_student +
                  lw.get("stft_mag", 1.0) * loss_stft_mag_student +
                  lw.get("adv_g", 1.0) * loss_g_adv + # Renamed from "adv"
                  lw.get("fm", 1.0) * loss_fm)
        loss_g.backward()
        optim_g.step()

        # ==== Discriminator ====
        optim_d.zero_grad()
        # Discriminator sees real audio and detached generator output
        y_d_r_out, y_d_g_out, _, _ = discriminator(audio, y_hat.detach())

        loss_d_real = sum([mse(y_r, torch.ones_like(y_r)) for y_r in y_d_r_out])
        loss_d_fake = sum([mse(y_g, torch.zeros_like(y_g)) for y_g in y_d_g_out])
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optim_d.step()

        # ==== Logging ====
        if step % config["log_interval"] == 0:
            logging.info(f"Step {step}, G_loss: {loss_g.item():.3f}, D_loss: {loss_d.item():.3f}")

        if step % config["save_interval"] == 0:
            ckpt_path = os.path.join(config["out_dir"], f"student_{step}.pth")
            torch.save(student.state_dict(), ckpt_path)

        step += 1
