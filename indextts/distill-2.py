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

from indextts.distill-utils import Generator44kHzWithSpeaker  # 修改版 Generator（包含音色嵌入）
from models.discriminator import DiscriminatorMultiRes
from utils.audio import MelSpectrogramLoss, STFTLoss
from utils.vocoder import PQMF
from indextts.distill-utils import AudioMelDataset  # 自定义数据集类
from utils.logger import setup_logger

# 配置参数（建议移至配置文件）
config = {
    "batch_size": 16,
    "lr": 2e-4,
    "epochs": 300,
    "segment_size": 65536,
    "mel_dim": 100,
    "mel_dim_target": 128,
    "speaker_embed_dim": 512,
    "sampling_rate": 44100,
    "sampling_rate_teacher": 24000,
    "log_interval": 100,
    "save_interval": 1000,
    "out_dir": "checkpoints_distill",
    "teacher_ckpt": "bigvgan_generator.pth",
    "student_ckpt": "bigvgan_generator-44k.pth",
    "teacher_mel_hop": 256,
    "student_mel_hop": 512,
    "loss_weights": {
        "teacher": 1.0,
        "mel": 15.0,
        "stft": 1.0,
        "mag": 1.0,
        "adv": 1.0,
        "fm": 2.0
    }
}

logger = setup_logger("distill", config["out_dir"])

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
    speaker_embed_dim=config["speaker_embed_dim"]
).to(device)

if os.path.exists(config["student_ckpt"]):
    print(f"Loading student checkpoint from {config['student_ckpt']}")
    student.load_state_dict(torch.load(config["student_ckpt"], map_location=device))

# 判别器
discriminator = DiscriminatorMultiRes().to(device)

# ==== 2. 准备数据集 ====
train_dataset = AudioMelDataset(
    wav_dir="data/wavs",
    mel_dir="data/mels",
    segment_size=config["segment_size"],
    hop_length=config["student_mel_hop"],
    sampling_rate=config["sampling_rate"],
    speaker_embed_dim=config["speaker_embed_dim"]
)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

# ==== 3. 损失函数 ====
stft_loss = STFTLoss()
mel_loss = MelSpectrogramLoss(sr=config["sampling_rate"], n_mels=config["mel_dim_target"])

mse = nn.MSELoss()
l1 = nn.L1Loss()

mel_teacher_extractor = torchaudio.transforms.MelSpectrogram(
    sample_rate=config["sampling_rate_teacher"], n_fft=1024, hop_length=config["teacher_mel_hop"], n_mels=config["mel_dim"]
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
        loss_teacher = l1(y_hat_down, y_teacher.detach())
        loss_stft, loss_mag = stft_loss(y_hat, audio)
        loss_mel = mel_loss(y_hat, audio)

        # === Adversarial Loss ===
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(audio, y_hat)
        loss_g_adv = sum([mse(y, torch.ones_like(y)) for y in y_d_hat_g])

        # === Feature Matching ===
        loss_fm = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss_fm += l1(rl, gl)

        lw = config["loss_weights"]
        loss_g = lw["teacher"] * loss_teacher + lw["mel"] * loss_mel + lw["stft"] * loss_stft + lw["mag"] * loss_mag + lw["adv"] * loss_g_adv + lw["fm"] * loss_fm
        loss_g.backward()
        optim_g.step()

        # ==== Discriminator ====
        optim_d.zero_grad()
        y_d_hat_r, y_d_hat_g, _, _ = discriminator(audio, y_hat.detach())

        loss_d_real = sum([mse(y, torch.ones_like(y)) for y in y_d_hat_r])
        loss_d_fake = sum([mse(y, torch.zeros_like(y)) for y in y_d_hat_g])
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optim_d.step()

        # ==== Logging ====
        if step % config["log_interval"] == 0:
            logger.info(f"Step {step}, G_loss: {loss_g.item():.3f}, D_loss: {loss_d.item():.3f}")

        if step % config["save_interval"] == 0:
            ckpt_path = os.path.join(config["out_dir"], f"student_{step}.pth")
            torch.save(student.state_dict(), ckpt_path)

        step += 1
