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

from indextts.distill-utils import Generator44kHzWithSpeaker
from indextts.BigVGAN.models import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from indextts.BigVGAN.models import discriminator_loss as bigvgan_discriminator_loss
from indextts.BigVGAN.models import feature_loss as bigvgan_feature_loss
from indextts.BigVGAN.models import generator_loss as bigvgan_generator_loss

from utils.audio import MelSpectrogramLoss, STFTLoss
from indextts.distill-utils import AudioMelDataset # Updated AudioMelDataset
from utils.logger import setup_logger

# Helper HParams Class for BigVGAN Discriminators
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if 'use_spectral_norm' not in kwargs:
            self.use_spectral_norm = False
        if 'mrd_use_spectral_norm' not in kwargs:
            self.mrd_use_spectral_norm = self.use_spectral_norm
        if 'discriminator_channel_mult' not in kwargs:
            self.discriminator_channel_mult = 1
        if 'mrd_channel_mult' not in kwargs:
            self.mrd_channel_mult = self.discriminator_channel_mult

    def get(self, key, default=None):
        return getattr(self, key, default)

# BigVGAN Discriminator Wrapper
class BigVGANDiscriminatorWrapper(nn.Module):
    def __init__(self, hparams_config_dict):
        super().__init__()
        h = HParams(**hparams_config_dict)
        self.mpd = MultiPeriodDiscriminator(h)
        self.mrd = MultiResolutionDiscriminator(cfg=h)

    def forward(self, y, y_hat):
        y_d_rs_mpd, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = self.mpd(y, y_hat)
        y_d_rs_mrd, y_d_gs_mrd, fmap_rs_mrd, fmap_gs_mrd = self.mrd(y, y_hat)

        y_d_rs = y_d_rs_mpd + y_d_rs_mrd
        y_d_gs = y_d_gs_mpd + y_d_gs_mrd
        fmap_rs = fmap_rs_mpd + fmap_rs_mrd
        fmap_gs = fmap_gs_mpd + fmap_gs_mrd

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# --- Configuration ---
config = {
    # Training Hyperparameters
    "epochs": 3000,
    "batch_size": 16,
    "lr": 2e-4,
    "adam_betas": (0.8, 0.99),
    "segment_size": 16384,
    "speaker_embed_dim": 512, # Dimension of speaker embeddings

    # Model Checkpoint Paths
    "teacher_ckpt": "path/to/your/pretrained_bigvgan_24khz_teacher.pth", # USER: MUST SET THIS
    "student_generator_ckpt": "checkpoints_distill_bigvgan_speaker/student_generator.pth",
    "student_discriminator_ckpt": "checkpoints_distill_bigvgan_speaker/student_discriminator.pth",
    "out_dir": "checkpoints_distill_bigvgan_speaker", # Output directory

    # Student Model & Data Parameters (44.1kHz)
    "sampling_rate": 44100,      # student_sampling_rate
    "student_mel_bands": 128,
    "student_n_fft": 2048,
    "student_hop_length": 512,
    "student_win_length": 2048,

    # Teacher Model & Data Parameters (24kHz)
    "sampling_rate_teacher": 24000,
    "teacher_mel_bands": 100,
    "teacher_n_fft": 1024,
    "teacher_hop_length": 256,
    "teacher_win_length": 1024,

    # Speaker Embedding Extraction (ECAPA_TDNN) Parameters
    "speaker_embed_dir": "data/speaker_embeddings/", # USER: Path to pre-computed speaker embeddings
    "ecapa_model_path": "path/to/your/ecapa_tdnn.pth", # USER: Path to pre-trained ECAPA_TDNN model (optional)
    "ecapa_sampling_rate": 44100, # SR for audio fed to ECAPA (likely same as student SR)
    "ecapa_mel_bands": 128,       # Mel bands for ECAPA input (e.g., 80 or 128)
    "ecapa_n_fft": 2048,          # FFT for ECAPA mels
    "ecapa_hop_length": 512,      # Hop for ECAPA mels
    "ecapa_win_length": 2048,     # Win for ECAPA mels

    # Logging and Saving Intervals
    "log_interval": 100,
    "save_interval": 1000,

    # Loss Weights
    "loss_weights": {
        "teacher": 1.0, "mel": 45.0, "stft_sc": 0.5, "stft_mag": 0.5,
        "adv_g": 1.0, "fm": 2.0,
    },

    # Discriminator Hyperparameters
    "discriminator_hparams": {
        "mpd_reshapes": [2, 3, 5, 7, 11, 17, 23, 31],
        "resolutions": [[1024,120,600], [2048,240,1200], [512,60,300]],
        "use_spectral_norm": False, "discriminator_channel_mult": 1,
        "mrd_use_spectral_norm": False, "mrd_channel_mult": 1,
    }
}

# --- Setup ---
logger = setup_logger(os.path.basename(config["out_dir"]), config["out_dir"])
os.makedirs(config["out_dir"], exist_ok=True)
os.makedirs(config["speaker_embed_dir"], exist_ok=True) # Ensure speaker embed dir exists
torch.manual_seed(1234)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Teacher Model ---
teacher = None
# (Teacher loading logic remains the same placeholder - user needs to implement)
if config["teacher_ckpt"] and os.path.exists(config["teacher_ckpt"]) and config["teacher_ckpt"] != "path/to/your/pretrained_bigvgan_24khz_teacher.pth":
    try:
        # USER: Implement actual teacher model loading here.
        logger.info(f"Attempting to load Teacher model from {config['teacher_ckpt']}...")
        # ... (actual loading code) ...
        # teacher.eval()
        # logger.info(f"Teacher model loaded successfully from {config['teacher_ckpt']}")
        pass
    except Exception as e:
        logger.error(f"Failed to load teacher model from {config['teacher_ckpt']}: {e}. Using dummy.")
        teacher = None
else:
    logger.warning(f"Teacher checkpoint {config['teacher_ckpt']} not found, not specified, or is default placeholder. Using dummy teacher.")

if teacher is None:
    class DummyTeacher(nn.Module):
        def __init__(self, target_sr, segment_size, original_sr):
            super().__init__()
            self.target_len = segment_size * target_sr // original_sr
        def forward(self, mel, d_vector):
            return torch.zeros(mel.shape[0], 1, self.target_len, device=mel.device)
    teacher = DummyTeacher(config["sampling_rate_teacher"], config["segment_size"], config["sampling_rate"]).to(device)
    logger.info("Using dummy teacher model.")

# --- Student Models ---
student_generator = Generator44kHzWithSpeaker(
    mel_dim=config["student_mel_bands"],
    speaker_embed_dim=config["speaker_embed_dim"]
).to(device)
student_discriminator = BigVGANDiscriminatorWrapper(config["discriminator_hparams"]).to(device)

# --- Optimizers ---
optim_g = torch.optim.AdamW(student_generator.parameters(), lr=config["lr"], betas=config["adam_betas"])
optim_d = torch.optim.AdamW(student_discriminator.parameters(), lr=config["lr"], betas=config["adam_betas"])

# --- Load Checkpoints if exist ---
start_epoch = 0
global_step = 0
# (Checkpoint loading logic remains the same)
if os.path.exists(config["student_generator_ckpt"]):
    try:
        logger.info(f"Loading student generator checkpoint: {config['student_generator_ckpt']}")
        checkpoint_g = torch.load(config["student_generator_ckpt"], map_location=device)
        student_generator.load_state_dict(checkpoint_g['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint_g: optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        start_epoch = checkpoint_g.get('epoch', 0)
        global_step = checkpoint_g.get('step', 0)
    except Exception as e:
        logger.error(f"Error loading student generator checkpoint: {e}. Starting from scratch.")
        start_epoch = 0; global_step = 0
else:
    logger.info("Student generator checkpoint not found. Starting from scratch.")

if os.path.exists(config["student_discriminator_ckpt"]):
    try:
        logger.info(f"Loading student discriminator checkpoint: {config['student_discriminator_ckpt']}")
        checkpoint_d = torch.load(config["student_discriminator_ckpt"], map_location=device)
        student_discriminator.load_state_dict(checkpoint_d['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint_d: optim_d.load_state_dict(checkpoint_d['optimizer_state_dict'])
    except Exception as e:
        logger.error(f"Error loading student discriminator checkpoint: {e}. Starting from scratch.")
else:
    logger.info("Student discriminator checkpoint not found. Starting from scratch.")

# --- Dataset & DataLoader ---
# Note: AudioMelDataset now requires ECAPA related params
train_dataset = AudioMelDataset(
    wav_dir="data/wavs_44k", # USER: Ensure this path exists
    # mel_dir is not used by the current AudioMelDataset for student mels
    segment_size=config["segment_size"],
    # Student mel params
    student_sampling_rate=config["sampling_rate"],
    student_hop_length=config["student_hop_length"],
    student_n_fft=config["student_n_fft"],
    student_win_length=config["student_win_length"],
    student_n_mels=config["student_mel_bands"],
    # Speaker embedding params
    speaker_embed_dim=config["speaker_embed_dim"],
    speaker_embed_dir=config["speaker_embed_dir"],
    ecapa_model_path=config["ecapa_model_path"],
    ecapa_input_mels=config["ecapa_mel_bands"],
    ecapa_sampling_rate=config["ecapa_sampling_rate"],
    ecapa_n_fft=config["ecapa_n_fft"],
    ecapa_hop_length=config["ecapa_hop_length"],
    ecapa_win_length=config["ecapa_win_length"],
    device=device # Pass device for on-the-fly ECAPA processing
)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

# --- Loss Functions ---
mel_loss_fn = MelSpectrogramLoss(
    sample_rate=config["sampling_rate"], n_mels=config["student_mel_bands"],
    n_fft=config["student_n_fft"], hop_length=config["student_hop_length"],
    win_length=config["student_win_length"], normalized=True
).to(device)
stft_loss_fn = STFTLoss(
    fft_size=config["student_n_fft"], hop_size=config["student_hop_length"],
    win_length=config["student_win_length"]
).to(device)
l1_loss = nn.L1Loss().to(device)
mel_teacher_extractor = torchaudio.transforms.MelSpectrogram(
    sample_rate=config["sampling_rate_teacher"], n_mels=config["teacher_mel_bands"],
    n_fft=config["teacher_n_fft"], hop_length=config["teacher_hop_length"],
    win_length=config["teacher_win_length"]
).to(device)

# --- Training Loop ---
logger.info(f"Starting training from epoch {start_epoch} and step {global_step}")
# (Training loop logic remains largely the same as per previous refactoring)
for epoch in range(start_epoch, config["epochs"]):
    for batch_idx, batch in enumerate(train_loader):
        # d_vector is now handled by AudioMelDataset, potentially generated by ECAPA
        mel_student_input, audio_44k_gt, d_vector = batch["mel"].to(device), batch["audio"].to(device), batch["d_vector"].to(device)
        audio_44k_gt = audio_44k_gt.unsqueeze(1)

        student_generator.train()
        optim_g.zero_grad()

        y_hat_44k = student_generator(mel_student_input, d_vector)
        y_hat_24k_from_student = torchaudio.functional.resample(
            y_hat_44k, orig_freq=config["sampling_rate"], new_freq=config["sampling_rate_teacher"]
        )

        with torch.no_grad():
            audio_24k_for_teacher_mel = torchaudio.functional.resample(
                audio_44k_gt.squeeze(1), orig_freq=config["sampling_rate"], new_freq=config["sampling_rate_teacher"]
            )
            mel_for_teacher = mel_teacher_extractor(audio_24k_for_teacher_mel)
            # Teacher model should be able to accept d_vector if it's a modified BigVGAN
            y_teacher_24k = teacher(mel_for_teacher.to(device), d_vector.to(device))


        loss_teacher = l1_loss(y_hat_24k_from_student, y_teacher_24k.detach())
        loss_mel_student = mel_loss_fn(y_hat_44k.squeeze(1), audio_44k_gt.squeeze(1))
        loss_stft_sc_student, loss_stft_mag_student = stft_loss_fn(y_hat_44k.squeeze(1), audio_44k_gt.squeeze(1))

        y_d_rs, y_d_gs, fmap_rs, fmap_gs = student_discriminator(audio_44k_gt, y_hat_44k)
        loss_g_adv, _ = bigvgan_generator_loss(y_d_gs)
        loss_fm = bigvgan_feature_loss(fmap_rs, fmap_gs)

        lw = config["loss_weights"]
        loss_g = (lw["teacher"] * loss_teacher +
                    lw["mel"] * loss_mel_student +
                    lw["stft_sc"] * loss_stft_sc_student +
                    lw["stft_mag"] * loss_stft_mag_student +
                    lw["adv_g"] * loss_g_adv +
                    lw["fm"] * loss_fm)
        loss_g.backward()
        optim_g.step()

        optim_d.zero_grad()
        y_d_rs_d, y_d_gs_d, _, _ = student_discriminator(audio_44k_gt, y_hat_44k.detach())
        loss_d, _, _ = bigvgan_discriminator_loss(y_d_rs_d, y_d_gs_d)
        loss_d.backward()
        optim_d.step()

        if global_step % config["log_interval"] == 0:
            logger.info(f"Epoch {epoch}, Step {global_step}, Batch {batch_idx}/{len(train_loader)}, G_Loss: {loss_g.item():.4f}, D_Loss: {loss_d.item():.4f}, Mel: {loss_mel_student.item():.4f}, SC: {loss_stft_sc_student.item():.4f}, Mag: {loss_stft_mag_student.item():.4f}, G_Adv: {loss_g_adv.item():.4f}, FM: {loss_fm.item():.4f}, Teacher_L1: {loss_teacher.item():.4f}")

        if global_step > 0 and global_step % config["save_interval"] == 0:
            g_ckpt_path = os.path.join(config["out_dir"], f"student_generator_step_{global_step}.pth")
            torch.save({
                'epoch': epoch, 'step': global_step,
                'model_state_dict': student_generator.state_dict(),
                'optimizer_state_dict': optim_g.state_dict(),
            }, g_ckpt_path)
            d_ckpt_path = os.path.join(config["out_dir"], f"student_discriminator_step_{global_step}.pth")
            torch.save({
                'epoch': epoch, 'step': global_step,
                'model_state_dict': student_discriminator.state_dict(),
                'optimizer_state_dict': optim_d.state_dict(),
            }, d_ckpt_path)
            logger.info(f"Saved checkpoints at step {global_step}")
            # Update main checkpoint paths to point to these step checkpoints for easy resume
            # Or save to config["student_generator_ckpt"] directly if preferred for simplicity
            # For now, keeping main ckpt paths for final/manual saves.

        global_step += 1
    logger.info(f"Epoch {epoch} completed.")

logger.info("Training finished.")
final_g_ckpt_path = os.path.join(config["out_dir"], config["student_generator_ckpt"].split('/')[-1]) # Use base name
final_d_ckpt_path = os.path.join(config["out_dir"], config["student_discriminator_ckpt"].split('/')[-1]) # Use base name

torch.save({'epoch': config["epochs"] -1, 'step': global_step, 'model_state_dict': student_generator.state_dict(), 'optimizer_state_dict': optim_g.state_dict()}, final_g_ckpt_path)
torch.save({'epoch': config["epochs"] -1, 'step': global_step, 'model_state_dict': student_discriminator.state_dict(), 'optimizer_state_dict': optim_d.state_dict()}, final_d_ckpt_path)
logger.info(f"Saved final models (with optimizer states): {final_g_ckpt_path}, {final_d_ckpt_path}")

torch.save(student_generator.state_dict(), final_g_ckpt_path.replace(".pth", "_weights.pth"))
torch.save(student_discriminator.state_dict(), final_d_ckpt_path.replace(".pth", "_weights.pth"))
logger.info(f"Saved final model weights (state_dict only) for inference.")
