#!/usr/bin/env python
# coding: utf-8

"""
BigVGAN 24kHz -> 44kHz 蒸馏训练脚本
"""

import os
from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from distill_utils import Generator44kHzWithSpeaker, AudioMelDataset
from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.BigVGAN.models import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from indextts.BigVGAN.models import discriminator_loss as bigvgan_discriminator_loss
from indextts.BigVGAN.models import feature_loss as bigvgan_feature_loss
from indextts.BigVGAN.models import generator_loss as bigvgan_generator_loss
from indextts.custom_audio_utils import MelSpectrogramLoss, STFTLoss

import logging

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging # Alias for convenience

# BigVGAN Discriminator Wrapper
class BigVGANDiscriminatorWrapper(nn.Module):
    def __init__(self, hparams_obj):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(hparams_obj)
        self.mrd = MultiResolutionDiscriminator(cfg=hparams_obj)

    def forward(self, y, y_hat):
        y_d_rs_mpd, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = self.mpd(y, y_hat)
        y_d_rs_mrd, y_d_gs_mrd, fmap_rs_mrd, fmap_gs_mrd = self.mrd(y, y_hat)

        all_y_d_rs = y_d_rs_mpd + y_d_rs_mrd
        all_y_d_gs = y_d_gs_mpd + y_d_gs_mrd
        all_fmap_rs = fmap_rs_mpd + fmap_rs_mrd
        all_fmap_gs = fmap_gs_mpd + fmap_gs_mrd

        return all_y_d_rs, all_y_d_gs, all_fmap_rs, all_fmap_gs

# --- Configuration ---
config = {
    # Training Hyperparameters
    "epochs": 100,
    "batch_size": 16,
    "lr_g": 2e-4, # Learning rate for generator
    "lr_d": 2e-4, # Learning rate for discriminator
    "adam_betas_g": (0.8, 0.99),
    "adam_betas_d": (0.8, 0.99),
    "segment_size": 16384,
    "speaker_embed_dim": 512,

    # Model Checkpoint Paths
    "teacher_ckpt": "checkpoints/bigvgan_generator.pth",
    "student_ckpt": "checkpoints/bigvgan_generator-44k.pth",
    "discriminator_ckpt": "checkpoints/bigvgan_discriminator-44k.pth",
    "out_dir": "checkpoints_distill",

    # Student Model & Data Parameters (44.1kHz)
    "sampling_rate": 44100,
    "student_input_mel_bands": 128,
    "student_n_fft": 2048,
    "student_hop_length": 512,
    "student_win_length": 2048,
    "student_mel_fmin": 0.0,
    "student_mel_fmax": None,
    "student_mel_power": 1.0,
    "student_mel_normalized": False,
    "student_mel_center": True,
    "student_mel_pad_mode": "reflect",

    "bigvgan_student_hparams": {
        "upsample_rates": [8,8,2,2,2], "upsample_kernel_sizes": [16,16,4,4,4],
        "upsample_initial_channel": 512, "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]], "activation": "snake", "resblock": "1",
        "feat_upsample": False, "cond_d_vector_in_each_upsampling_layer": True,
        "use_cuda_kernel": False, "snake_logscale": True
    },

    "stft_window_fn": "torch.hann_window", "stft_center": True, "stft_pad_mode": "reflect",

    "sampling_rate_teacher": 24000, "teacher_mel_bands": 100,
    "teacher_n_fft": 1024, "teacher_hop_length": 256, "teacher_win_length": 1024,

    "speaker_embed_dir": "data/speaker_embeddings/",
    "ecapa_model_path": "path/to/your/ecapa_tdnn.pth",
    "ecapa_input_mels": 80, "ecapa_sampling_rate": 24000,
    "ecapa_n_fft": 1024, "ecapa_hop_length": 256, "ecapa_win_length": 1024,
    "ecapa_mel_fmin": 0.0, "ecapa_mel_fmax": None,

    "log_interval": 100, "save_interval": 1000,

    "loss_weights": {
        "teacher": 1.0, "mel": 45.0, "stft_sc": 0.5, "stft_mag": 0.5,
        "adv_g": 1.0, "fm": 2.0,
    },
    "discriminator_hparams": {
        "mpd_reshapes": [2, 3, 5, 7, 11, 17, 23, 31],
        "resolutions": [[1024,120,600], [2048,240,1200], [512,60,300]],
        "use_spectral_norm": False, "discriminator_channel_mult": 1,
    }
}

# --- Setup ---
os.makedirs(config["out_dir"], exist_ok=True)
if config.get("speaker_embed_dir"):
    os.makedirs(config["speaker_embed_dir"], exist_ok=True)
torch.manual_seed(1234)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def remove_prefix_from_state_dict(state_dict, prefix):
    """
    去除 state_dict 中所有以指定 prefix 开头的 key 的前缀。
    
    Args:
        state_dict (dict): 原始的模型权重字典。
        prefix (str): 要去掉的前缀，例如 "bigvgan."
        
    Returns:
        dict: 新的 state_dict，key 已去除前缀。
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
def add_prefix_to_state_dict(state_dict, prefix):
    """
    给 state_dict 中的所有 key 添加指定的前缀。
    
    Args:
        state_dict (dict): 原始的模型权重字典。
        prefix (str): 要添加的前缀，例如 "bigvgan."
        
    Returns:
        dict: 新的 state_dict，每个 key 都加上了前缀。
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = prefix + key  # 添加前缀
        new_state_dict[new_key] = value
    return new_state_dict
# ==== 1. Models ====
# Teacher Model
teacher = None
if config["teacher_ckpt"] and os.path.exists(config["teacher_ckpt"]):
    try:
        # USER: Implement actual teacher model loading here
        tmodel = torch.load(config["teacher_ckpt"], map_location=device)["generator"]
        bigvgan_cfg = OmegaConf.load("checkpoints/config.yaml").bigvgan
        bigvgan = Generator(bigvgan_cfg, use_cuda_kernel=False)
        vocoder_dict = torch.load(config["teacher_ckpt"], map_location="cpu")
        for key in list(vocoder_dict["generator"])[:5]:
          print(key)
        bigvgan.load_state_dict(vocoder_dict["generator"])
        #下面这句来自qwen
        bigvgan.requires_grad_(False)
        bigvgan = bigvgan.to("cpu")
        
        teacher = bigvgan.eval()
        logger.info(f"Teacher model loaded successfully from {config['teacher_ckpt']}")
        pass
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}. Using dummy.")
        teacher = None
else:
    logger.warning(f"Teacher checkpoint {config['teacher_ckpt']} not found or placeholder. Using dummy teacher.")

if teacher is None:
    class DummyTeacher(nn.Module):
        def __init__(self, target_len): super().__init__(); self.target_len = target_len
        def forward(self, mel, d_vector): return torch.zeros(mel.shape[0], 1, self.target_len, device=mel.device)
    teacher_target_len = config["segment_size"] * config["sampling_rate_teacher"] // config["sampling_rate"]
    teacher = DummyTeacher(teacher_target_len).to(device)
    logger.info("Using dummy teacher model.")

# Student Generator
student = Generator44kHzWithSpeaker(
    mel_dim_input=config["student_input_mel_bands"],
    speaker_embed_dim=config["speaker_embed_dim"],
    bigvgan_config_dict=config["bigvgan_student_hparams"]
).to(device)

# Discriminator
discriminator_hparams_obj = AttrDict(config["discriminator_hparams"])
discriminator = BigVGANDiscriminatorWrapper(discriminator_hparams_obj).to(device)

# Optimizers
# optim_g = torch.optim.AdamW(student.parameters(), lr=config["lr_g"], betas=config.get("adam_betas_g", (0.8, 0.99)))
optim_d = torch.optim.AdamW(discriminator.parameters(), lr=config["lr_d"], betas=config.get("adam_betas_d", (0.8, 0.99)))

# Load Checkpoints
start_epoch = 0
global_step = 0
if os.path.exists(config["student_ckpt"]):
    try:
        ckpt_g = torch.load(config["student_ckpt"], map_location=device)
        
        # student.load_state_dict(ckpt_g['generator'], strict=False)
        student.load_state_dict(add_prefix_to_state_dict(ckpt_g['generator'],"bigvgan."), strict=False)
        # optim_g.load_state_dict(ckpt_g['optim_g'], strict=False)
        start_epoch = ckpt_g.get('epoch', 0)
        global_step = ckpt_g.get('step', 0)
        logger.info(f"Loaded student G checkpoint from epoch {start_epoch}, step {global_step}")
    except Exception as e:
        logger.warning(f"Could not load student G checkpoint: {e}. Starting from scratch.")
if os.path.exists(config["discriminator_ckpt"]):
    try:
        ckpt_d = torch.load(config["discriminator_ckpt"], map_location=device)
        for key in list(ckpt_d)[:50]:
          print(key)
        discriminator.load_state_dict(ckpt_d['mpd'])
        optim_d.load_state_dict(ckpt_d['optim_d'])
        logger.info(f"Loaded discriminator D checkpoint.")
    except Exception as e:
        logger.warning(f"Could not load discriminator D checkpoint: {e}. Starting from scratch.")

# ==== 2. Dataset & DataLoader ====
train_dataset = AudioMelDataset(
    wav_dir="data/wavs_44k",
    segment_size=config["segment_size"],
    student_sampling_rate=config["sampling_rate"],
    student_hop_length=config["student_hop_length"],
    student_n_fft=config["student_n_fft"],
    student_win_length=config["student_win_length"],
    student_n_mels=config["student_input_mel_bands"],
    speaker_embed_dim=config["speaker_embed_dim"],
    speaker_embed_dir=config.get("speaker_embed_dir"),
    ecapa_model_path=config.get("ecapa_model_path"),
    ecapa_input_mels=config["ecapa_input_mels"],
    ecapa_sampling_rate=config["ecapa_sampling_rate"],
    ecapa_n_fft=config["ecapa_n_fft"],
    ecapa_hop_length=config["ecapa_hop_length"],
    ecapa_win_length=config["ecapa_win_length"],
    ecapa_mel_fmin=config["ecapa_mel_fmin"],
    ecapa_mel_fmax=config.get("ecapa_mel_fmax"),
    device=device
)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

# ==== 3. Loss Functions ====
stft_window_fn_str = config.get("stft_window_fn", "torch.hann_window")
try: stft_window_constructor = getattr(torch, stft_window_fn_str.split('.')[-1])
except AttributeError: stft_window_constructor = torch.hann_window; logger.warning(f"STFT window fn not found. Using hann.")

stft_loss_fn = STFTLoss(config["student_n_fft"], config["student_hop_length"], config["student_win_length"], stft_window_constructor, config.get("stft_center", True), config.get("stft_pad_mode", "reflect")).to(device)
mel_loss_fn = MelSpectrogramLoss(config["sampling_rate"], config["student_n_fft"], config["student_hop_length"], config["student_win_length"], config["student_input_mel_bands"], config.get("student_mel_fmin",0.0), config.get("student_mel_fmax",None), config.get("student_mel_power",1.0), config.get("student_mel_normalized",False), config.get("student_mel_center",True), config.get("student_mel_pad_mode","reflect")).to(device)
l1_loss = nn.L1Loss().to(device)
mel_teacher_extractor = torchaudio.transforms.MelSpectrogram(config["sampling_rate_teacher"], config["teacher_n_fft"], config["teacher_hop_length"], n_mels=config["teacher_mel_bands"]).to(device)

# ==== 5. Training Loop ====
logger.info(f"Starting training from epoch {start_epoch}, global_step {global_step}")
for epoch in range(start_epoch, config["epochs"]):
    for batch_idx, batch in enumerate(train_loader):
        mel_for_student, audio_44k_gt, d_vector = batch["mel"].to(device), batch["audio"].to(device), batch["d_vector"].to(device)
        if audio_44k_gt.ndim == 2: audio_44k_gt = audio_44k_gt.unsqueeze(1) # Ensure (B, 1, T)

        # --- Generator Training ---
        student.train()
        # optim_g.zero_grad()
        y_hat_44k = student(mel_for_student, d_vector)

        y_hat_24k_from_student = torchaudio.functional.resample(y_hat_44k.squeeze(1), orig_freq=config["sampling_rate"], new_freq=config["sampling_rate_teacher"]).unsqueeze(1)
        with torch.no_grad():
            audio_24k_for_teacher_mel = torchaudio.functional.resample(audio_44k_gt.squeeze(1), config["sampling_rate"], config["sampling_rate_teacher"])
            mel_for_teacher = mel_teacher_extractor(audio_24k_for_teacher_mel)
            y_teacher_24k = teacher(mel_for_teacher, d_vector)
        loss_teacher = l1_loss(y_hat_24k_from_student, y_teacher_24k.detach())

        loss_mel_student = mel_loss_fn(y_hat_44k.squeeze(1), audio_44k_gt.squeeze(1))
        loss_stft_sc_student, loss_stft_mag_student = stft_loss_fn(y_hat_44k.squeeze(1), audio_44k_gt.squeeze(1))

        y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(audio_44k_gt, y_hat_44k)
        loss_g_adv, _ = bigvgan_generator_loss(y_d_gs)
        loss_fm = bigvgan_feature_loss(fmap_rs, fmap_gs)

        lw = config["loss_weights"]
        loss_g = (lw["teacher"]*loss_teacher + lw["mel"]*loss_mel_student +
                  lw["stft_sc"]*loss_stft_sc_student + lw["stft_mag"]*loss_stft_mag_student +
                  lw["adv_g"]*loss_g_adv + lw["fm"]*loss_fm)
        loss_g.backward()
        # optim_g.step()

        # --- Discriminator Training ---
        optim_d.zero_grad()
        y_d_rs_d, y_d_gs_d, _, _ = discriminator(audio_44k_gt, y_hat_44k.detach())
        loss_d, _, _ = bigvgan_discriminator_loss(y_d_rs_d, y_d_gs_d)
        loss_d.backward()
        optim_d.step()

        if global_step % config["log_interval"] == 0:
            log_msg = f"Epoch {epoch}, Step {global_step}, Batch {batch_idx}/{len(train_loader)}, G_Loss: {loss_g.item():.4f}, D_Loss: {loss_d.item():.4f}, Mel: {loss_mel_student.item():.4f}, SC: {loss_stft_sc_student.item():.4f}, Mag: {loss_stft_mag_student.item():.4f}, G_Adv: {loss_g_adv.item():.4f}, FM: {loss_fm.item():.4f}, Teacher_L1: {loss_teacher.item():.4f}"
            logger.info(log_msg)

        if global_step > 0 and global_step % config["save_interval"] == 0:
            student_ckpt_path = os.path.join(config["out_dir"], os.path.basename(f"student_step_{global_step}.pth"))
            discriminator_ckpt_path = os.path.join(config["out_dir"], os.path.basename(f"discriminator_step_{global_step}.pth"))
            # torch.save({'epoch': epoch, 'step': global_step, 'generator': student.state_dict(), 'optimizer_state_dict': optim_g.state_dict()}, student_ckpt_path)
            torch.save({'epoch': epoch, 'step': global_step, 'generator': student.state_dict()}, student_ckpt_path)
            torch.save({'epoch': epoch, 'step': global_step, 'discriminator': discriminator.state_dict(), 'optimizer_state_dict': optim_d.state_dict()}, discriminator_ckpt_path)
            logger.info(f"Saved step checkpoints at step {global_step}")

        global_step += 1
    logger.info(f"Epoch {epoch} completed.")

logger.info("Training finished.")
final_student_ckpt_path = os.path.join(config["out_dir"], os.path.basename(config["student_ckpt"]))
final_discriminator_ckpt_path = os.path.join(config["out_dir"], os.path.basename(config["discriminator_ckpt"]))
for key in list(student.state_dict())[:5]:
  print(key)
# torch.save({'epoch': config["epochs"]-1, 'step': global_step, 'generator': student.state_dict(), 'optimizer_state_dict': optim_g.state_dict()}, final_student_ckpt_path)
torch.save({'epoch': config["epochs"]-1, 'step': global_step, 'generator': student.state_dict()}, final_student_ckpt_path)
torch.save({'epoch': config["epochs"]-1, 'step': global_step, 'discriminator': discriminator.state_dict(), 'optimizer_state_dict': optim_d.state_dict()}, final_discriminator_ckpt_path)
logger.info(f"Saved final models: {final_student_ckpt_path}, {final_discriminator_ckpt_path}")

# Save final model weights only for inference (optional)
torch.save(student.state_dict(), final_student_ckpt_path.replace(".pth", "_weights.pth"))
torch.save(discriminator.state_dict(), final_discriminator_ckpt_path.replace(".pth", "_weights.pth"))
logger.info(f"Saved final model weights (state_dict only).")
