import sys
import os
import shutil
import argparse
import random
import uuid
import concurrent.futures

import torch
import torch.nn.functional as F  # 用于插值对齐
import torchaudio
import numpy as np
import librosa
from tqdm import tqdm

# ================= 导入 Whisper 相关依赖 =================
try:
    from whisper.model import Whisper, ModelDimensions
    from whisper.audio import pad_or_trim, log_mel_spectrogram
except ImportError:
    print("Warning: whisper not installed or not found in path.")
    Whisper = None

# ================= 导入 WeNet (原有) =================
try:
    import wenet
except ImportError:
    print("Warning: wenet not installed.")
    wenet = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speaker_embedding", "espnet"))

try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False

from logger import utils
from logger.utils import traverse_dir
from reflow.extractors import F0_Extractor, Volume_Extractor, Units_Encoder
from speaker_embedding.espnet.espnet2.bin.spk_inference import Speech2Embedding
from reflow.vocoder import Vocoder


# =========================================================
# 1. 新增：Whisper PPG 封装类
# =========================================================
class WhisperPPGWrapper:
    def __init__(self, model_path, device='cuda'):
        if Whisper is None:
            raise ImportError("Please install openai-whisper or ensure whisper code is in path.")
        
        print(f"Loading Whisper model from {model_path} ...")
        checkpoint = torch.load(model_path, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        print(f"Whisper Dims: {dims}")
        
        model = Whisper(dims)
        
        # === 核心逻辑：删除 Decoder 和 顶层 Encoder ===
        del model.decoder
        cut = len(model.encoder.blocks) // 4
        cut = -1 * cut
        del model.encoder.blocks[cut:]
        
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        self.model = model.to(device)
        self.device = device
        print("Whisper model loaded and pruned.")

    def extract(self, audio_t, sample_rate):
        """
        input: audio_t (1, T) tensor
        output: ppg (T_frames, 1280) numpy array
        """
        # 1. 确保音频是 16k
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(audio_t.device)
            audio_16k = resampler(audio_t)
        else:
            audio_16k = audio_t

        # 2. 准备数据
        # Whisper 接收 (T,) 的 tensor 或 numpy
        audio_16k = audio_16k.squeeze(0) 
        
        # 计算原始长度对应的特征长度 (original logic: audln // 320)
        audln = audio_16k.shape[0]
        ppgln = audln // 320
        
        # Pad 到 30s (Whisper 要求)
        audio_input = pad_or_trim(audio_16k)
        
        # Mel Spectrogram
        mel = log_mel_spectrogram(audio_input).to(self.device)
        
        # 3. 前向传播
        with torch.no_grad():
            # encoder out: (1, 1500, 1280) -> squeeze -> (1500, 1280)
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze(0)
            
            # 4. 截取有效部分
            ppg = ppg[:ppgln, :] # [length, dim=1280]
            
        return ppg # tensor on device


# =========================================================
# 2. WeNet Wrapper (保持不变)
# =========================================================
class WenetEncoderWrapper:
    def __init__(self, model_dir, device='cuda'):
        if wenet is None:
            raise ImportError("Please install wenet first.")
        print(f"Loading WeNet model from {model_dir} ...")
        self.model = wenet.load_model(model_dir)
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, audio_t, sample_rate):
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(audio_t.device)
            waveform = resampler(audio_t)
        else:
            waveform = audio_t

        temp_file = f"temp_16k_{uuid.uuid4()}.wav"
        try:
            torchaudio.save(temp_file, waveform.cpu(), 16000)
            feats = self.model.compute_feature(temp_file)
            if isinstance(feats, tuple): feats = feats[0]
            feats = feats.to(self.device)
            if feats.dim() == 2: feats = feats.unsqueeze(0)
            feats_lengths = torch.tensor([feats.shape[1]], dtype=torch.long, device=self.device)
            
            if hasattr(self.model, 'lfr'):
                feats, feats_lengths = self.model.lfr(feats, feats_lengths)
            
            with torch.no_grad():
                encoder_out, _ = self.model.encoder(feats, feats_lengths)
            return encoder_out.squeeze(0) 
        finally:
            if os.path.exists(temp_file): os.remove(temp_file)

# =========================================================
# 3. 主逻辑
# =========================================================

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default=None)
    return parser.parse_args(args=args, namespace=namespace)
    
# 修改 preprocess 签名，增加 whisper_encoder
def preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, wenet_encoder, whisper_encoder, sample_rate, hop_size, device='cuda', use_pitch_aug=False, extensions=['wav']):
    
    path_srcdir = os.path.join(path, 'audio')
    
    # === 路径定义 ===
    path_unitsdir = os.path.join(path, 'units')
    path_wenetdir = os.path.join(path, 'wenet') 
    path_whisperdir = os.path.join(path, 'whisper') # 【新增】Whisper 路径
    
    path_f0dir = os.path.join(path, 'f0')
    path_volumedir = os.path.join(path, 'volume')
    path_augvoldir = os.path.join(path, 'aug_vol')
    path_meldir = os.path.join(path, 'mel')
    path_augmeldir = os.path.join(path, 'aug_mel')
    path_skipdir = os.path.join(path, 'skip')
    path_timbredir = os.path.join(path, 'timbre')

    # [新增] S_rev 特征的保存目录
    path_timbre_revdir = os.path.join(path, 'timbre_rev')
    
    filelist = traverse_dir(path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True)
    pitch_aug_dict = {}
    
    def process(file):
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_wenetfile = os.path.join(path_wenetdir, binfile)
        path_whisperfile = os.path.join(path_whisperdir, binfile) # 【新增】
        
        path_f0file = os.path.join(path_f0dir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)
        path_augvolfile = os.path.join(path_augvoldir, binfile)
        path_melfile = os.path.join(path_meldir, binfile)
        path_augmelfile = os.path.join(path_augmeldir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)
        path_timbrefile = os.path.join(path_timbredir, file + '.timbre.npy')

        # [新增] S_rev Timbre 路径
        path_timbre_revfile = os.path.join(path_timbre_revdir, file + '.timbre_rev.npy')
        
        # Load Audio
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1: audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device).unsqueeze(0)
        
        # Volume
        volume = volume_extractor.extract(audio)
        
        # Mel & Aug
        if mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            keyshift = random.uniform(-5, 5) if use_pitch_aug else 0
            aug_mel_t = mel_extractor.extract(audio_t * (10 ** log10_vol_shift), sample_rate, keyshift=keyshift)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            aug_vol = volume_extractor.extract(audio * (10 ** log10_vol_shift))
            
        # 1. 提取 Units (基准长度)
        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to('cpu').numpy() # (T_units, 768)
        target_len = units.shape[0]

        # 2. 提取 WeNet 并对齐
        wenet_t = wenet_encoder.encode(audio_t, sample_rate)
        # 对齐 WeNet
        if wenet_t.shape[0] != target_len:
            wenet_t = wenet_t.unsqueeze(0).transpose(1, 2)
            wenet_t = F.interpolate(wenet_t, size=target_len, mode='linear', align_corners=False)
            wenet_t = wenet_t.transpose(1, 2).squeeze(0)
        wenet_feats = wenet_t.cpu().numpy()
        
        # 3. 提取 Whisper PPG 并对齐 【新增逻辑】
        whisper_t = whisper_encoder.extract(audio_t, sample_rate)
        # 对齐 Whisper (虽然 320 hop_size 近似等于 units，但为了拼接安全，强制对齐)
        if whisper_t.shape[0] != target_len:
            whisper_t = whisper_t.unsqueeze(0).transpose(1, 2)
            whisper_t = F.interpolate(whisper_t, size=target_len, mode='linear', align_corners=False)
            whisper_t = whisper_t.transpose(1, 2).squeeze(0)
        whisper_feats = whisper_t.cpu().numpy()

        # F0 & Timbre
        f0 = f0_extractor.extract(audio, uv_interp=False)
        timbre = timbre_extractor(audio).squeeze().to('cpu').numpy()

        # [新增] extract timbre_rev (时间反转)
        # 将音频沿时间轴反转
        audio_vocal_rev = np.flip(audio).copy()
        
        # 提取反转后的特征
        timbre_rev = timbre_extractor(audio_vocal_rev).squeeze().to('cpu').numpy()
        
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            # Save All
            os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
            np.save(path_unitsfile, units)
            
            os.makedirs(os.path.dirname(path_wenetfile), exist_ok=True)
            np.save(path_wenetfile, wenet_feats)
            
            os.makedirs(os.path.dirname(path_whisperfile), exist_ok=True) # 保存 Whisper
            np.save(path_whisperfile, whisper_feats)

            os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
            np.save(path_f0file, f0)
            os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
            np.save(path_volumefile, volume)
            os.makedirs(os.path.dirname(path_timbrefile), exist_ok=True)
            np.save(path_timbrefile, timbre) 
            # [新增] 保存 S_rev Timbre
            os.makedirs(os.path.dirname(path_timbre_revfile), exist_ok=True)
            np.save(path_timbre_revfile, timbre_rev)
            if mel_extractor is not None:
                pitch_aug_dict[file] = keyshift
                os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
                np.save(path_melfile, mel)
                os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
                np.save(path_augmelfile, aug_mel)
                os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
                np.save(path_augvolfile, aug_vol)
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            
    print('Preprocess audio in :', path_srcdir)
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    
    if mel_extractor is not None:
        np.save(os.path.join(path, 'pitch_aug_dict.npy'), pitch_aug_dict)

if __name__ == '__main__':
    cmd = parse_args()
    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    extensions = args.data.extensions
    
    f0_extractor = F0_Extractor(args.data.f0_extractor, args.data.sampling_rate, args.data.block_size, args.data.f0_min, args.data.f0_max)
    volume_extractor = Volume_Extractor(args.data.block_size)
    timbre_extractor = Speech2Embedding(model_file=args.data.timbre_model_path, train_config=args.data.timbre_model_config, device="cuda" if torch.cuda.is_available() else "cpu")

    mel_extractor = None
    use_pitch_aug = False
    if args.model.type in ['RectifiedFlow_VAE']:
        mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
        if mel_extractor.vocoder_sample_rate != sample_rate:
            mel_extractor = None
        elif args.model.use_pitch_aug:
            use_pitch_aug = True
    
    # 1. Init Units Encoder (Hubert/ContentVec)
    cnhubertsoft_gate = args.data.cnhubertsoft_gate if args.data.encoder == 'cnhubertsoftfish' else 10
    print(f"Initializing Units Encoder ({args.data.encoder})...")
    units_encoder = Units_Encoder(args.data.encoder, args.data.encoder_ckpt, args.data.encoder_sample_rate, args.data.encoder_hop_size, cnhubertsoft_gate=cnhubertsoft_gate, device=device)

    # 2. Init WeNet Encoder
    wenet_model_path = args.data.wenet_model_path
    print(f"Initializing WeNet Encoder ({wenet_model_path})...")
    wenet_encoder = WenetEncoderWrapper(model_dir=wenet_model_path, device=device)
    
    # 3. Init Whisper PPG Encoder 【新增初始化】
    # 请确保此路径指向正确的 Whisper large-v2.pt 文件
    whisper_model_path = args.data.whisper_model_path
    # 如果你需要绝对路径，请修改这里，例如：
    # whisper_model_path = r"D:\research\code\SVC\whisper_pretrain\large-v2.pt"
    
    print(f"Initializing Whisper PPG Encoder ({whisper_model_path})...")
    if not os.path.exists(whisper_model_path):
        # 简单的回退检查，如果当前目录找不到，尝试找一个假设的绝对路径，或者报错
        print(f"Error: Whisper checkpoint not found at {whisper_model_path}")
        # whisper_encoder = None # 或者 raise Error
        
    whisper_encoder = WhisperPPGWrapper(model_path=whisper_model_path, device=device)

    # 运行预处理
    preprocess(args.data.train_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, wenet_encoder, whisper_encoder, sample_rate, hop_size, device=device, use_pitch_aug=use_pitch_aug, extensions=extensions)
    preprocess(args.data.valid_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, wenet_encoder, whisper_encoder, sample_rate, hop_size, device=device, use_pitch_aug=False, extensions=extensions)