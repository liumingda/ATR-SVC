import sys
import os
import shutil
import argparse
import random
import uuid
import hashlib
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from ast import literal_eval

import torch
import torch.nn.functional as F
import torchaudio

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

# ================= 导入项目依赖 =================
# 假设目录结构保持不变，添加 espnet 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speaker_embedding", "espnet"))

try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False

# 导入本地模块 (确保 reflow, slicer, speaker_embedding 等文件夹在同级或 pythonpath 下)
from slicer import Slicer
from reflow.extractors import F0_Extractor, Volume_Extractor, Units_Encoder
from reflow.vocoder import load_model_vocoder
from speaker_embedding.espnet.espnet2.bin.spk_inference import Speech2Embedding
from logger import utils

# =========================================================
# 1. Whisper PPG 封装类 (从 preprocess 移植)
# =========================================================
class WhisperPPGWrapper:
    def __init__(self, model_path, device='cuda'):
        if Whisper is None:
            raise ImportError("Please install openai-whisper or ensure whisper code is in path.")
        
        print(f"Loading Whisper model from {model_path} ...")
        checkpoint = torch.load(model_path, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        # print(f"Whisper Dims: {dims}")
        
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
        output: ppg (T_frames, 1280) tensor
        """
        # 1. 确保音频是 16k
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(audio_t.device)
            audio_16k = resampler(audio_t)
        else:
            audio_16k = audio_t

        # 2. 准备数据
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
            
        return ppg 


# =========================================================
# 2. WeNet Wrapper (从 preprocess 移植)
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

        # WeNet compute_feature 依赖文件路径，这里使用临时文件
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
# 3. 辅助函数
# =========================================================

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument("-d", "--device", type=str, default=None, required=False, help="cpu/cuda/musa, auto if not set")
    parser.add_argument("-i", "--input", type=str, required=True, help="path to the input audio file")
    parser.add_argument("-o", "--output", type=str, required=True, help="path to the output audio file")
    parser.add_argument("-tw", "--target_wav_path", type=str, required=True, help="reference audio for timbre")
    parser.add_argument("-k", "--key", type=str, required=False, default=0, help="key changed (semitones)")
    parser.add_argument("-f", "--formant_shift_key", type=str, required=False, default=0, help="formant changed (semitones)")
    parser.add_argument("-pe", "--pitch_extractor", type=str, required=False, default='rmvpe', help="pitch extrator type")
    parser.add_argument("-fmin", "--f0_min", type=str, required=False, default=50, help="min f0 (Hz)")
    parser.add_argument("-fmax", "--f0_max", type=str, required=False, default=1100, help="max f0 (Hz)")
    parser.add_argument("-th", "--threhold", type=str, required=False, default=-60, help="response threhold (dB)")
    parser.add_argument("-step", "--infer_step", type=str, required=False, default='auto', help="sample steps")
    parser.add_argument("-method", "--method", type=str, required=False, default='auto', help="euler or rk4")
    return parser.parse_args(args=args, namespace=namespace)

def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = F.interpolate(torch.cat((signal,signal[:,:,-1:]),2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:,:,:-1]
    return signal.permute(0, 2, 1)

def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 5000):
    slicer = Slicer(sr=sample_rate, threshold=db_thresh, min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result

def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


# =========================================================
# 4. 主程序
# =========================================================

if __name__ == '__main__':
    cmd = parse_args()
    
    # Device Config
    device = cmd.device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif use_torch_musa:
            if torch.musa.is_available():
                device = 'musa'
            else:
                device = 'cpu'
        else:
            device = 'cpu'
    
    # Load Model & Vocoder
    print(f"Loading model from {cmd.model_ckpt}...")
    model, vocoder, args = load_model_vocoder(cmd.model_ckpt, device=device)
    
    # Load Input Audio
    audio, sample_rate = librosa.load(cmd.input, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
    
    # MD5 & Cache
    md5_hash = ""
    with open(cmd.input, 'rb') as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
    
    cache_dir_path = os.path.join(os.path.dirname(__file__), "cache")
    cache_file_path = os.path.join(cache_dir_path, f"{cmd.pitch_extractor}_{hop_size}_{cmd.f0_min}_{cmd.f0_max}_{md5_hash}.npy")
    
    # F0 Extraction
    if os.path.exists(cache_file_path):
        print('Loading pitch curves from cache...')
        f0 = np.load(cache_file_path, allow_pickle=False)
    else:
        print(f'Extracting pitch using {cmd.pitch_extractor}...')
        pitch_extractor = F0_Extractor(
                            cmd.pitch_extractor, 
                            sample_rate, 
                            hop_size, 
                            float(cmd.f0_min), 
                            float(cmd.f0_max))
        f0 = pitch_extractor.extract(audio, uv_interp = True, device = device)
        os.makedirs(cache_dir_path, exist_ok=True)
        np.save(cache_file_path, f0, allow_pickle=False)
    
    # F0 & Formant Shift
    input_f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
    output_f0 = input_f0 * 2 ** (float(cmd.key) / 12)
    formant_shift_key = torch.from_numpy(np.array([[float(cmd.formant_shift_key)]])).float().to(device)
    
    # Init Units Encoder
    cnhubertsoft_gate = args.data.cnhubertsoft_gate if args.data.encoder == 'cnhubertsoftfish' else 10
    print(f"Initializing Units Encoder ({args.data.encoder})...")
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size,
                        cnhubertsoft_gate=cnhubertsoft_gate,
                        device = device)
    
    # ================= 补全: 初始化新增的 Encoder =================
    # Init WeNet Encoder
    print(f"Initializing WeNet Encoder ({args.data.wenet_model_path})...")
    wenet_encoder = WenetEncoderWrapper(model_dir=args.data.wenet_model_path, device=device)
    
    # Init Whisper Encoder
    print(f"Initializing Whisper Encoder ({args.data.whisper_model_path})...")
    whisper_encoder = WhisperPPGWrapper(model_path=args.data.whisper_model_path, device=device)
    # ==============================================================

    # Extract Volume
    print('Extracting volume...')
    volume_extractor = Volume_Extractor(hop_size)
    volume = volume_extractor.extract(audio)
    mask = (volume > 10 ** (float(cmd.threhold) / 20)).astype('float')
    mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = upsample(mask, args.data.block_size).squeeze(-1)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)
        
    # Spk Embedding
    print(f"Loading target speaker from {cmd.target_wav_path}...")
    target_singer, _ = torchaudio.load(cmd.target_wav_path)
    speech2spk_embed = Speech2Embedding(model_file=args.data.timbre_model_path, train_config=args.data.timbre_model_config, device= "cuda" if torch.cuda.is_available() else "cpu")
    spk_embedding = speech2spk_embed(target_singer[0])
    
    # Steps & Method
    method = args.infer.method if cmd.method == 'auto' else cmd.method
    infer_step = args.infer.infer_step if cmd.infer_step == 'auto' else int(cmd.infer_step)
    if infer_step < 0:
        print('infer step cannot be negative!')
        exit(0)
    
    # Inference Loop
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sample_rate, hop_size)
    print(f'Cut audio into {len(segments)} slices, starting inference...')
    
    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            # (1, T_sample)
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            
            # 1. Extract Units (Benchmark length)
            # (1, T_units, 768)
            seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
            target_len = seg_units.size(1)
            
            # 2. Extract & Align WeNet
            # (T_wenet, 512)
            seg_wenet_raw = wenet_encoder.encode(seg_input, sample_rate) 
            if seg_wenet_raw.shape[0] != target_len:
                # Interpolate: (1, dim, time)
                seg_wenet_raw = seg_wenet_raw.unsqueeze(0).transpose(1, 2)
                seg_wenet_raw = F.interpolate(seg_wenet_raw, size=target_len, mode='linear', align_corners=False)
                seg_wenet_raw = seg_wenet_raw.transpose(1, 2).squeeze(0)
            # Add batch dim -> (1, T, 512)
            seg_wenet = seg_wenet_raw.unsqueeze(0)

            # 3. Extract & Align Whisper
            # (T_whisper, 1280)
            seg_whisper_raw = whisper_encoder.extract(seg_input, sample_rate)
            if seg_whisper_raw.shape[0] != target_len:
                seg_whisper_raw = seg_whisper_raw.unsqueeze(0).transpose(1, 2)
                seg_whisper_raw = F.interpolate(seg_whisper_raw, size=target_len, mode='linear', align_corners=False)
                seg_whisper_raw = seg_whisper_raw.transpose(1, 2).squeeze(0)
            # Add batch dim -> (1, T, 1280)
            seg_whisper = seg_whisper_raw.unsqueeze(0)

            # 4. Prepare other features
            seg_f0 = output_f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
            
            # 5. Model Forward
            seg_output = model(
                seg_units, 
                seg_wenet,
                seg_whisper,
                seg_f0, 
                seg_volume, 
                spk_embedding,
                aug_shift = formant_shift_key,
                vocoder=vocoder,
                infer=True,
                return_wav=True,
                infer_step=infer_step, 
                method=method)
            
            # 6. Apply Volume Mask & Save
            seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]           
            seg_output = seg_output.squeeze().cpu().numpy()
            
            silent_length = round(start_frame * args.data.block_size) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)
            
        sf.write(cmd.output, result, args.data.sampling_rate)
        print(f"Done! Output saved to {cmd.output}")

"""
python main.py \
    -i /root/autodl-tmp/code/interspeech/baseline/Flow-Matching-SVC/result_mms_0207/001_爵士_隐匿的城市角落_003_svc_唐宋元明清_1/temp_vocals.wav\
    -m /root/autodl-tmp/code/interspeech/svc_three_branch_0123/Flow-Matching-SVC/exp/reflowvae-test/model_400000.pt\
    -o ./out.wav \
    -tw /root/autodl-tmp/code/interspeech/baseline_add_time_rev/Flow-Matching-SVC/data/train/audio/001_爵士_隐匿的城市角落_003.wav
"""