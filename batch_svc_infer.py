import sys
import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
import uuid
import hashlib
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from ast import literal_eval

# ==========================================
# 1. 定义项目绝对路径 (根据你的环境配置)
# ==========================================
MSS_ROOT = "/root/autodl-tmp/code/SVC/music_source_separation"
SVC_ROOT = "/root/autodl-tmp/code/interspeech/final_timbre_three_branch_0215/Flow-Matching-SVC"

# ==========================================
# 2. 动态挂载路径
# ==========================================
if MSS_ROOT not in sys.path:
    sys.path.insert(0, MSS_ROOT)

if SVC_ROOT not in sys.path:
    sys.path.insert(0, SVC_ROOT)

# 将 espnet 加入搜索路径
espnet_path = os.path.join(SVC_ROOT, "speaker_embedding", "espnet")
if espnet_path not in sys.path:
    sys.path.insert(0, espnet_path)

# ==========================================
# 3. 导入依赖
# ==========================================

# --- Whisper & WeNet 依赖检测 ---
try:
    from whisper.model import Whisper, ModelDimensions
    from whisper.audio import pad_or_trim, log_mel_spectrogram
except ImportError:
    print("Warning: whisper not installed or not found in path.")
    Whisper = None

try:
    import wenet
except ImportError:
    print("Warning: wenet not installed.")
    wenet = None

try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False

# --- 加载 MSS 和 SVC 本地模块 ---
try:
    # MSS
    from train import get_model as get_mss_model
    from mss.utils import parse_yaml, separate_overlap_add
    print(f"✅ 成功加载 MSS 模块: {MSS_ROOT}")

    # SVC
    from slicer import Slicer
    from reflow.extractors import F0_Extractor, Volume_Extractor, Units_Encoder
    from reflow.vocoder import load_model_vocoder
    from speaker_embedding.espnet.espnet2.bin.spk_inference import Speech2Embedding
    from logger import utils
    print(f"✅ 成功加载 SVC 模块: {SVC_ROOT}")

except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    sys.exit(1)


# =========================================================
# 4. 新增：Wrapper 类 (Whisper & WeNet)
# =========================================================

class WhisperPPGWrapper:
    def __init__(self, model_path, device='cuda'):
        if Whisper is None:
            raise ImportError("Please install openai-whisper or ensure whisper code is in path.")
        
        print(f"Loading Whisper model from {model_path} ...")
        checkpoint = torch.load(model_path, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        
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
        
        # 计算原始长度对应的特征长度
        audln = audio_16k.shape[0]
        ppgln = audln // 320
        
        # Pad 到 30s (Whisper 要求)
        audio_input = pad_or_trim(audio_16k)
        
        # Mel Spectrogram
        mel = log_mel_spectrogram(audio_input).to(self.device)
        
        # 3. 前向传播
        with torch.no_grad():
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze(0)
            # 4. 截取有效部分
            ppg = ppg[:ppgln, :] # [length, dim=1280]
            
        return ppg 


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


# ==========================================
# 5. 通用辅助函数
# ==========================================

def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = F.interpolate(torch.cat((signal, signal[:, :, -1:]), 2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)

def split(audio, sample_rate, hop_size, db_thresh=-40, min_len=5000):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
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


# ==========================================
# 6. 核心功能函数
# ==========================================

def run_mss(config_path, ckpt_path, audio_path, device="cuda"):
    """执行人声分离，返回 (人声, 伴奏, 采样率)"""
    print(f"\n[Stage 1] MSS 人声分离...")
    configs = parse_yaml(config_path)
    sr = configs["sample_rate"]
    segment_samples = round(configs["segment_duration"] * sr)
    
    # 加载模型
    model = get_mss_model(configs=configs, ckpt_path=ckpt_path).to(device)
    model.eval()

    # 加载音频
    audio, _ = librosa.load(audio_path, sr=sr, mono=False)
    if audio.ndim == 1: 
        audio = np.array([audio, audio])

    # 推理
    with torch.no_grad():
        vocals = separate_overlap_add(
            model=model, audio=audio, 
            segment_samples=segment_samples, 
            hop_length=segment_samples // 4, 
            batch_size=1
        )

    # 伴奏 = 原曲 - 人声 (对齐长度)
    min_len = min(audio.shape[1], vocals.shape[1])
    audio = audio[:, :min_len]
    vocals = vocals[:, :min_len]
    instrumental = audio - vocals

    return vocals.T, instrumental.T, sr


def run_svc(vocal_path, svc_model_path, target_wav, device="cuda", 
            key=0, formant_shift_key=0, 
            pitch_extractor_type='rmvpe', f0_min=50, f0_max=1100, 
            threshold=-60, infer_step='auto', method='auto'):
    """
    执行 SVC 歌声转换 (使用三分支 Reflow 模型)
    """
    print(f"\n[Stage 2] SVC 歌声转换 (Reflow Three-Branch)...")

    # 1. Load Model
    print(f" -> Loading model from: {svc_model_path}")
    model, vocoder, args = load_model_vocoder(svc_model_path, device=device)

    # 2. Load Input Audio
    audio, sample_rate = librosa.load(vocal_path, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    # Calculate hop_size
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate

    # 3. F0 Extraction
    print(f' -> Extracting Pitch ({pitch_extractor_type})...')
    pitch_extractor = F0_Extractor(
                        pitch_extractor_type, 
                        sample_rate, 
                        hop_size, 
                        float(f0_min), 
                        float(f0_max))
    f0 = pitch_extractor.extract(audio, uv_interp=True, device=device)

    # Key & Formant shift
    input_f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
    output_f0 = input_f0 * 2 ** (float(key) / 12)
    formant_shift_key_tensor = torch.from_numpy(np.array([[float(formant_shift_key)]])).float().to(device)

    # 4. Initialize Encoders
    print(' -> Initializing Encoders...')
    # Unit Encoder
    cnhubertsoft_gate = args.data.cnhubertsoft_gate if args.data.encoder == 'cnhubertsoftfish' else 10
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size,
                        cnhubertsoft_gate=cnhubertsoft_gate,
                        device=device)

    # WeNet Encoder (New)
    print(f" -> Initializing WeNet: {args.data.wenet_model_path}")
    wenet_encoder = WenetEncoderWrapper(model_dir=args.data.wenet_model_path, device=device)
    
    # Whisper Encoder (New)
    print(f" -> Initializing Whisper: {args.data.whisper_model_path}")
    whisper_encoder = WhisperPPGWrapper(model_path=args.data.whisper_model_path, device=device)

    # 5. Volume Extraction & Mask
    print(' -> Extracting Volume...')
    volume_extractor = Volume_Extractor(hop_size)
    volume = volume_extractor.extract(audio)
    
    mask = (volume > 10 ** (float(threshold) / 20)).astype('float')
    mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = upsample(mask, args.data.block_size).squeeze(-1)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)

    # 6. Speaker Embedding
    print(f' -> Extracting Speaker Embedding from: {target_wav}')
    target_singer, _ = torchaudio.load(target_wav)
    if target_singer.shape[0] > 1:
        target_singer = target_singer[0:1, :]

    speech2spk_embed = Speech2Embedding(
        model_file=args.data.timbre_model_path, 
        train_config=args.data.timbre_model_config, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    spk_embedding = speech2spk_embed(target_singer[0])
    spk_embedding_rev = speech2spk_embed(np.flip(target_singer[0].cpu().numpy()).copy())

    # 7. Sampling Method & Steps
    if method == 'auto':
        method = args.infer.method
    
    if infer_step == 'auto':
        real_infer_step = args.infer.infer_step
    else:
        real_infer_step = int(infer_step)

    if real_infer_step < 0:
        print('Infer step cannot be negative!')
        exit(0)

# 8. Inference Loop (融合 Padding 逻辑，防止 WeNet 报错)
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sample_rate, hop_size)
    print(f' -> Cut input audio into {len(segments)} slices')

    # 定义最小安全长度：WeNet 窗口 400，补到 1600 (16k 下的 0.1s) 绝对安全
    # 注意：如果 sample_rate 是 44100，对应的采样点数需换算
    MIN_SAFE_SAMPLES = int(0.1 * sample_rate) 
    PADDING_SAMPLES = int(0.2 * sample_rate) # 前后各补 0.2s 静音

    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            audio_data = segment[1]
            original_len = len(audio_data)
            
            # --- 关键修改：动态 Padding 逻辑 ---
            is_padded = False
            if original_len < MIN_SAFE_SAMPLES:
                # 左右各补 PADDING_SAMPLES 个采样点的静音
                audio_data = np.pad(audio_data, (PADDING_SAMPLES, PADDING_SAMPLES), mode='constant')
                is_padded = True
            
            seg_input = torch.from_numpy(audio_data).float().unsqueeze(0).to(device)
            
            # 1. Units 提取
            seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
            target_len = seg_units.size(1)

            # 2. WeNet (现在有了 Padding，绝对不会再报错 186 < 400)
            seg_wenet_raw = wenet_encoder.encode(seg_input, sample_rate)
            # 对齐 WeNet 特征到 Units 长度
            if seg_wenet_raw.shape[0] != target_len:
                seg_wenet_raw = seg_wenet_raw.unsqueeze(0).transpose(1, 2)
                seg_wenet_raw = F.interpolate(seg_wenet_raw, size=target_len, mode='linear', align_corners=False)
                seg_wenet_raw = seg_wenet_raw.transpose(1, 2).squeeze(0)
            seg_wenet = seg_wenet_raw.unsqueeze(0)

            # 3. Whisper 提取与对齐
            seg_whisper_raw = whisper_encoder.extract(seg_input, sample_rate)
            if seg_whisper_raw.shape[0] != target_len:
                seg_whisper_raw = seg_whisper_raw.unsqueeze(0).transpose(1, 2)
                seg_whisper_raw = F.interpolate(seg_whisper_raw, size=target_len, mode='linear', align_corners=False)
                seg_whisper_raw = seg_whisper_raw.transpose(1, 2).squeeze(0)
            seg_whisper = seg_whisper_raw.unsqueeze(0)

            # 4. 其他特征 (F0, Volume) 
            # 注意：如果是 Padded 过的片段，F0 和 Volume 需要重新提取或特殊处理
            # 简单做法：针对 Padded 段重新提取 F0/Volume 以保持长度一致
            if is_padded:
                seg_f0_np = pitch_extractor.extract(audio_data, uv_interp=True, device=device)
                seg_f0 = torch.from_numpy(seg_f0_np).float().to(device).unsqueeze(-1).unsqueeze(0)
                seg_f0 = seg_f0 * 2 ** (float(key) / 12)
                
                seg_volume_np = volume_extractor.extract(audio_data)
                seg_volume = torch.from_numpy(seg_volume_np).float().to(device).unsqueeze(-1).unsqueeze(0)
            else:
                seg_f0 = output_f0[:, start_frame : start_frame + target_len, :]
                seg_volume = volume[:, start_frame : start_frame + target_len, :]
            
            # --- 模型推理 ---
            seg_output = model(
                seg_units, seg_wenet, seg_whisper,
                seg_f0, seg_volume, spk_embedding,
                spk_embedding_rev,
                aug_shift=formant_shift_key_tensor,
                vocoder=vocoder, infer=True, return_wav=True,
                infer_step=real_infer_step, method=method)
            
            # --- 还原长度与裁剪 ---
            seg_output = seg_output.squeeze().cpu().numpy()
            
            if is_padded:
                # 换算出 Padding 对应的生成音频长度并裁掉
                pad_len_output = int(PADDING_SAMPLES * args.data.sampling_rate / sample_rate)
                # 裁掉前后补充的静音部分
                seg_output = seg_output[pad_len_output : -pad_len_output]
                # 修正 target_len 用于后续 mask 索引
                actual_target_len = round(original_len / hop_size)
            else:
                actual_target_len = target_len

            # 5. 应用 Volume Mask
            # 确保 mask 索引不越界并与裁剪后的音频长度匹配
            seg_mask = mask[:, start_frame * args.data.block_size : (start_frame * args.data.block_size) + len(seg_output)]
            if seg_mask.shape[-1] < len(seg_output):
                # 填充 mask 或裁剪音频
                seg_output = seg_output[:seg_mask.shape[-1]]
            else:
                seg_mask = seg_mask[:, :len(seg_output)]
                
            seg_output *= seg_mask.squeeze().cpu().numpy()
            
            # 6. 拼接逻辑
            silent_length = round(start_frame * args.data.block_size) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)

    return result, args.data.sampling_rate


# ==========================================
# 7. 主程序 (批量处理逻辑)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MSS 参数
    parser.add_argument('--mss_config', type=str, required=True)
    parser.add_argument('--mss_ckpt', type=str, required=True)
    parser.add_argument('--source_list', type=str, required=True)
    # SVC 参数
    parser.add_argument('--svc_model', type=str, required=True)
    parser.add_argument('--target_list', type=str, required=True)
    parser.add_argument('--svc_key', type=int, default=0)
    # 输出
    parser.add_argument('--output_dir', type=str, default="results")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 读 txt
    with open(args.source_list, "r") as f:
        sources = [l.strip() for l in f if l.strip()]

    with open(args.target_list, "r") as f:
        targets = [l.strip() for l in f if l.strip()]

    print(f"共找到 {len(sources)} 对音频，开始处理🔥")

    from tqdm import tqdm

    total = len(sources) * len(targets)
    pbar = tqdm(total=total, desc="🎶 Processing")

    for i, src in enumerate(sources):
        for j, tar in enumerate(targets):

            src_name = os.path.splitext(os.path.basename(src))[0]
            tar_name = os.path.splitext(os.path.basename(tar))[0]
            folder_name = f"{tar_name}_svc_{src_name}"
            mms_folder_name = f"mms_{folder_name}.wav"
            pair_out = os.path.join(args.output_dir, folder_name)
            os.makedirs(pair_out, exist_ok=True)

            try:
                # 1. MSS
                vocals, inst, sr_mss = run_mss(args.mss_config, args.mss_ckpt, src, device)
                temp_vocal = os.path.join(pair_out, "temp_vocals.wav")
                temp_inst  = os.path.join(pair_out, "temp_inst.wav")
                sf.write(temp_vocal, vocals, sr_mss)
                sf.write(temp_inst, inst, sr_mss)

                # 2. SVC (使用新的三分支逻辑)
                svc_out_wav, sr_svc = run_svc(temp_vocal, args.svc_model, tar, device, args.svc_key)
                svc_path = os.path.join(pair_out, "svc_output.wav")
                sf.write(svc_path, svc_out_wav, sr_svc)

                # 3. 混音
                inst_audio, _ = librosa.load(temp_inst, sr=sr_svc, mono=False)
                min_len = min(inst_audio.shape[1], len(svc_out_wav))
                inst_audio = inst_audio[:, :min_len]
                svc_out_wav = svc_out_wav[:min_len]
                mixed = inst_audio + svc_out_wav

                mix_filename = f"{folder_name}.wav"
                mix_path = os.path.join(pair_out, mix_filename)
                sf.write(mix_path, mixed.T, sr_svc)

                # 4. MSS Cleanup (可选)
                # 这一步根据你的原始逻辑，似乎是将混音后的结果再次分离？
                # 如果不需要可以注释掉，或者保留以验证混音质量
                vocals, inst, sr_mss = run_mss(args.mss_config, args.mss_ckpt, mix_path, device)
                final_vocal_path = os.path.join(pair_out, mms_folder_name)
                sf.write(final_vocal_path, vocals, sr_mss)
            
            except Exception as e:
                print(f"Error processing {src_name} -> {tar_name}: {e}")
                import traceback
                traceback.print_exc()

            pbar.update(1)

    pbar.close()
    print("\n🎉 全部 Source × Target 混音完毕！")


"""
python batch_svc_infer.py \
    --mss_config /root/autodl-tmp/code/SVC/music_source_separation/configs/small.yaml \
    --mss_ckpt /root/autodl-tmp/code/SVC/music_source_separation/checkpoints/train/small/step=200000_ema.pth \
    --source_list /root/autodl-tmp/code/interspeech/baseline/Flow-Matching-SVC/data/test/source_list.txt \
    --target_list /root/autodl-tmp/code/interspeech/baseline/Flow-Matching-SVC/data/test/target_list.txt \
    --svc_model /root/autodl-tmp/code/interspeech/final_timbre_three_branch_0215/Flow-Matching-SVC/exp/reflowvae-test/model_200000.pt \
    --output_dir /root/autodl-tmp/code/interspeech/final_timbre_three_branch_0215/Flow-Matching-SVC/results/final_reverse_three_wenet_200k_0216

python batch_svc_infer.py \
    --mss_config /root/autodl-tmp/code/SVC/music_source_separation/configs/small.yaml \
    --mss_ckpt /root/autodl-tmp/code/SVC/music_source_separation/checkpoints/train/small/step=200000_ema.pth \
    --source_list /root/autodl-tmp/code/interspeech/baseline/Flow-Matching-SVC/data/test/source_list_2.txt \
    --target_list /root/autodl-tmp/code/interspeech/baseline/Flow-Matching-SVC/data/test/target_list.txt \
    --svc_model /root/autodl-tmp/code/interspeech/svc_three_branch_0123/Flow-Matching-SVC/exp/reflowvae-test/model_200000.pt \
    --output_dir /root/autodl-tmp/code/interspeech/svc_three_branch_0123/Flow-Matching-SVC/result_three_units_200k_0214
"""