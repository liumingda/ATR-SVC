import sys
import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
import hashlib
from tqdm import tqdm
from ast import literal_eval
import torch.nn.functional as F

# ==========================================
# 1. 定义项目绝对路径 (请确保这些路径正确)
# ==========================================
MSS_ROOT = "/root/autodl-tmp/code/SVC/music_source_separation"
SVC_ROOT = "/root/autodl-tmp/code/interspeech/baseline/Flow-Matching-SVC"

# ==========================================
# 2. 动态挂载路径
# ==========================================
# 挂载 MSS
if MSS_ROOT not in sys.path:
    sys.path.insert(0, MSS_ROOT)

# 挂载 SVC 及其子模块 espnet
if SVC_ROOT not in sys.path:
    sys.path.insert(0, SVC_ROOT)

# [新增] 按照第一段代码的要求，将 speaker_embedding/espnet 加入搜索路径
espnet_path = os.path.join(SVC_ROOT, "speaker_embedding", "espnet")
if espnet_path not in sys.path:
    sys.path.insert(0, espnet_path)

# ==========================================
# 3. 导入模块
# ==========================================
# 3.1 尝试导入 Torch Musa (摩尔线程显卡支持，保留源码逻辑)
try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False

try:
    # --- MSS 模块 ---
    from train import get_model as get_mss_model
    from mss.utils import parse_yaml, separate_overlap_add
    print(f"✅ 成功加载 MSS 模块: {MSS_ROOT}")

    # --- SVC 模块 (使用新代码的引用方式) ---
    from slicer import Slicer
    from reflow.extractors import F0_Extractor, Volume_Extractor, Units_Encoder
    from reflow.vocoder import load_model_vocoder
    from speaker_embedding.espnet.espnet2.bin.spk_inference import Speech2Embedding
    print(f"✅ 成功加载 SVC 模块: {SVC_ROOT}")

except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print("请检查路径是否完全正确，且在该路径下确实存在对应的 python 文件。")
    sys.exit(1)


# ==========================================
# 4. 辅助函数 (来自新代码)
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
# 5. 主要逻辑函数
# ==========================================

def run_mss(config_path, ckpt_path, audio_path, device="cuda"):
    """执行人声分离，返回 (人声, 伴奏, 采样率) - 保持不变"""
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
    执行 SVC 歌声转换 (替换为新代码逻辑)
    """
    print(f"\n[Stage 2] SVC 歌声转换 (Reflow)...")

    # 1. Load Model
    print(f" -> Loading model from: {svc_model_path}")
    model, vocoder, args = load_model_vocoder(svc_model_path, device=device)

    # 2. Load Input Audio
    audio, sample_rate = librosa.load(vocal_path, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    # Calculate hop_size
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate

    # 3. F0 Extraction (直接提取，为了流程简洁去掉了本地文件缓存)
    print(f' -> Extracting Pitch ({pitch_extractor_type})...')
    pitch_extractor = F0_Extractor(
                        pitch_extractor_type, 
                        sample_rate, 
                        hop_size, 
                        float(f0_min), 
                        float(f0_max))
    f0 = pitch_extractor.extract(audio, uv_interp=True, device=device)

    # Key change processing
    input_f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
    output_f0 = input_f0 * 2 ** (float(key) / 12)

    # Formant change processing
    formant_shift_key_tensor = torch.from_numpy(np.array([[float(formant_shift_key)]])).float().to(device)

    # 4. Units Encoder
    print(' -> Encoding Units...')
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size,
                        cnhubertsoft_gate=cnhubertsoft_gate,
                        device=device)

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
    # 确保 target_singer 是单声道或取第一个通道
    if target_singer.shape[0] > 1:
        target_singer = target_singer[0:1, :]

    speech2spk_embed = Speech2Embedding(
        model_file=args.data.timbre_model_path, 
        train_config=args.data.timbre_model_config, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    spk_embedding = speech2spk_embed(target_singer[0])

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

    # 8. Inference Loop
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sample_rate, hop_size)
    print(f' -> Cut input audio into {len(segments)} slices')

    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
            
            # 这里的切片逻辑非常重要，确保维度对齐
            seg_f0 = output_f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
            
            seg_output = model(
                seg_units, 
                seg_f0, 
                seg_volume, 
                spk_embedding,
                aug_shift=formant_shift_key_tensor,
                vocoder=vocoder,
                infer=True,
                return_wav=True,
                infer_step=real_infer_step, 
                method=method)
            
            # Apply Volume Mask
            seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]          
            seg_output = seg_output.squeeze().cpu().numpy()
            
            # Cross-fade Logic
            silent_length = round(start_frame * args.data.block_size) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)

    return result, args.data.sampling_rate


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- MSS 参数 ---
    parser.add_argument('--mss_config', type=str, required=True, help="MSS模型配置文件路径")
    parser.add_argument('--mss_ckpt', type=str, required=True, help="MSS模型权重路径")
    parser.add_argument('--input_audio', type=str, required=True, help="输入音频路径")
    
    # --- SVC 参数 ---
    parser.add_argument('--svc_model', type=str, required=True, help="SVC模型Checkpoint路径")
    parser.add_argument('--target_voice', type=str, required=True, help="参考音色音频路径")
    parser.add_argument('--svc_key', type=float, default=0, help="变调 (半音数量)")
    parser.add_argument('--svc_formant_shift', type=float, default=0, help="共振峰偏移 (半音)")
    parser.add_argument('--svc_infer_step', type=str, default='auto', help="推理步数 (默认 auto)")
    parser.add_argument('--svc_method', type=str, default='auto', help="推理方法 (euler/rk4/auto)")
    parser.add_argument('--svc_f0_method', type=str, default='rmvpe', help="F0提取算法 (rmvpe/fcpe/...)")
    parser.add_argument('--svc_threshold', type=float, default=-60, help="静音阈值 (dB)")

    # --- 输出 ---
    parser.add_argument('--output_dir', type=str, default="results")
    
    args = parser.parse_args()
    
    # 设备选择
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif use_torch_musa and torch.musa.is_available():
        device = "musa"
    
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------
    # 1. 运行 MSS (人声分离)
    # ------------------------------------------------
    vocals, inst, sr_mss = run_mss(args.mss_config, args.mss_ckpt, args.input_audio, device)
    
    # 临时保存 MSS 结果
    temp_vocal_path = os.path.join(args.output_dir, "temp_vocals.wav")
    temp_inst_path = os.path.join(args.output_dir, "temp_inst.wav")
    sf.write(temp_vocal_path, vocals, sr_mss)
    sf.write(temp_inst_path, inst, sr_mss)
    print(f"MSS 中间文件已保存至: {args.output_dir}")

    # ------------------------------------------------
    # 2. 运行 SVC (歌声转换 - 使用新逻辑)
    # ------------------------------------------------
    svc_out_wav, sr_svc = run_svc(
        vocal_path=temp_vocal_path,
        svc_model_path=args.svc_model,
        target_wav=args.target_voice,
        device=device,
        key=args.svc_key,
        formant_shift_key=args.svc_formant_shift,
        pitch_extractor_type=args.svc_f0_method,
        infer_step=args.svc_infer_step,
        method=args.svc_method,
        threshold=args.svc_threshold
    )
    
    svc_path = os.path.join(args.output_dir, "svc_output.wav")
    sf.write(svc_path, svc_out_wav, sr_svc)
    print(f"SVC 结果已保存: {svc_path}")

    # ------------------------------------------------
    # 3. 最终混合 (伴奏 + SVC干声)
    # ------------------------------------------------
    print("\n[Stage 3] 最终合成...")
    # 重新加载伴奏，确保采样率与 SVC 输出一致
    inst_audio, _ = librosa.load(temp_inst_path, sr=sr_svc, mono=False) 
    
    # 转换 SVC 输出为 tensor 或 numpy 进行对齐
    if inst_audio.ndim == 1:
        # 如果伴奏变成单声道了，转成立体声方便混合
        inst_audio = np.array([inst_audio, inst_audio])
    
    # 确保 svc_out_wav 也是双声道 (如果需要) 或者单声道广播
    # 通常 svc_out 是单声道，我们需要广播到双声道
    svc_final = np.array([svc_out_wav, svc_out_wav])

    # 对齐长度
    min_len = min(inst_audio.shape[1], svc_final.shape[1])
    inst_audio = inst_audio[:, :min_len]
    svc_final = svc_final[:, :min_len]

    # 混合 (简单相加，可按需添加音量参数)
    mixed = inst_audio + svc_final
    
    final_path = os.path.join(args.output_dir, "final_complete.wav")
    sf.write(final_path, mixed.T, sr_svc)
    print(f"✅ 全流程结束！最终文件: {final_path}")

"""
python run_pipeline.py --mss_config "/root/autodl-tmp/code/SVC/music_source_separation/configs/small.yaml" --mss_ckpt "/root/autodl-tmp/code/SVC/music_source_separation/checkpoints/train/small/step=200000_ema.pth" --input_audio "/root/autodl-tmp/code/SVC/music_source_separation/assets/music_10s.wav" --svc_model "/root/autodl-tmp/code/interspeech/baseline/Flow-Matching-SVC/exp/reflowvae-test/model_500000.pt" --target_voice "/root/autodl-tmp/code/SVC/DDSP-SVC_instru_copy_test/svc_result/test.wav" --output_dir "final_output"
"""