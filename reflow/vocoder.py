import os
import yaml
import torch
try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nsf_hifigan.nvSTFT import STFT
from nsf_hifigan.models import load_model,load_config
from torchaudio.transforms import Resample
from .reflow import Bi_RectifiedFlow
from .naive_v2_diff import NaiveV2Diff
from .wavenet import WaveNet

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

        
def load_model_vocoder(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model    
    if args.model.type == 'RectifiedFlow_VAE':
        model = Unit2Wav_VAE(
                    args.data.sampling_rate,
                    args.data.block_size,
                    args.model.win_length,
                    args.data.encoder_out_channels, 
                    args.data.wenet_dim,
                    args.data.whisper_dim,
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans,
                    args.model.n_hidden,
                    args.model.back_bone,
                    args.model.use_attention)
                    
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
        
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args


class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
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
        self.device = device
        
        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device = device)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device = device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
            
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()
        
    def extract(self, audio, sample_rate=0, keyshift=0):
                
        # resample
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)    
        
        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel
   
    def infer(self, mel, f0):
        f0 = f0[:,:mel.size(1),0] # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio
        
        
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
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
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.num_mels
        
    def extract(self, audio, keyshift=0):       
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel
    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class NsfHifiGANLog10(NsfHifiGAN):    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class Unit2Wav_VAE(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            wenet_dim, 
            whisper_dim, 
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=6, 
            n_chans=512,
            n_hidden=256,
            back_bone='lynxnet',
            use_attention=False):
        super().__init__()
        self.f0_embed = nn.Linear(1, n_hidden)
        self.use_attention = use_attention
        if use_attention:
            self.unit_embed = nn.Linear(n_unit, n_hidden)
            self.wenet_embed = nn.Linear(wenet_dim, n_hidden)
            self.whisper_embed = nn.Linear(whisper_dim, n_hidden)
            self.volume_embed = nn.Linear(1, n_hidden)
            self.phase_embed = nn.Linear(1, n_hidden)
            self.spk_proj = nn.Linear(192, n_hidden)
            self.attention = nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=n_hidden,
                    nhead=8,
                    dim_feedforward=n_hidden * 4,
                    dropout=0.1,
                    activation='gelu',
                ),
                nn.Linear(n_hidden, out_dims),
            )
            self.attention_wenet = nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=n_hidden,
                    nhead=8,
                    dim_feedforward=n_hidden * 4,
                    dropout=0.1,
                    activation='gelu',
                ),
                nn.Linear(n_hidden, out_dims),
            )
            self.attention_whisper = nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=n_hidden,
                    nhead=8,
                    dim_feedforward=n_hidden * 4,
                    dropout=0.1,
                    activation='gelu',
                ),
                nn.Linear(n_hidden, out_dims),
            )
        else:
            self.unit_embed = nn.Linear(n_unit, out_dims)
            self.volume_embed = nn.Linear(1, out_dims)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        self.block_size = block_size
        self.sampling_rate = sampling_rate
        self.flow_unit = Bi_RectifiedFlow(WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans, n_hidden=n_hidden))
        self.flow_wenet = Bi_RectifiedFlow(WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans, n_hidden=n_hidden))
        self.flow_whisper = Bi_RectifiedFlow(WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans, n_hidden=n_hidden))

    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        phase_frames = 2 * np.pi * rad[:, :, :1]
        return combtooth, phase_frames

    def forward(self, units, wenet, whisper, f0, volume,  spk_embedding, spk_embedding_rev, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        # combtooth exciter signal 
        combtooth, phase_frames = self.fast_source_gen(f0)
        # condition
        cond = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase_frames / np.pi)
        if not infer:
            # 【训练阶段】：数据增强模式
            # 50% 概率完全使用原音色，50% 概率完全使用反转音色
            # 这样强迫模型学习：无论输入哪种 Embedding，都要还原出正确的目标音频
            prob = torch.rand(1).item()
            if prob < 0.5:
                spk_clean = spk_embedding
            else:
                spk_clean = spk_embedding_rev
        else:
            spk_clean = spk_embedding_rev



        # print(cond.shape)          # torch.Size([48, 172, 256])
        spk_expanded = spk_clean.unsqueeze(1).repeat(1, cond.shape[1], 1)
        # print(spk_embedding.shape) # torch.Size([48, 172, 192])
        cond = cond + self.spk_proj(spk_expanded)
        # print(cond.shape)          # torch.Size([48, 172, 256])
        if self.aug_shift_embed is not None and aug_shift is not None:
            cond = cond + self.aug_shift_embed(aug_shift / 5)
        # print(cond.shape)          # torch.Size([48, 172, 256])
        
        # vae mean
        x_unit = self.unit_embed(units) + self.volume_embed(volume)
        x_wenet = self.wenet_embed(wenet) + self.volume_embed(volume)
        x_whisper = self.whisper_embed(whisper) + self.volume_embed(volume)

        # print(x.shape)             # torch.Size([48, 172, 256])
        if self.use_attention:
            x = self.attention(x_unit)
            x_wenet = self.attention_wenet(x_wenet)
            x_whisper = self.attention_whisper(x_whisper)

            
        # print(x.shape)             # torch.Size([48, 172, 128])
        # exit()

        same_noise = torch.randn_like(x)

        # vae noise
        x += same_noise
        x_wenet += same_noise
        x_whisper += same_noise

        # cond_all = cond + x_unit
        # B, *rest, C = cond_all.shape
        # x_noise = torch.randn(B, *rest, 128, device=cond_all.device, dtype=cond_all.dtype)

        # x = self.reflow_model(infer=infer, x_start=x_noise, x_end=gt_spec, cond=cond_all, infer_step=infer_step, method='euler', use_tqdm=True)
        
        # x = self.flow_unit(infer=infer, x_start=x, x_end=gt_spec, cond=cond, infer_step=infer_step, method='euler', use_tqdm=True)
        # x_f0 = self.flow_f0(infer=infer, x_start=x_f0, x_end=gt_spec, cond=x_unit, infer_step=infer_step, method='euler', use_tqdm=True)

        if not infer:
            # 1. 【关键】生成共享的时间步 t
            # 这样两个分支看到的是“同一时刻”的状态，平均才有物理意义
            B = x.shape[0]
            t = torch.rand(B, device=x.device).clamp(1e-7, 1-1e-7)

            # 2. 调用两个 Flow，获取 Loss 和 预测的频谱 (return_pred=True)
            # 注意传入 t_input=t
            loss_unit = self.flow_unit(infer=False, x_start=x, x_end=gt_spec, cond=cond, t_input=t, return_pred=False)
            loss_wenet = self.flow_wenet(infer=False, x_start=x_wenet, x_end=gt_spec, cond=cond, t_input=t, return_pred=False)
            loss_whisper = self.flow_whisper(infer=False, x_start=x_whisper, x_end=gt_spec, cond=cond, t_input=t, return_pred=False)

            # # 3. 【核心需求】计算平均预测的 Mel Loss
            # avg_pred = (pred_unit + pred_f0) / 2
            
            # 注意：gt_spec 需要进行同样的归一化才能比较
            # 我们可以复用 flow_unit 里的 norm_spec 函数
            # target_norm = self.flow_unit.norm_spec(gt_spec).transpose(1, 2).unsqueeze(1)
            
            # 计算平均后的 L1 Loss (推荐用 L1 保持清晰度)
            # loss_avg_mel = (avg_pred - target_norm).abs().mean()

            # 4. 总 Loss
            # 建议：保留各自的 Flow Loss 以保证 ODE 轨迹的正确性，再加上平均 Mel Loss
            # 你可以给 loss_avg_mel 加一个权重，例如 1.0 或 0.5
            total_loss = loss_unit + loss_wenet + loss_whisper
            
            return total_loss


        else:
            # ==================== 推理逻辑修改 ====================
            
            # 1. 准备初始状态
            # 必须和训练时一样，使用共享的噪声 (same_noise)
            # x 此时是经过 Attention 后的 Unit Embedding
            # x_f0 此时是经过 Attention 后的 F0 Embedding
            # same_noise = torch.randn_like(x)
            
            # 构造起点 (Source): Embedding + Noise
            # 对应训练时的 x_start
            x_start_unit = x 
            x_start_wenet = x_wenet 
            x_start_whisper = x_whisper 
            
            # 2. 并行推理 (Parallel Inference)      
            # 分支 A: Unit Flow
            # 它的条件是 cond (包含 F0, Phase, Spk 等)
            # mel_unit = self.flow_unit(
            #     infer=infer, 
            #     x_start=x_start_unit, 
            #     x_end=None, # 推理时不知道终点
            #     cond=cond, 
            #     infer_step=infer_step, 
            #     method=method, 
            #     use_tqdm=use_tqdm
            # )

            # 分支 B: Wenet Flow
            # 它的条件是 cond (包含 F0, Phase, Spk 等)
            mel_wenet = self.flow_wenet(
                infer=infer, 
                x_start=x_start_wenet, 
                x_end=None, # 推理时不知道终点
                cond=cond, 
                infer_step=infer_step, 
                method=method, 
                use_tqdm=use_tqdm
            )
            # 分支 C: Whisper Flow
            # 它的条件是 cond (包含 F0, Phase, Spk 等)
            # mel_whisper = self.flow_whisper(
            #     infer=infer, 
            #     x_start=x_start_whisper, 
            #     x_end=None, # 推理时不知道终点
            #     cond=cond, 
            #     infer_step=infer_step, 
            #     method=method, 
            #     use_tqdm=use_tqdm
            # )
            
            
            # 4. 声码器生成波形
            if return_wav and infer:
                return vocoder.infer(mel_wenet, f0)
            else:
                return mel_wenet
            
