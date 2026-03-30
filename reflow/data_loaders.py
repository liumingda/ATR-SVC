import os
import random
import re
import numpy as np
import librosa
import torch
try:
    import torch_musa
except ImportError:
    pass
import random
from tqdm import tqdm
from torch.utils.data import Dataset

def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        # n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=True)
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        # n_spk=args.model.n_spk
        )
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        # n_spk=2,
        device='cpu',
        fp16=False,
        use_aug=False,
    ):
        super().__init__()
        
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.pitch_aug_dict = np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()

        self.timbre_data_buffer={}

        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)

        # speech2spk_embed = Speech2Embedding(model_file="/home/ma-user/modelarts/user-job-dir/ddsp-svc_0321/spk_embedding_model_data_preprocessing/40epoch.pth", train_config="/home/ma-user/modelarts/user-job-dir/ddsp-svc_0321/spk_embedding_model_data_preprocessing/config.yaml")

        for name_ext in tqdm(self.paths, total=len(self.paths)):
            # print(name_ext)
            # print(name_ext)


            name = os.path.splitext(name_ext)[0]
            # print(name)
            path_audio = os.path.join(self.path_root, 'audio', name_ext)
            # print(path_audio)
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)

            # wav, _ = librosa.load(path_audio, sr=sample_rate)
            # waveform_spk_embedding = torch.FloatTensor(wav)
            # spk_embedding = speech2spk_embed(waveform_spk_embedding).to("cpu")

            path_timbre = os.path.join(self.path_root, 'timbre', name_ext) + '.timbre.npy'
            spk_embedding = np.load(path_timbre)
            spk_embedding = torch.from_numpy(spk_embedding).float().to(device)

            path_timbre_rev = os.path.join(self.path_root, 'timbre_rev', name_ext) + '.timbre_rev.npy'
            spk_embedding_rev = np.load(path_timbre_rev)
            spk_embedding_rev = torch.from_numpy(spk_embedding_rev).float().to(device)

            # print(spk_embedding.shape)
            # tt
            
            path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
            # print(path_f0)
            # tt
            f0 = np.load(path_f0)
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)

                
            path_volume = os.path.join(self.path_root, 'volume', name_ext) + '.npy'
            volume = np.load(path_volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)
            
            path_augvol = os.path.join(self.path_root, 'aug_vol', name_ext) + '.npy'
            aug_vol = np.load(path_augvol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)
                        

            if load_all_data:
                '''
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).to(device)
                '''
                path_mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)
                
                path_augmel = os.path.join(self.path_root, 'aug_mel', name_ext) + '.npy'
                aug_mel = np.load(path_augmel)
                aug_mel = torch.from_numpy(aug_mel).to(device)
                
                path_units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
                units = np.load(path_units)
                units = torch.from_numpy(units).to(device)

                path_wenet = os.path.join(self.path_root, 'wenet', name_ext) + '.npy'
                wenet = np.load(path_wenet)
                wenet = torch.from_numpy(wenet).to(device)

                path_whisper = os.path.join(self.path_root, 'whisper', name_ext) + '.npy'
                whisper = np.load(path_whisper)
                whisper = torch.from_numpy(whisper).to(device)

                # audio_accompany_path_mel = os.path.join(self.path_root, 'audio_accompany_mel', name_ext) + '.npy'
                # accompany_mel = np.load(audio_accompany_path_mel)
                # accompany_mel = torch.from_numpy(accompany_mel).to(device)
                
                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()
                    wenet = wenet.half()
                    whisper = whisper.half()
                    # accompany_mel = accompany_mel.half()
                    
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'wenet': wenet,
                        'whisper': whisper,
                        # 'accompany_mel': accompany_mel,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_embedding': spk_embedding,
                        'spk_embedding_rev': spk_embedding_rev
                        # 'spk_id': spk_id
                        }
            else:
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_embedding': spk_embedding,
                        'spk_embedding_rev': spk_embedding_rev
                        # 'spk_id': spk_id
                        }
           
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name_ext)
            # print(path_audio)
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)

            path_timbre = os.path.join(self.path_root, 'timbre', name_ext) + '.timbre.npy'
            spk_embedding = np.load(path_timbre)
            spk_embedding = torch.from_numpy(spk_embedding).float().to(device)   

            path_timbre_rev = os.path.join(self.path_root, 'timbre_rev', name_ext) + '.timbre_rev.npy'
            spk_embedding_rev = np.load(path_timbre_rev)
            spk_embedding_rev = torch.from_numpy(spk_embedding_rev).float().to(device)                   
                    
            self.timbre_data_buffer[name_ext] = {
                    'duration': duration,
                    'spk_embedding': spk_embedding,
                    'spk_embedding_rev': spk_embedding_rev
                    }


    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]

        timbre_name_ext = random.choice([name for name in self.paths if name != self.paths[file_idx]])

        timbre_data_buffer = self.timbre_data_buffer[timbre_name_ext]
        # check duration. if too short, then skip
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))
        if timbre_data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))            
        # get item
        return self.get_data(name_ext, data_buffer, timbre_name_ext, timbre_data_buffer)

    def get_data(self, name_ext, data_buffer, timbre_name_ext, timbre_data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        
        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug

        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = os.path.join(self.path_root, mel_key, name_ext) + '.npy'
            mel = np.load(mel)
            mel = mel[start_frame : start_frame + units_frame_len]
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]

        # # load accompany mel
        # accompany_mel = data_buffer.get('accompany_mel')
        # if accompany_mel is None:
        #     accompany_mel = os.path.join(self.path_root, 'audio_accompany_mel', name_ext) + '.npy'
        #     accompany_mel = np.load(accompany_mel)
        #     accompany_mel = accompany_mel[start_frame : start_frame + units_frame_len]
        #     accompany_mel = torch.from_numpy(accompany_mel).float() 
        # else:
        #     accompany_mel = accompany_mel[start_frame : start_frame + units_frame_len]

        # load units
        units = data_buffer.get('units')
        if units is None:
            units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
            units = np.load(units)
            units = units[start_frame : start_frame + units_frame_len]
            units = torch.from_numpy(units).float() 
        else:
            units = units[start_frame : start_frame + units_frame_len]
        
        # laod wenet
        wenet = data_buffer.get('wenet')
        if wenet is None:
            wenet = os.path.join(self.path_root, 'wenet', name_ext) + '.npy'
            wenet = np.load(wenet)
            wenet = wenet[start_frame : start_frame + units_frame_len]
            wenet = torch.from_numpy(wenet).float() 
        else:
            wenet = wenet[start_frame : start_frame + units_frame_len]

        # load whisper
        whisper = data_buffer.get('whisper')
        if whisper is None:
            whisper = os.path.join(self.path_root, 'whisper', name_ext) + '.npy'
            whisper = np.load(whisper)
            whisper = whisper[start_frame : start_frame + units_frame_len]
            whisper = torch.from_numpy(whisper).float() 
        else:
            whisper = whisper[start_frame : start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[name_ext]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # # load spk_id
        # spk_id = data_buffer.get('spk_id')
        spk_embedding = data_buffer.get('spk_embedding')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()

        content_dict = dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, wenet=wenet, whisper=whisper, spk_embedding=spk_embedding, aug_shift=aug_shift, name=name, name_ext=name_ext)
        
        ############# timbre #############

        timbre_name = os.path.splitext(timbre_name_ext)[0]

        timbre_spk_embedding = timbre_data_buffer.get('spk_embedding')
        timbre_spk_embedding_rev = timbre_data_buffer.get('spk_embedding_rev')

        timbre_dict = dict(spk_embedding=timbre_spk_embedding, spk_embedding_rev=timbre_spk_embedding_rev, name=timbre_name, name_ext=timbre_name_ext)
        
        return content_dict, timbre_dict

    def __len__(self):
        return len(self.paths)