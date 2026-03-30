# ATR-SVC

codes and demo are coming! 还在整理中ing！

预处理：

```bash
python preprocess.py -c configs/reflow-vae-wavenet.yaml
```

训练：

```bash
python train.py -c configs/reflow-vae-wavenet.yaml
```

推理：

```bash
python main.py -i source.wav -m model_ckpt.pt -o ./out.wav  -tw target.wav -k <keychange (semitones)> -step <infer_step> -method <method>
```

## 🎵 Demos

| Description | Audio Demo1 |
| :--- | :--- |
| ATR-SVC | [If player not showing, click here](assets/audio1.wav) 
| Sovits-SVC | [If player not showing, click here](assets/audio2.wav) 
| Whisper-SVC | [If player not showing, click here](assets/audio3.wav) 
| DDSP-SVC | [If player not showing, click here](assets/audio4.wav) 
