import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import miditoolkit

'''
./dataset_midi/1.mid, 1.mp3, 2.....
'''


def read_label(path, sr):
    midi = miditoolkit.MidiFile(path)
    bpm = midi.tempo_changes[0].tempo
    sec_per_beat = 60/bpm # 每拍的秒数
    ticks_per_beat = midi.ticks_per_beat # 每拍的tick数
    events = []
    for inst in midi.instruments:
        for note in inst.notes:
            events.append(note.start / ticks_per_beat * sec_per_beat)
            # events.append(("note_on", note.start, note.pitch, note.velocity))
            # events.append(("note_off", note.end, note.pitch))
    label_idx = np.array(events) * sr
    label_idx = np.round(label_idx).astype(int)
    return label_idx



class MidiDataset(Dataset):
    def __init__(self, data_dir="./dataset_midi", sr=2205, chunk_max_size=2205 * 10, device="cpu"):
        super().__init__()
        self.data_dir = data_dir
        self.sr = sr
        self.chunk_max_size = chunk_max_size
        self.device = device
    def __getitem__(self, idx:int):
        if idx < 1 or idx > 31:
            raise ValueError("idx out of range")
        name = str(idx)
        sr = self.sr
        chunk_max_size = self.chunk_max_size
        data_dir = self.data_dir
        audio_path = os.path.join(data_dir, f"{name}.mp3")
        label_path = os.path.join(data_dir, f"{name}.mid")
        audio = librosa.load(audio_path, sr=sr)[0]
        label = read_label(label_path, sr)
        if audio.shape[0] > chunk_max_size:
            if (label > chunk_max_size).any():
                pass
            else:
                audio = audio[:chunk_max_size]
                label = label[label <= chunk_max_size]
        audio = torch.tensor(audio, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)
        return audio, label
    def __len__(self):
        return 31


class AudioChunkDataset(Dataset):
    """
    用于加载单个长音频文件并按 chunk_size 切分。
    chunk_size 单位 = 采样点。
    """
    def __init__(self, audio_path, sr=44100, chunk_size=44100*5, mono=True, device="cpu"):
        super().__init__()
        self.audio_path = audio_path
        self.sr = sr
        self.chunk_size = chunk_size
        self.mono = mono
        self.device = device

        # 加载音频
        audio, _ = librosa.load(audio_path, sr=sr, mono=mono)

        # 转成 numpy → torch
        audio = torch.tensor(audio, dtype=torch.float32)

        # 保存完整音频
        self.audio = audio
        self.total_len = audio.shape[-1]

        # chunk 数
        self.num_chunks = self.total_len // chunk_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        """
        返回 shape: (1, chunk_size)
        """
        start = idx * self.chunk_size
        end = start + self.chunk_size

        chunk = self.audio[start:end]      # (chunk_size,)
        chunk = chunk.unsqueeze(0)         # (1, chunk_size)
        return chunk.to(self.device)
