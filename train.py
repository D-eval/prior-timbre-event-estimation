import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.ndimage import gaussian_filter1d  # 用于平滑，可选
import os
import miditoolkit

from model import Model
from dataset import MidiDataset
from evaluation import Evaluator

from timbre_lib import ConcatTimbre, KickInterpKernel

device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
sr = 44100 // 20
batch_size = 1
chunk_max_size = sr*5
kernel_len = sr//4
num_interp = 5
# 组装音色
timbres = [
    KickInterpKernel(kernel_len,sr,num_interp,{'T':[0,1,-3]}),
    KickInterpKernel(kernel_len,sr,num_interp,{'T':[0,1,0]})
]
_timbre = ConcatTimbre(timbres,kernel_len,sr)
dataset = MidiDataset(data_dir="./dataset_midi", sr=sr, chunk_max_size=chunk_max_size, device=device)
model = Model()
evaluator = Evaluator()

save_root = "./fig/save_dir"
result_name = "eval_results.json"
evaluator.load(os.path.join(save_root, result_name))

temp_key = 0
if len(evaluator.idx2eval) > 0:
    temp_key = max(
        evaluator.idx2eval,
        key=lambda k: int(k)
    )
    temp_key = int(temp_key)

for i in range(temp_key+1,32):
    print(i)
    save_dir = f"./fig/save_dir/{i}"
    x, label = dataset[i]
    x = x.to(device)
    print("x shape:", x.shape)
    timbre, pred_tau, x_minus = model(x_ref = x,
                x_tar = x,
                timbre = _timbre,
                fig_save_dir = save_dir,
                label = label)
    pred_tau = torch.tensor(pred_tau)
    # eval
    evaluator.eval(i, pred_tau.cpu(), label.cpu(),
                   x_minus.detach().cpu(), x.detach().cpu(),
                   timbre, save_dir)
    evaluator.save(os.path.join(save_root, result_name))
    print("save eval results")
