import torch
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
import json

def plot_wave_only(wave, save_path, color="black", linewidth=1.0):
    """
    只绘制波形，没有坐标轴、没有边框、没有文字。
    非常适合 PPT 示意图。
    """
    wave = wave.detach().cpu().numpy()

    plt.figure(figsize=(8, 2))
    plt.plot(wave, linewidth=linewidth, color=color)

    # 去掉所有坐标轴与边框
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved waveform to: {save_path}")


def plot_all_three(x, x_minus, kernel, pred_tau, save_dir="./fig/ppt"):
    os.makedirs(save_dir, exist_ok=True)

    # ------ 图 1：原始音频 ------
    plot_wave_only(
        wave=x,
        save_path=f"{save_dir}/wave_x.pdf"
    )

    # ------ 图 2：残差信号 ------
    plot_wave_only(
        wave=x_minus,
        save_path=f"{save_dir}/wave_x_minus.pdf"
    )

    # ------ 图 3：kernel ------
    plot_wave_only(
        wave=kernel,
        save_path=f"{save_dir}/wave_kernel.pdf"
    )


def delta_real_pred_per_element(pred_tau, label):
    pred_tau = torch.tensor(pred_tau)
    label = torch.tensor(label)
    delta_real_by_pred = []
    delta_pred_by_real = []
    offset = np.round(pred_tau.float().mean() - label.float().mean()).to(int).item()
    label = label + offset
    for idx in label:
        delta = pred_tau - idx
        delta_min, delta_idx = torch.min(delta.abs(), dim=0)
        delta_pred_by_real.append(delta[delta_idx].item())
    for idx in pred_tau:
        delta = idx - label
        delta_min, delta_idx = torch.min(delta.abs(), dim=0)
        delta_real_by_pred.append(delta[delta_idx].item())
    return delta_real_by_pred, delta_pred_by_real


def eval_dist(pred_tau, label):
    if len(pred_tau)==0:
        if len(label)==0:
            return 0, 0, 0, 0
        else:# 完全没有预测到
            return None,None,None,0
    else:
        if len(label)==0:
            return 0,0,0,None
        else:
            pass
    delta_real_by_pred, delta_pred_by_real = delta_real_pred_per_element(pred_tau, label)
    lead_pred_max = -min(delta_pred_by_real) # 超前预测
    lag_pred_max = max(delta_pred_by_real) # 延迟预测
    mean_pred_error = np.mean(np.abs(delta_pred_by_real)) # 平均预测
    surplus_pred = max(np.abs(delta_real_by_pred)) # 多余预测度
    return lead_pred_max, lag_pred_max, mean_pred_error, surplus_pred

def cal_dE_avgTimbre(x, x_minus, n):
    E0 = ((x - 0)**2).mean()
    Em = ((x_minus - 0)**2).mean()
    dE_avgTimbre = E0/Em / n
    return dE_avgTimbre


def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, "item"):  # torch.tensor 或 numpy scalar
        return obj.item()
    elif isinstance(obj, (np.integer, np.float32, np.float64)):
        return obj.item()
    else:
        return obj

class Evaluator:
    def __init__(self):
        self.idx2eval = {}
    def eval(self, idx, pred_tau, label,
             x_minus, x,
             timbre, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        lead_pred_max, lag_pred_max, mean_pred_error, surplus_pred = eval_dist(pred_tau, label)
        print("mean pred error:", mean_pred_error)
        if len(pred_tau) != 0:
            dE_avgTimbre = cal_dE_avgTimbre(x, x_minus, len(pred_tau)).item()
        else:
            dE_avgTimbre = None
        self.idx2eval[idx] = (lead_pred_max, lag_pred_max, mean_pred_error, surplus_pred, dE_avgTimbre)
        # save_audio
        kernel = timbre.generate_kernel().detach().cpu().numpy()
        sr = timbre.sr
        sf.write(os.path.join(save_dir, "pred.wav"), x_minus.cpu().numpy(), sr)
        sf.write(os.path.join(save_dir, "kernel.wav"), kernel, sr)
        sf.write(os.path.join(save_dir, "x.wav"), x.cpu().numpy(), sr)
        # draw_audio_curve
        # plt kernel
        # plt x
        # plt x_minus
        # 论文图 x 在 pred_tau 处 减 kernel 得到 x_minus
        plot_all_three(
            x=x,
            x_minus=x_minus,
            kernel=timbre.generate_kernel(),
            pred_tau=pred_tau,
            save_dir= os.path.join(save_dir, "ppt_figs")
        )
    def save(self, save_path="eval_results.json"):
        """
        Save evaluation table to JSON.
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        clean_dict = to_serializable(self.idx2eval)
        with open(save_path, "w") as f:
            json.dump(clean_dict, f, indent=4)
        print(f"[Evaluator] Saved results to {save_path}")
    def load(self, load_path="eval_results.json"):
        """
        Load evaluation table from JSON.
        """
        if not os.path.exists(load_path):
            print("not load")
            return
            # raise FileNotFoundError(load_path)
        with open(load_path, "r") as f:
            self.idx2eval = json.load(f)
        print(f"[Evaluator] Loaded results from {load_path}")
        return self.idx2eval