import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # 用于平滑，可选
import os
import numpy as np



def plot_intra_energy(intra_dE_all, label,
                      save_path="./fig/intra_dE_all.pdf",
                      smooth_sigma=1.0,
                      x_lim=None, y_lim=None):
    """
    绘制 intra-class residual energy 曲线，用于论文。

    Args:
        intra_dE_all : list or ndarray
        save_path    : 输出 PDF 路径
        smooth_sigma : 高斯平滑 (sigma=0 表示不平滑)
        x_lim        : tuple (xmin, xmax) 或 None
        y_lim        : tuple (ymin, ymax) 或 None
    """
    # 转 numpy
    dE = np.array(intra_dE_all)
    # 平滑（可选）
    if smooth_sigma > 0:
        dE_smooth = gaussian_filter1d(dE, sigma=smooth_sigma)
    else:
        dE_smooth = dE
    # 查找局部最小值
    mins = []
    for i in range(1, len(dE_smooth) - 1):
        if dE_smooth[i] < dE_smooth[i-1] and dE_smooth[i] < dE_smooth[i+1]:
            mins.append(i)
    # ground truth 标签
    # label_idx = read_label()
    # label_idx = label_idx[(label_idx > 7000)*(label_idx < 9000)] - 7000
    # 绘图
    plt.figure(figsize=(7, 3))
    # 标签竖线
    if label is not None:
        for idx in label:
            plt.axvline(x=idx, color='r', linestyle='--', alpha=0.7)
    # 曲线
    plt.plot(dE_smooth, linewidth=1.2, color="black")
    # 局部最小值
    if len(mins) > 0:
        plt.scatter(mins, dE_smooth[mins], color="red", s=20, label="Local minima")
    # 论文风格标签
    plt.xlabel("Frame index")
    plt.ylabel(r"Intra-class residual $\Delta E^{\mathrm{intra}}_t$")
    plt.title("Intra-class Energy Curve")
    # legend
    plt.legend(loc="upper right", frameon=False)
    # 坐标范围控制
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    # 去掉上右框线
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"saved: {save_path}")
