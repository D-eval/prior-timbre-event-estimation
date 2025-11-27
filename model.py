import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from visualize import plot_intra_energy

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        pass    
    def cal_dE_curve(self, x, timbre, intra,
                        lr=1e-1, eps=1e-4, num_epoch=500):
        # x: (T,), ref audio
        # intra: bool, 决定训练 theta or (theta and phi)
        # timbre: 已确定的音色，用于初始化
        T, = x.shape
        L = timbre.kernel_len
        device = x.device
        dE_all = []
        # print(T-L)
        for t in range(T-L):
            # print(t)
            x_temp = x[t:t+L]
            timbre2 = timbre.clone_init()
            timbre2.load_state_dict(timbre.state_dict())
            timbre2.to(x.device)
            if intra:
                params = timbre.get_intra_params()
            else:
                params = timbre.get_opt_params()
            optimizer = torch.optim.Adam(params, lr=lr)
            loss_last = 100
            d_loss_last = 10
            ela_times = 10
            for epoch in range(num_epoch):
                kernel2 = timbre2.generate_kernel()
                E0 = ((x_temp - 0)**2).sum()
                Ek = ((x_temp - kernel2)**2).sum()
                dE = Ek - E0
                loss = dE
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                d_loss = (loss.item() - loss_last) # d_loss < 0
                has_ela = d_loss * d_loss_last < 0
                d_loss_last = d_loss
                if has_ela:
                    ela_times -= 1
                else:
                    if ela_times < 10:
                        ela_times += 1
                if ela_times == 0:
                    print("震荡")
                    break
                loss_last = loss.detach().item()
                if abs(d_loss) < eps:
                    # print("converge ")
                    break
                if epoch == 0:
                    #print(f"start loss: {loss.item()}")
                    pass
                # print(f"epoch: {epoch}, loss: {loss.item()}")
            dE_all.append(loss.item())
            # print(f"end loss: {loss.item()}")
        return dE_all
    def opt_timbre(self, x_temp, timbre, intra, lr=1e-1, num_epoch=500):
        # x_temp: (L,)
        timbre2 = timbre.clone_init()
        timbre2.load_state_dict(timbre.state_dict())
        timbre2.to(x_temp.device)
        if intra:
            params = timbre2.get_intra_params()
        else:
            params = timbre2.get_opt_params()
        # early stop params
        loss_last = 100
        d_loss_last = 10
        eps = 1e-4
        ela_times = 10
        optimizer = torch.optim.Adam(params, lr=lr)
        for epoch in range(num_epoch):
            kernel = timbre2.generate_kernel()
            E0 = ((x_temp - 0)**2).sum()
            Ek = ((x_temp - kernel)**2).sum()
            dE = Ek - E0
            loss = dE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # early stop
            d_loss = (loss.item() - loss_last) # d_loss < 0
            has_ela = d_loss * d_loss_last < 0
            d_loss_last = d_loss
            if has_ela:
                ela_times -= 1
            else:
                if ela_times < 10:
                    ela_times += 1
            if ela_times == 0:
                #print("震荡")
                break
            loss_last = loss.detach().item()
            if abs(d_loss) < eps:
                # print("converge ")
                break
        return timbre2, loss.item()
    def minus_timbre(self, x, timbre, intra, fig_save_dir,
                     max_num=100, threshold=0, label=None):
        os.makedirs(fig_save_dir, exist_ok=True)
        # x: (T,) target audio
        # timbre: 已确定的音色，用于初始化
        T = x.shape[0]
        dE_curve = self.cal_dE_curve(x, timbre, intra)
        i = 0
        Y = max(max(dE_curve),-min(dE_curve))
        plot_intra_energy(dE_curve, label,
                            os.path.join(fig_save_dir, f"intra_round_{i}.pdf"),
                        x_lim=(0,T),
                        y_lim=(-Y,Y))
        x_minus = x.detach().clone()
        wrong_idx = []
        pred_tau = []
        for i in range(max_num):
            # print(i)
            dE_tensor = torch.tensor(dE_curve).clone()
            if len(wrong_idx) > 0:
                dE_tensor[wrong_idx] = float('inf')
            dE_min, tau = torch.min(dE_tensor, dim=0)
            print('dE_min:', dE_min.item())
            if dE_min >= threshold:
                break
            tau = tau.item()
            timbre2, dE = self.opt_timbre(x_minus[tau:tau+timbre.kernel_len], timbre, intra)
            kernel = timbre2.generate_kernel()
            x_minus_new = x_minus.detach().clone()
            x_minus_new[tau:tau+timbre.kernel_len] -= kernel
            E_new = ((x_minus_new - 0)**2).sum()
            E_old = ((x_minus - 0)**2).sum()
            if E_new >= E_old:
                wrong_idx.append(tau)
                continue
            pred_tau.append(tau)
            x_minus = x_minus_new.detach()
            dE_curve = self.cal_dE_curve(x_minus, timbre, intra)
            plot_intra_energy(dE_curve, label,
                              os.path.join(fig_save_dir, f"intra_round_{i+1}.pdf"),
                            x_lim=(0,T),
                            y_lim=(-Y,Y))
        return pred_tau, x_minus
    def forward(self, x_ref, x_tar, timbre, fig_save_dir, label):
        # x_ref: (T0,) ref audio, 用来确定音色
        # x_tar: (T,) target audio, 需要分析的整个声音
        # 1, 确定音色
        print("计算曲线")
        dE_curve = self.cal_dE_curve(x_ref, timbre, intra=False) # 类间参数
        dE_tensor = torch.tensor(dE_curve)
        dE_min, tau = torch.min(dE_tensor, dim=0)
        if dE_min >= 0:
            print("无音色")
            return None
        print("音色确定")
        timbre, _ = self.opt_timbre(x_ref[tau:tau+timbre.kernel_len], timbre, intra=False)
        # 2, 锁定位置
        print("锁定位置")
        pred_tau, x_minus = self.minus_timbre(x_tar, timbre, intra=True, fig_save_dir=fig_save_dir, label=label) # 类内参数
        return timbre, pred_tau, x_minus
