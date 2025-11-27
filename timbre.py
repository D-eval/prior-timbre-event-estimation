import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

pi = np.pi


class TimbreKernel(nn.Module, ABC):
    def __init__(self, kernel_len, sr):
        super().__init__()
        self.kernel_len = kernel_len
        self.sr = sr
        self.dt = 1/sr
        self.register_buffer("ts",torch.arange(kernel_len)*self.dt)
        # 类内参数在乐曲中是均匀分布的
        # 记录均值和半径
    # @property
    # @abstractmethod
    # def intra_params_mean(self):
    #     pass
    # @property
    # @abstractmethod
    # def intra_params_radius(self):
    #     pass
    # @property
    # @abstractmethod
    # def params(self):
    #     pass
    # @property
    # def kernel_len(self):
    #     return self.kernel_len
    # @property
    # def dt(self):
    #     return self.dt
    # @property
    # def ts(self):
    #     return self.ts
    @abstractmethod
    def generate_kernel(self):
        # post_param 和 intra_params_mean 有相同的 key
        pass
    
    @abstractmethod
    def get_opt_params(self):
        pass
    
    @abstractmethod
    def get_intra_params(self):
        pass


    def _fit(self, x, epoch=200, lr=1e-1, eps=1e-4):
        # x: (L,)
        # return: {"dE":(1,)}
        L, = x.shape
        optimizer = torch.optim.Adam(self.get_opt_params(), lr=lr)
        # --- 计算输入信号 x 的 FFT 频域幅值 ---
        # 目标 FFT 幅值是常数，可以在循环外计算
        X_fft = torch.fft.fft(x) 
        # 只取幅值 (Magnitude Spectrum)
        X_abs = torch.abs(X_fft) 

        dE_last = 100
        for _ in range(epoch):
            # 1. 生成当前的 kernel (L,)
            kernel = self.generate_kernel() 
            
            # 2. 计算 kernel 的 FFT 频域幅值
            Kernel_fft = torch.fft.fft(kernel)
            Kernel_abs = torch.abs(Kernel_fft)
            
            # 3. 定义 FFT L2 Loss (频域损失)
            # 损失函数 L = || |X_fft| - |Kernel_fft| ||^2
            # 我们的目标是让 kernel 的频域幅值匹配输入 x 的频域幅值
            
            # 使用 MSE 来比较两个幅值谱
            # 损失越小，表示频域匹配度越高
            loss_fft = F.mse_loss(Kernel_abs, X_abs) 
            
            # 保持原有的变量名 dE，但它现在是 FFT 损失
            dE = loss_fft 

            optimizer.zero_grad()
            dE.backward()
            optimizer.step()
            
            print(f"Epoch {_}: Loss (FFT) = {dE.item():.6f}")
            if _ == 0:
                print(f"start opt (FFT Loss): {dE.item():.6f}")
                
            d_dE = (dE.item() - dE_last)
            # 这里的收敛条件需要根据 FFT Loss 的值域重新设定
            # if abs(d_dE) < eps:
            #     break
            dE_last = dE.detach().item()

        last_dE = dE.detach().item()
        print(f"end opt (FFT Loss): {last_dE:.6f}")
        return {"dE": last_dE}

    def fit(self, x, epoch=500, lr=1e-1, eps=1e-4):
        # x: (L,)
        # return: {"dE":(1,)}
        L, = x.shape
        
        optimizer = torch.optim.Adam(self.get_opt_params(), lr=lr)
        dE_last = 100
        for _ in range(epoch):
            kernel = self.generate_kernel() # (L,)
            E0 = ((x - 0)**2).sum()
            Ek = ((x - kernel)**2).sum()
            dE = Ek - E0
            optimizer.zero_grad()
            dE.backward()
            optimizer.step()
            
            print(_,dE.item())
            if _ == 0:
                print("start opt:", dE.item())
            
            d_dE = (dE.item() - dE_last)
            if abs(d_dE) < eps:
                # print("converge ")
                break
            dE_last = dE.detach().item()

        last_dE = dE.detach().item()
        print("end opt:", last_dE)
        # if dE < -0.4:
        #     raise NotImplementedError()
        return {"dE":last_dE}

    def _auto_post_estimate(self,x,power_func,power_metric,
                           num_epoch=100,lr=1e-20):
        B,L,TmL = x.shape
        post_param = {
            k: v.detach()[None,:,None] #.repeat(B,1,TmL)) # 在0维度复制B份
            for k,v in self.intra_params_mean.items()
        }
        return post_param
        
    # 先不考虑它了
    def auto_post_estimate(self,x,power_func,power_metric,
                           num_epoch=100,lr=1e-20):
        # x: (L,T-L+1)
        # return: {"inter":{"k":(1,T-L+1)},"intra":...}
        L,TmL = x.shape
        # 每个可能的midi位置都要优化后验参数
        pos_soft_mask = torch.ones((L,TmL)).to(x.device)
        # 冻结自身
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        post_param = {
            k: nn.Parameter(v.detach()[None,:,None].repeat(1,1,TmL)) # 在0维度复制B份
            for k,v in self.intra_params_mean.items()
        }
        optimizer = torch.optim.Adam(post_param.values(), lr=lr)
        
        for epoch in range(num_epoch):
            loss_silence = self.get_loss_silence(x,post_param,
                                                 pos_soft_mask,
                                                 power_func,power_metric)
            optimizer.zero_grad()
            loss_silence.backward()
            optimizer.step()
        # print("post param mean:", post_param["phase0"][0,0,:])
            # print("post Est",epoch,loss_silence)

        for p in self.parameters():
            p.requires_grad = True

        return post_param
    

    def prior_update(self, post_param, midi):
        # 统计量
        post_param_mean = {
            k: v.mean()
            for k,v in post_param.items()
        }
        prior_param = self.intra_params_mean
        optimizer_prior = torch.optim.Adam(prior_param.values(), lr=lr * 1)
        loss_prior_postMean = 0
        for k in post_param.keys():
            prior_param_k = prior_param[k]
            post_param_k = post_param_mean[k]
            loss_prior_postMean += (prior_param_k - post_param_k)**2
        optimizer_prior.zero_grad()
        loss_prior_postMean.backward()
        optimizer_prior.step()
        print("phase0 prior: ",self.intra_params_mean["phase0"])
        
    def get_loss_silence(self,x,post_param,
                 pos_soft_mask,
                 power_func,power_metric):
        # pos_mask: (B,T-L), 越大表示存在midi
        # x: (B,L,T-L)
        # post_param: {"paramName":(B,)}
        # power_func: (...,L) -> (...)
        # power_metric: (...),(...) -> (...)
        y = self.generate_kernel(post_param) # (B,L)
        x_subbed = (x - y) # (B,L,T-L)
        x_origin_power = power_func(x.permute(0,2,1)) # (B,T-L)
        x_subbed_power = power_func(x_subbed.permute(0,2,1)) # (B,T-L)
        decay_metric = power_metric(x_origin_power, x_subbed_power) # (B,T-L)
        loss_silence = -(decay_metric * pos_soft_mask).mean()
        return loss_silence

