import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import librosa
from timbre import TimbreKernel
import matplotlib.pyplot as plt
pi = np.pi

def bounded(raw, low, high):
    return low + torch.sigmoid(raw) * (high - low)



class RangeParams(nn.Module):
    def __init__(self, p_min, p_max, p_init=0):
        super().__init__()
        self.param_raw = nn.Parameter(torch.Tensor([p_init]))
        self.p_min = p_min
        self.p_max = p_max
    def value(self):
        param_raw = self.param_raw
        p_min = self.p_min
        p_max = self.p_max
        return bounded(param_raw, p_min, p_max)


class DecreaseCurve(nn.Module):
    def __init__(self, num_point=5):
        super().__init__()
        params_raw = [
            RangeParams(0, 1, 0)
            for n in range(num_point,0,-1)
        ] # [5/6, 4/5, 3/4, 2/3, 1/2]
        self.param_raw = nn.ModuleList(params_raw)
    def value(self):
        param_raw = self.param_raw
        params = []
        temp_value = 1
        for pr in param_raw:
            temp_value = temp_value * pr.value()
            params.append(temp_value)
        # print(params)
        # raise NotImplementedError()
        return torch.stack(params,dim=0)[:,0] # (n,)

class XYDecreaseCurve(nn.Module):
    def __init__(self, num_point=5):
        super().__init__()
        x_params_raw = [
            RangeParams(0, 1, -3)
            for n in range(num_point,0,-1)
        ]
        y_params_raw = [
            RangeParams(0, 1, 0)
            for n in range(num_point,0,-1)
        ]
        self.num_point = num_point
        self.x_param_raw = nn.ModuleList(x_params_raw)
        self.y_param_raw = nn.ModuleList(y_params_raw)
    def _value(self):
        x_param_raw = self.x_param_raw
        y_param_raw = self.y_param_raw
        x_params = [] # 控制点x
        y_params = [] # 控制点y
        rest = 1
        for pr in x_param_raw:
            temp_value = 1 - rest + rest * pr.value()
            rest = rest - temp_value
            x_params.append(temp_value)
        x_params = torch.stack(x_params,dim=0)[:,0]
        x_params = x_params / self.num_point
        temp_value = 1
        for pr in y_param_raw:
            temp_value = temp_value * pr.value()
            y_params.append(temp_value)
        y_params = torch.stack(y_params,dim=0)[:,0]
        return x_params, y_params # (n,), (n,)
    def value(self, ts, end_time):
        # ts: (T)
        device = ts.device
        x_params, y_params = self._value()
        # 插值
        x_params = torch.concat([
            torch.zeros(1,device=device),
             x_params,
             torch.ones(1,device=device)
        ])
        x_params *= end_time
        y_params = torch.concat([
            torch.ones(1,device=device),
             y_params,
             torch.zeros(1,device=device)
        ])
        all_y = linear_interp(ts, x_params, y_params) # (T,)
        mask = ts <= end_time
        all_y *= mask
        return all_y


class XYCurve(nn.Module):
    def __init__(self, num_point=5):
        super().__init__()
        x_params_raw = [
            RangeParams(0, 1, 0)
            for n in range(num_point,0,-1)
        ]
        y_params_raw = [
            RangeParams(0, 1, 0)
            for n in range(num_point,0,-1)
        ]
        self.num_point = num_point
        self.x_param_raw = nn.ModuleList(x_params_raw)
        self.y_param_raw = nn.ModuleList(y_params_raw)
    def _value(self):
        x_param_raw = self.x_param_raw
        y_param_raw = self.y_param_raw
        x_params = [] # 控制点x
        y_params = [] # 控制点y
        rest = 1
        for pr in x_param_raw:
            temp_value_delta = rest * pr.value()
            temp_value = 1 - rest + temp_value_delta
            rest = 1 - temp_value
            x_params.append(temp_value)
        x_params = torch.stack(x_params,dim=0)[:,0]
        x_params = x_params / self.num_point
        # temp_value = 1
        for pr in y_param_raw:
            temp_value = pr.value()
            y_params.append(temp_value)
        y_params = torch.stack(y_params,dim=0)[:,0]
        return x_params, y_params # (n,), (n,)
    def value(self, ts, end_time):
        # ts: (T)
        device = ts.device
        x_params, y_params = self._value()
        # 插值
        x_params = torch.concat([
            torch.zeros(1,device=device),
             x_params,
             torch.ones(1,device=device)
        ])
        x_params *= end_time
        y_params = torch.concat([
            torch.ones(1,device=device),
             y_params,
             torch.zeros(1,device=device)
        ])
        all_y = linear_interp(ts, x_params, y_params) # (T,)
        mask = ts <= end_time
        all_y *= mask
        return all_y

        

def linear_interp(x, xp, fp):
    """
    x:  (L,)         要插值的时间序列
    xp: (K,)         控制点时间
    fp: (K,)         控制点数值
    return: (L,)     插值结果
    """
    # 保证维度正确
    x = x.unsqueeze(0)      # (1,L)
    xp = xp.unsqueeze(0)    # (1,K)
    fp = fp.unsqueeze(0)    # (1,K)

    # 查找 x 落在哪两个控制点之间
    idx = torch.searchsorted(xp, x, right=True)  # (1,L)
    idx = torch.clamp(idx, 1, xp.size(1)-1)

    x0 = torch.gather(xp, 1, idx-1)
    x1 = torch.gather(xp, 1, idx)

    y0 = torch.gather(fp, 1, idx-1)
    y1 = torch.gather(fp, 1, idx)

    # 线性插值
    t = (x - x0) / (x1 - x0 + 1e-8)
    res = y0 + t*(y1 - y0)
    return res[0,:]


class KickInterpKernel(TimbreKernel):
    def __init__(self, kernel_len=44100//10//4,sr=44100//10,
                 num_interp=5,controlInit=None):
        super().__init__(kernel_len,sr)
        self.init_params = [kernel_len, sr, num_interp, controlInit]
        self.params = nn.ModuleDict({
            "T": RangeParams(0,1,-3), # 占据kernel_len的比例
            "ls": XYDecreaseCurve(num_interp),
            "fs": XYDecreaseCurve(num_interp),
            "f0": RangeParams(60,200),
            "f1": RangeParams(60,120),
            "l0": RangeParams(0.3,1),
        })
        if controlInit is not None:
            for k,v in controlInit.items():
                module_type = type(self.params[k])
                self.params[k] = module_type(*v)
        self.num_interp = num_interp
        self.intra_params_mean = nn.ParameterDict({
            "phase0": nn.Parameter(torch.Tensor([0.]))
        })
    def get_opt_params(self):
        return self.parameters()
    def get_intra_params(self):
        return self.intra_params_mean.parameters()
    def generate_kernel(self):

        phase0 = self.intra_params_mean["phase0"] # (1,)
        
        ts = self.ts # (L)
        device = ts.device
        L = ts.shape[0]
        T = self.params["T"].value()
        f0 = self.params["f0"].value()
        f1 = self.params["f1"].value()
        l0 = self.params["l0"].value()
        
        end_time = T*ts[-1]
        
        # print(end_time)
        all_ls = self.params["ls"].value(ts, end_time) # (num_point)
        all_fs = self.params["fs"].value(ts, end_time) # (num_point)
        
        all_fs = f0 + (f1-f0) * all_fs # (num_point)
        
        phase_t = (2*pi*all_fs).cumsum(0) * self.dt # (L,)
        y = l0 * all_ls * torch.sin(phase_t + phase0) # (L,)
        return y
    def __str__(self):
        return "kickInterp"
    def clone_init(self):
        return KickInterpKernel(*self.init_params)


class ConcatTimbre(TimbreKernel):
    def __init__(self, timbres, kernel_len=44100//10//4,sr=44100//10):
        super().__init__(kernel_len,sr)
        self.timbres = nn.ModuleList(timbres)
    def get_opt_params(self):
        return self.parameters()
    def get_intra_params(self):
        all_intra_params = []
        for timbre in self.timbres:
            all_intra_params += list(timbre.get_intra_params())
        return all_intra_params
    def generate_kernel(self):

        ts = self.ts
        
        y = torch.zeros(ts.shape[0],device=ts.device)
        temp_start = 0
        for idx in range(len(self.timbres)):
            timbre = self.timbres[idx]
            time_sustain = timbre.params["T"].value() * ts[-1]
            y_temp = timbre.generate_kernel()
            ts_offset = ts + temp_start
            
            # print(ts.shape)
            # print(ts_offset.shape)
            # print(y_temp.shape)
            y_add = linear_interp(ts, ts_offset, y_temp)
            mask = (ts >= temp_start) * (ts < temp_start + time_sustain)
            y_add *= mask
            
            y += y_add
            
            temp_start += time_sustain
        return y

    def __str__(self):
        return "ConcatTimbre"
    def clone_init(self):
        timbre_clones = [
            timbre.clone_init()
            for timbre in self.timbres
        ]
        return ConcatTimbre(timbre_clones, self.kernel_len, self.sr)

'''
kick_interp = KickInterpKernel()
y = kick_interp.generate_prior_kernel()

plt.plot(ts.detach(),all_ls[0].detach())
plt.plot(ts.detach(),y[0].detach())
# plt.plot(ts.detach(),all_fs[0].detach())
plt.savefig("./fig/y.pdf")
plt.close()

sf.write("./fig/y.wav", y[0].detach(), samplerate=4410)
'''

