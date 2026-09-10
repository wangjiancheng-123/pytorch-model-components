"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

### 采样：Euler + CFG

从初始噪声出发，将 0 到 1 分成 N 段，所以 `dt=1/N`。第 step 步 t=step/N，`torch.full((B,), ...)` 让 batch 内所有样本处于本次积分时间。

`v = v_uncond + scale*(v_cond-v_uncond)`，`x_next=x+dt*v`。

scale=0 使用无条件，scale=1 使用普通条件，scale>1 放大条件差异。每一步用更新的 x 重新预测速度，不能始终把 initial_noise 传回去。每次更新后将 padding 清零。

调用者先 `model.eval()`，函数的 `no_grad()` 只关闭梯度记录。测试用已知恒定速度验证积分结果，可以检查 dt 和 CFG 是否正确；随机初始化 MLP 的输出不能用于评价生成质量。

前提：模型接口为 model(x,t,condition,mask)，无条件输入采用训练时使用过的全零 condition。
示例使用随机初始化模型，仅展示接口和形状，不代表训练效果。
"""
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def sample_flow(model, initial_noise, condition, mask, num_steps=10, guidance_scale=3.0):
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError('num_steps must be a positive integer')
    # 调用者先执行 model.eval()；no_grad 不会自动关闭 dropout。
    valid = mask.bool().unsqueeze(-1)
    x = initial_noise.clone().masked_fill(~valid, 0.0)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((x.size(0),), step * dt, device=x.device, dtype=x.dtype)
        conditional = model(x, t, condition, mask)
        unconditional = model(x, t, torch.zeros_like(condition), mask)
        velocity = unconditional + guidance_scale * (conditional - unconditional)
        x = (x + dt * velocity).masked_fill(~valid, 0.0)
    return x


if __name__ == "__main__":
    from tiny_velocity_model import TinyVelocityModel
    model = TinyVelocityModel(4, 3, 16).eval()
    noise, cond = torch.randn(2, 5, 4), torch.randn(2, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    output = sample_flow(model, noise, cond, mask)
    print('output:', output.shape)
    print('padding max:', output[mask == 0].abs().max().item())
