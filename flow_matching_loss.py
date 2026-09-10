"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

## 6. Flow Matching（flow_matching.py）

### 训练：学习每个状态该往哪走

噪声 x0、真实数据 x1、每条样本独立 t∈[0,1)：

`xt=(1-t)*x0+t*x1`；`target_velocity=x1-x0`。

t 的原形状 [B] 用于模型接口，插值时扩成 [B,1,1]，在一条样本的所有帧/特征上共享。预测速度与 target 做 Masked MSE；不把预测速度误认为最终干净音频。

采用线性路径时导数是 x1-x0。换成其他噪声日程，目标速度也要跟着推导，不能机械套用。此例是独立噪声/数据配对的线性条件路径，没有实现额外的最优传输配对算法。

### 无条件分支为什么存在？

训练时按样本概率将 condition 替换为全零，模型由此学习无条件预测。采样时才可使用相同的零条件作为无条件输入。未经这种训练，直接拿零向量跑一次不能保证得到有意义的无条件速度。
"""
import torch
from torch import nn
from torch.nn import functional as F
from masked_mse import masked_mse


def flow_matching_loss(model, clean, condition, mask, condition_drop_prob=0.1):
    if not 0 <= condition_drop_prob <= 1:
        raise ValueError('condition_drop_prob must be in [0,1]')
    b = clean.size(0)
    noise = torch.randn_like(clean)
    t = torch.rand(b, device=clean.device, dtype=clean.dtype)
    time = t[:, None, None]
    xt = ((1 - time) * noise + time * clean)
    xt = xt.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
    # 本示例用全零向量代表无条件，并在训练时实际出现此条件。
    drop = torch.rand(b, device=clean.device) < condition_drop_prob
    train_condition = condition.masked_fill(drop[:, None], 0.0)
    prediction = model(xt, t, train_condition, mask)
    return masked_mse(prediction, clean - noise, mask)


if __name__ == "__main__":
    from tiny_velocity_model import TinyVelocityModel
    model = TinyVelocityModel(4, 3, 16)
    clean, cond = torch.randn(2, 5, 4), torch.randn(2, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    loss = flow_matching_loss(model, clean, cond, mask)
    loss.backward()
    print('loss:', loss.item())
    print('gradient norm:', model.net[0].weight.grad.norm().item())
