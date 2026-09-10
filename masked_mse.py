"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

### Masked MSE

先逐元素平方误差，再对有效标量求平均：

`loss = sum(valid_squared_error) / max(valid_frames * D, 1)`。

[B,T,D] 若只除以有效帧数，结果是“每帧各维误差之和的平均”，比逐元素平均大 D 倍。两种约定都能定义，但多任务权重必须与约定对应。本例选择逐元素平均。

全 padding 返回与 prediction 保持计算图联系的零标量。Mask 在求和前生效；不能只把 target 设为零却仍将 padding 预测误差算进 loss。
"""
import torch
from torch import nn
from torch.nn import functional as F


def masked_mse(prediction, target, mask):
    # [B,T,D] 按有效标量元素平均，而非仅除以帧数。
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('prediction and target must share shape [B,T,D]')
    valid = mask.bool().unsqueeze(-1)
    prediction = prediction.masked_fill(~valid, 0.0)
    target = target.masked_fill(~valid, 0.0)
    count = mask.bool().sum() * prediction.size(-1)
    return (prediction - target).square().sum() / count.clamp_min(1)


if __name__ == "__main__":
    prediction = torch.ones(2, 3, 4, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    loss = masked_mse(prediction, target, mask)
    print('loss (expected 1):', loss.item())
    loss.backward()
    print('padding gradients:', prediction.grad[mask == 0])
