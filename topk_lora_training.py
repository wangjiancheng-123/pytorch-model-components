"""
Top-K 模块筛选与 LoRA 训练
========================
本文件结构：技术说明 → 筛选函数 → LoRA 注入 → 可运行训练示例。
依赖同目录 lora_linear.py；运行：python topk_lora_training.py。

一、这里选择的是什么？
选择 K 个 nn.Linear 模块，例如 q_proj、v_proj 或 FFN 的 up/down。
不是选权重矩阵中最大的 K 个数，也不是给 token 做 top-k 采样。
基础 W=[out,in] 整体冻结，所选模块新增 A=[rank,in]、B=[out,rank]。
最终只训练 A/B。top_k 是模块数量；rank 是每个模块的低秩维度。

二、为什么先校准打分？
使用少量代表目标任务的数据，在当前基础模型上计算任务 Loss 对 W 的梯度 g。
采用一个容易理解、可复现的启发式指标：

    batch_score(layer) = sqrt(mean((W * dLoss/dW)^2))
    score(layer) = mean(batch_score(layer), calibration_batches)

Loss 的一阶变化约为 sum(g * delta_W)。若参数受到相对大小的扰动，W*g
可以作为局部敏感度线索。RMS 归约按元素数量归一，避免直接求和偏向大矩阵。
分数越大，只代表在当前参数与校准数据上对此指标更敏感，不保证 LoRA 收益更大。
这不是 Fisher 信息矩阵，也不是原先项目筛选脚本的精确复现。

三、流程与技术要点
1. 使用尚未注入 LoRA 的模型，枚举待选 Linear；include_names 可限定候选。
2. 临时打开候选 W 的梯度，在 eval 模式用固定数量的校准 batch 计算梯度。
3. autograd.grad 返回梯度，不累加到 parameter.grad，不执行 optimizer.step。
4. 按平均分排序，选择前 K 个模块，给出完整排名表方便核查。
5. 先创建 LoRA 包装层，然后冻结全模型，仅打开新 A/B 的梯度。
6. 替换完成后创建 optimizer；不能沿用注入前创建的优化器，否则会漏掉 A/B。

四、注意事项
- 建议使用预训练模型与真实目标任务 Loss；示例的随机小模型仅验证流程。
- 各校准 batch 等权平均，建议 batch 大小、有效 token 数接近；不同 Loss
  的归约方式和数值尺度会影响排名。校准集不使用最终测试集。
- 候选 W 临时需要反传，校准峰值显存可能远高于 LoRA 正式训练。
- 本例用普通浮点 nn.Linear；不支持量化/分布式/编译模型、共享模块或共享权重。
  根模型本身若是单个 Linear，请先放入 nn.Sequential 容器。
- 不调用 AMP 或 GradScaler；建议 FP32 校准。缩放过的梯度不能直接与未缩放梯度混比。
- 从未参与 Loss 计算的候选不参与排名；梯度存在但恰好为零的候选得分为零。
- score 函数恢复 requires_grad 与各子模块 train/eval 标记，不更新参数或 .grad。
  用户自定义 forward/回调产生的额外状态变更不在恢复范围内。
- init B=0，所以注入后输出应不变。基础层被冻结不等于 no_grad：梯度仍需
  穿过中间计算回传到前面的 LoRA。
- 本例将所有非 LoRA 参数都冻结，含输出头和归一化参数。
- 若只想按经验选择 q_proj/v_proj，可直接向 attach_lora 传模块名，无需校准。
- 正式实验应比较本筛选法、常规 Q/V、随机同数量模块；参数预算也要考虑
  不同模块尺寸。不能只用校准分数宣称效果提升。
- 保存 LoRA 时还要记录模块名、rank、alpha、基础模型版本；只保存 A/B 而
  不知道注入位置无法可靠重建。本例最后打印可记录的配置，不自动写权重文件。

这是个人练习的 AI 辅助整理版，用于展示模块筛选与训练流程。
参考：LoRA https://arxiv.org/abs/2106.09685
梯度接口：https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
"""
import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F
from lora_linear import LoRALinear


def _linear_candidates(model, include_names=None):
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('Use a model without existing LoRA wrappers')
    # 显式拒绝共享别名，避免同一个层只替换一个路径，或错误打破 tied weights。
    modules = list(model.named_modules(remove_duplicate=False))
    linear = [(name, m) for name, m in modules if isinstance(m, nn.Linear)]
    parameter_ids = [id(p) for _, m in linear for p in m.parameters(recurse=False)]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError('Shared Linear modules/parameters are not supported')
    candidates = {name: m for name, m in linear if name}
    if include_names is not None:
        names = list(include_names)
        if len(names) != len(set(names)) or any(n not in candidates for n in names):
            raise ValueError('Candidate names must be unique named Linear modules')
        candidates = {name: candidates[name] for name in names}
    if not candidates:
        raise ValueError('No named Linear candidates found')
    return candidates


def select_topk_linear_modules(model, calibration_loader, loss_fn,
                               top_k=2, num_batches=4, include_names=None):
    """loss_fn(model,batch) -> scalar Tensor；回调负责将输入移动到正确设备。"""
    candidates = _linear_candidates(model, include_names)
    if not isinstance(top_k, int) or not 1 <= top_k <= len(candidates):
        raise ValueError('top_k must be between 1 and the candidate count')
    if not isinstance(num_batches, int) or num_batches <= 0:
        raise ValueError('num_batches must be positive')
    names = list(candidates)
    weights = [candidates[name].weight for name in names]
    parameter_flags = [(p, p.requires_grad) for p in model.parameters()]
    training_flags = [(m, m.training) for m in model.modules()]
    totals = {name: 0.0 for name in names}
    used_batches = {name: 0 for name in names}
    seen = 0
    try:
        model.eval()
        for p, _ in parameter_flags:
            p.requires_grad_(False)
        for weight in weights:
            weight.requires_grad_(True)
        with torch.enable_grad():
            for batch in itertools.islice(calibration_loader, num_batches):
                loss = loss_fn(model, batch)
                if loss.ndim != 0 or not torch.isfinite(loss).item() or not loss.requires_grad:
                    raise ValueError('Calibration loss must be a finite differentiable scalar')
                grads = torch.autograd.grad(loss, weights, allow_unused=True)
                for name, weight, grad in zip(names, weights, grads):
                    if grad is None:
                        continue
                    score = (weight.detach().float() * grad.detach().float()).square().mean().sqrt().item()
                    if not math.isfinite(score):
                        raise ValueError(f'Nonfinite score for {name}')
                    totals[name] += score
                    used_batches[name] += 1
                seen += 1
    finally:
        for p, flag in parameter_flags:
            p.requires_grad_(flag)
        for module, flag in training_flags:
            module.training = flag
    if seen == 0:
        raise ValueError('Calibration loader is empty')
    # 未激活 batch 的贡献为 0；分母统一使用实际消费的 batch 数。
    ranking = [{'name': name, 'score': totals[name] / seen,
                'active_batches': used_batches[name],
                'weight_shape': tuple(candidates[name].weight.shape)}
               for name in names if used_batches[name] > 0]
    ranking.sort(key=lambda row: (-row['score'], row['name']))
    if len(ranking) < top_k:
        raise ValueError('Fewer active candidates than top_k; inspect the loss/data')
    return [row['name'] for row in ranking[:top_k]], ranking


def attach_lora(model, selected_names, rank=4, alpha=8):
    """原地替换选中的层；调用后只有这些层的 A/B 可训练。"""
    names = list(selected_names)
    if not names:
        raise ValueError('Select at least one module')
    candidates = _linear_candidates(model, names)
    if not isinstance(rank, int) or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('positive integer rank and positive finite alpha required')
    replacements = {name: LoRALinear(layer, rank, alpha) for name, layer in candidates.items()}
    model.requires_grad_(False)
    for name, wrapper in replacements.items():
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper.A.requires_grad_(True)
        wrapper.B.requires_grad_(True)
        wrapper.train(candidates[name].training)
        setattr(parent, child_name, wrapper)
    return model


if __name__ == '__main__':
    torch.manual_seed(42)

    class SmallRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 16))
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            return self.head(self.encoder(x))

    model = SmallRegressor()
    # 示例是合成回归任务；真实音频任务可换成 mask 后的速度场 Loss。
    calibration = [(torch.randn(8, 8), torch.randn(8, 4)) for _ in range(4)]

    def loss_fn(current_model, batch):
        x, target = batch
        return F.mse_loss(current_model(x), target)

    selected, ranking = select_topk_linear_modules(model, calibration, loss_fn, top_k=2)
    print('All candidate scores:')
    for row in ranking:
        print(row)
    print('Selected:', selected)
    probe = calibration[0][0]
    model.eval()
    with torch.no_grad():
        before = model(probe).clone()
    attach_lora(model, selected, rank=4, alpha=8)
    with torch.no_grad():
        print('Injection difference (expected 0):', (before - model(probe)).abs().max().item())

    trainable = [p for p in model.parameters() if p.requires_grad]
    print('Trainable names:', [n for n, p in model.named_parameters() if p.requires_grad])
    print('Trainable / total parameters:', sum(p.numel() for p in trainable),
          sum(p.numel() for p in model.parameters()))
    # 一定在替换后创建优化器。
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    model.train()
    for step, batch in enumerate(calibration):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        print('step / loss / grad norm:', step, loss.item(), norm.item())
    print('Record with checkpoint:', {'selected_modules': selected, 'rank': 4, 'alpha': 8})
