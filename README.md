# PyTorch Model Components

根据个人 PyTorch 练习整理，使用 AI 辅助修正和文档整理。学习实现，不是原创算法或生产性能库。

## 阅读方式

一个文件对应一个模块或功能，技术说明直接写在该 Python 文件顶部。
每个文件按「原理和用途 → 维度与注意事项 → 实现 → 最小 print 示例」组织。
无需切换到集中说明文档。组合模块通过明确的 import 复用其他文件。

| 文件 | 内容 |
| --- | --- |
| [rmsnorm.py](rmsnorm.py) | RMSNorm |
| [swiglu.py](swiglu.py) | SwiGLU |
| [rotary_embedding.py](rotary_embedding.py) | RotaryEmbedding |
| [masked_softmax.py](masked_softmax.py) | masked_softmax |
| [cached_gqa.py](cached_gqa.py) | CachedGQA |
| [lora_linear.py](lora_linear.py) | LoRALinear |
| [masked_mse.py](masked_mse.py) | masked_mse |
| [causal_lm_loss.py](causal_lm_loss.py) | causal_lm_loss |
| [multi_positive_contrastive_loss.py](multi_positive_contrastive_loss.py) | multi_positive_contrastive_loss |
| [tiny_velocity_model.py](tiny_velocity_model.py) | TinyVelocityModel |
| [flow_matching_loss.py](flow_matching_loss.py) | flow_matching_loss |
| [flow_sampling.py](flow_sampling.py) | sample_flow |
| [topk_lora_training.py](topk_lora_training.py) | 校准梯度敏感度评分、Top-K 模块选择与 LoRA 训练 |

## 运行

新增：[topk_lora_training.py](topk_lora_training.py) — 根据校准梯度的相对敏感度筛选 Top-K Linear 模块，再注入并训练 LoRA。原理、公式、限制和完整示例均在此文件。
对应测试：`python test_topk_lora.py`。

建议 Python 3.10+；本次在 Python 3.9.12 / PyTorch 2.4.0 的 CPU 环境验证通过。版本覆盖范围见 [VALIDATION.md](VALIDATION.md)。在本目录执行：

```bash
python -m pip install -r requirements.txt
python rmsnorm.py
python cached_gqa.py
python lora_linear.py
python flow_matching_loss.py
```

其他模块也可以直接 `python 文件名.py`。底部示例只打印结果，方便理解；完整性质测试单独运行：

```bash
python -m unittest discover -s . -p "test_*.py" -v
```

## 范围

CPU 合成数据练习；验证范围见 [VALIDATION.md](VALIDATION.md)。默认输入有限、张量设备一致，mask 为 0/1 或 bool。
Attention 显式展开 KV，不宣称性能优势；cache 使用物理列位置，不支持动态 batch 重排或滑动窗口。
LoRA 导出独立合并层，保留原模块继续训练。Flow 示例采用逐帧 MLP，不是完整 DiT，也未提供真实音频效果。
代码已从原练习重整；公开展示前请自行阅读、运行和验证。

## 原理参考

- [PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [GLU Variants](https://arxiv.org/abs/2002.05202)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [GQA](https://arxiv.org/abs/2305.13245)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Flow Matching](https://arxiv.org/abs/2210.02747)

## 许可状态

本仓库暂未指定统一开源许可证，原理参考保留在上文。
