import unittest
import torch
from torch import nn
from torch.nn import functional as F
from rmsnorm import RMSNorm
from swiglu import SwiGLU
from rotary_embedding import RotaryEmbedding
from cached_gqa import CachedGQA
from lora_linear import LoRALinear
from masked_mse import masked_mse
from causal_lm_loss import causal_lm_loss
from multi_positive_contrastive_loss import multi_positive_contrastive_loss
from tiny_velocity_model import TinyVelocityModel
from flow_matching_loss import flow_matching_loss
from flow_sampling import sample_flow


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_components(self):
        x = torch.randn(2, 5, 24, requires_grad=True)
        norm = RMSNorm(24)
        expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + norm.eps)
        torch.testing.assert_close(norm(x), expected)
        y = x + SwiGLU(24, 48)(norm(x))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        rope = RotaryEmbedding(8)
        q, k = torch.randn(2, 3, 5, 8), torch.randn(2, 1, 5, 8)
        qr, kr = rope(q, k, torch.arange(5).expand(2, 5))
        torch.testing.assert_close(q.norm(dim=-1), qr.norm(dim=-1))
        torch.testing.assert_close(k.norm(dim=-1), kr.norm(dim=-1))
        self.assertNotIn('inv_freq', rope.state_dict())

    def test_attention_cache_and_masks(self):
        model = CachedGQA(32, 4, 2).eval()
        x = torch.randn(3, 6, 32, requires_grad=True)
        # 包含中间空洞、左 padding、全 padding。
        mask = torch.tensor([[1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
        full, weights, _ = model(x, mask)
        cache, parts = None, []
        for start, end in [(0, 2), (2, 5), (5, 6)]:
            part, _, cache = model(x[:, start:end], mask[:, start:end], cache)
            parts.append(part)
        torch.testing.assert_close(full, torch.cat(parts, 1), atol=1e-5, rtol=1e-5)
        self.assertEqual(cache[0].shape, (3, 2, 6, 8))
        self.assertTrue(torch.isfinite(full).all())
        self.assertEqual(full[mask == 0].abs().max().item(), 0)
        future = torch.ones(6, 6, dtype=torch.bool).triu(1)
        self.assertEqual(weights[:, :, future].abs().max().item(), 0)
        key_blocked = (~mask.bool())[:, None, None, :].expand_as(weights)
        self.assertEqual(weights[key_blocked].abs().max().item(), 0)
        torch.testing.assert_close(weights.sum(-1), mask[:, None, :].expand(3, 4, 6).float())
        changed = x.detach().clone()
        changed[:, 4:] += 100
        other, _, _ = model(changed, mask)
        torch.testing.assert_close(full[:, :4], other[:, :4])
        full.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_lora_gradient_and_merge(self):
        base = nn.Linear(8, 6)
        model = LoRALinear(base, rank=2, alpha=4)
        x = torch.randn(3, 5, 8)
        torch.testing.assert_close(base(x), model(x))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        optimizer.zero_grad()
        model(x).square().mean().backward()
        self.assertEqual(model.A.grad.abs().max().item(), 0)
        self.assertGreater(model.B.grad.norm().item(), 0)
        self.assertIsNone(base.weight.grad)
        optimizer.step()
        torch.testing.assert_close(model(x), model.to_merged_linear()(x))
        # 导出不改变原模型，重复导出不会重复叠加。
        torch.testing.assert_close(model.to_merged_linear()(x), model.to_merged_linear()(x))

    def test_masked_losses(self):
        logits = torch.randn(2, 5, 7, requires_grad=True)
        ids = torch.randint(0, 7, (2, 5))
        mask = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 0, 1, 1]])
        response = torch.ones_like(mask)
        loss, count, _ = causal_lm_loss(logits, ids, mask, response)
        expected = F.cross_entropy(torch.stack([logits[0, 1], logits[0, 2], logits[1, 0], logits[1, 3]]),
                                   torch.stack([ids[0, 2], ids[0, 3], ids[1, 1], ids[1, 4]]))
        self.assertEqual(count, 4)
        torch.testing.assert_close(loss, expected)
        empty, count, acc = causal_lm_loss(logits, ids, mask, torch.zeros_like(mask))
        empty.backward()
        self.assertEqual((empty.item(), count, acc), (0, 0, 0))
        self.assertEqual(logits.grad.abs().sum().item(), 0)
        pred = torch.ones(1, 3, 2, requires_grad=True)
        mse = masked_mse(pred, torch.zeros_like(pred), torch.tensor([[1, 0, 1]]))
        self.assertEqual(mse.item(), 1)
        mse.backward()
        self.assertEqual(pred.grad[0, 1].abs().sum().item(), 0)

    def test_contrastive(self):
        audio, text = torch.randn(4, 8), torch.randn(4, 8)
        pair_ids = torch.arange(4)
        s = F.normalize(audio, dim=-1) @ F.normalize(text, dim=-1).T / 0.1
        expected = (F.cross_entropy(s, pair_ids) + F.cross_entropy(s.T, pair_ids)) / 2
        torch.testing.assert_close(multi_positive_contrastive_loss(audio, text, pair_ids), expected)
        ids = torch.tensor([1, 1, 2, 2])
        perm = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(multi_positive_contrastive_loss(audio, text, ids),
                                   multi_positive_contrastive_loss(audio[perm], text[perm], ids[perm]))

    def test_flow(self):
        model = TinyVelocityModel(4, 3, 16)
        clean, condition = torch.randn(2, 5, 4), torch.randn(2, 3)
        mask = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
        loss = flow_matching_loss(model, clean, condition, mask)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

        class ConstantVelocity(nn.Module):
            def forward(self, x, t, condition, mask):
                return condition[:, :1, None].expand_as(x)

        # 条件速度为 2，无条件为 0，CFG=3，积分区间长度为 1：结果应为 6。
        noise = torch.zeros(2, 5, 4)
        cond = torch.full((2, 3), 2.0)
        out = sample_flow(ConstantVelocity().eval(), noise, cond, mask, num_steps=7)
        expected = torch.full_like(noise, 6).masked_fill(~mask.bool().unsqueeze(-1), 0)
        torch.testing.assert_close(out, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
