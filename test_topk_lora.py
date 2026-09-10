"""Top-K 选择与 LoRA 训练的性质测试。运行 python test_topk_lora.py。"""
import unittest
import torch
from torch import nn
from topk_lora_training import select_topk_linear_modules, attach_lora


class Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 1, bias=False)
        self.b = nn.Linear(2, 1, bias=False)
        self.unused = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(1)

    def forward(self, x):
        return 10 * self.a(x) + self.b(x)


class TopKTests(unittest.TestCase):
    def test_ranking_and_training(self):
        torch.manual_seed(42)
        model = Branches()
        model.b.eval()
        model.b.weight.requires_grad_(False)
        model.a.weight.grad = torch.ones_like(model.a.weight)
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        modes = [m.training for m in model.modules()]
        flags = [p.requires_grad for p in model.parameters()]
        selected, ranking = select_topk_linear_modules(
            model, [torch.ones(3, 2)], lambda m, x: m(x).mean(), top_k=1)
        self.assertEqual(selected, ['a'])
        self.assertEqual([r['name'] for r in ranking], ['a', 'b'])
        self.assertAlmostEqual(ranking[0]['score'], 10, places=5)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        torch.testing.assert_close(model.a.weight.grad, torch.ones_like(model.a.weight))
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, original[n])
        x = torch.ones(3, 2)
        before = model(x).detach()
        attach_lora(model, selected, rank=1, alpha=2)
        torch.testing.assert_close(before, model(x))
        self.assertEqual([n for n, p in model.named_parameters() if p.requires_grad], ['a.A', 'a.B'])
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
        self.assertGreater(model.a.B.abs().sum().item(), 0)
        torch.testing.assert_close(model.a.base.weight, original['a.weight'])
        torch.testing.assert_close(model.b.weight, original['b.weight'])

    def test_empty_loader_restores_flags(self):
        model = Branches().eval()
        model.requires_grad_(False)
        with self.assertRaises(ValueError):
            select_topk_linear_modules(model, [], lambda m, x: m(x).mean())
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
