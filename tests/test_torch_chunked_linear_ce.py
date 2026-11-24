"""Numerical parity checks for opt-in PyTorch sparse/chunked linear CE."""
import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.algorithm.loss.torch_chunked_linear_ce import (
    torch_chunked_linear_ce_sums,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0])
def test_chunked_linear_ce_matches_dense_reference(dtype: torch.dtype, exponent: float) -> None:
    torch.manual_seed(1234)
    batch, tokens, hidden_size, vocab = 3, 32, 64, 256
    x = torch.randn(batch, tokens, hidden_size, dtype=dtype, requires_grad=True)
    w = torch.randn(vocab, hidden_size, dtype=dtype, requires_grad=True)
    targets = torch.randint(vocab, (batch, tokens))
    targets[0, :] = -100  # Whole sample without supervision
    targets[1, :20] = -100
    targets[2, :5] = -100
    counts = targets.ne(-100).sum(dim=1).float()
    denom = counts.clamp(min=1).pow(1 - exponent).where(counts > 0, 0.0).sum()

    per_token = F.cross_entropy(
        F.linear(x, w).float().flatten(0, 1),
        targets.flatten(),
        ignore_index=-100,
        reduction="none",
    ).reshape(batch, tokens)
    ref_loss = (per_token.sum(dim=1) / counts.clamp(min=1).pow(exponent)).sum() / denom
    ref_dx, ref_dw = torch.autograd.grad(ref_loss, (x, w))

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    got_sums = torch_chunked_linear_ce_sums(x2, w2, targets, chunk_tokens=12)
    got_loss = (got_sums / counts.clamp(min=1).pow(exponent)).sum() / denom
    got_dx, got_dw = torch.autograd.grad(got_loss, (x2, w2))

    assert torch.allclose(got_loss, ref_loss, atol=1e-4, rtol=1e-4)
    if dtype == torch.float32:
        assert torch.allclose(got_dx, ref_dx, atol=1e-5, rtol=1e-5)
        assert torch.allclose(got_dw, ref_dw, atol=1e-5, rtol=1e-5)
    else:
        assert (got_dx - ref_dx).abs().max().item() < 0.002
        assert (got_dw - ref_dw).abs().mean().item() < 0.001


def test_all_ignored_tokens_have_zero_loss_and_gradients() -> None:
    x = torch.randn(2, 11, 16, requires_grad=True)
    w = torch.randn(32, 16, requires_grad=True)
    labels = torch.full((2, 11), -100, dtype=torch.long)
    loss = torch_chunked_linear_ce_sums(x, w, labels, chunk_tokens=8).sum()
    dx, dw = torch.autograd.grad(loss, (x, w))
    assert loss.item() == 0
    assert torch.count_nonzero(dx).item() == 0
    assert torch.count_nonzero(dw).item() == 0
