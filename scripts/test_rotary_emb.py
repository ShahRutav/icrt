import torch
from typing import Tuple

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"Invalid shape for freqs_cis: {freqs_cis.shape}, x: {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb_method1(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb_method2(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors using complex sinusoidal frequencies.

    Args:
        xq (torch.Tensor): Query tensor of shape (batch, seq_len, num_heads, head_dim).
        xk (torch.Tensor): Key tensor of shape (batch, seq_len, num_heads, head_dim).
        freqs_cis (torch.Tensor): Complex sinusoidal frequencies of shape (seq_len, head_dim // 2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed query and key tensors.
    """
    # Derive freqs_cos and freqs_sin from freqs_cis
    freqs_cos = freqs_cis.real
    freqs_sin = freqs_cis.imag

    # Split xq and xk into real and imaginary parts
    xq_r, xq_i = xq[..., ::2], xq[..., 1::2]
    xk_r, xk_i = xk[..., ::2], xk[..., 1::2]

    # Expand freqs_cos and freqs_sin to match xq_r and xq_i
    freqs_cos = freqs_cos.unsqueeze(1).expand_as(xq_r)
    freqs_sin = freqs_sin.unsqueeze(1).expand_as(xq_r)

    # Apply rotary embeddings
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine real and imaginary parts into the output tensors
    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)
    xk_out = torch.stack((xk_out_r, xk_out_i), dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)

'''
def apply_rotary_emb_method2(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors using separate cosine and sine values.

    Args:
        xq (torch.Tensor): Query tensor of shape (batch, seq_len, num_heads, head_dim).
        xk (torch.Tensor): Key tensor of shape (batch, seq_len, num_heads, head_dim).
        freqs_cos (torch.Tensor): Cosine values of shape (seq_len, head_dim // 2).
        freqs_sin (torch.Tensor): Sine values of shape (seq_len, head_dim // 2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed query and key tensors.
    """
    xq_r, xq_i = xq[..., ::2], xq[..., 1::2]
    xk_r, xk_i = xk[..., ::2], xk[..., 1::2]

    freqs_cos = freqs_cos.unsqueeze(1).expand_as(xq_r)
    freqs_sin = freqs_sin.unsqueeze(1).expand_as(xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)
    xk_out = torch.stack((xk_out_r, xk_out_i), dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
'''

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the complex sinusoidal frequencies for rotary embeddings.

    Args:
        dim (int): The head dimension.
        end (int): The sequence length.
        theta (float): The base of the exponential (default 10000.0).

    Returns:
        torch.Tensor: A complex tensor of shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def precompute_freqs_cos_sin(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the sinusoidal frequencies for rotary embeddings.

    Args:
        dim (int): The head dimension.
        end (int): The sequence length.
        theta (float): The base of the exponential (default 10000.0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine tensors of shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    angles = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(angles)
    freqs_sin = torch.sin(angles)
    return freqs_cos, freqs_sin

def test_apply_rotary_emb():
    # Create fake input tensors with the required shapes
    batch_size = 1
    sequence_length = 256
    num_heads = 12
    head_dim = 64
    complex_dim = head_dim // 2  # Since view_as_complex expects pairs

    # Ensure head_dim is divisible by 2
    assert head_dim % 2 == 0, "head_dim must be divisible by 2 for complex view"

    # Input tensors
    xq = torch.rand(batch_size, sequence_length, num_heads, head_dim, dtype=torch.float32)
    xk = torch.rand(batch_size, sequence_length, num_heads, head_dim, dtype=torch.float32)
    freqs_cis = precompute_freqs_cis(head_dim, sequence_length)
    xq_out1, xk_out1 = apply_rotary_emb_method1(xq, xk, freqs_cis)

    # Run both implementations
    freqs_cos, freqs_sin = precompute_freqs_cos_sin(head_dim, sequence_length)
    xq_out2, xk_out2 = apply_rotary_emb_method2(xq, xk, freqs_cis)

    # import ipdb; ipdb.set_trace()
    # match freqs_cos and freqs_sin values in both methods
    assert torch.allclose(freqs_cis.real, freqs_cos, atol=1e-6), "freqs_cos do not match"
    assert torch.allclose(freqs_cis.imag, freqs_sin, atol=1e-6), "freqs_sin do not match"

    # Assert that outputs are equal
    assert torch.allclose(xq_out1, xq_out2, atol=1e-6), "xq outputs do not match"
    assert torch.allclose(xk_out1, xk_out2, atol=1e-6), "xk outputs do not match"

    print("Test passed! Outputs of both methods match.")
    return xq_out1, xk_out1, xq_out2, xk_out2

# Execute the test
if __name__ == "__main__":
    xq_out1, xk_out1, xq_out2, xk_out2 = test_apply_rotary_emb()

