# Thanks https://github.com/Gsunshine/py-meanflow/blob/main/meanflow/models/time_sampler.py

import torch

def logit_normal_timestep_sample(P_mean: float, P_std: float, num_samples: int, device: torch.device, eps: float = 1e-4) -> torch.Tensor:
    rnd_normal = torch.randn((num_samples,), device=device)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clip(time, min=eps, max=1.0 - eps)
    return time

def uniform_timestep_sample(batch_size: int, device: torch.device, eps: float = 1e-4) -> torch.Tensor:
    """Uniform on (eps, 1 - eps) to avoid C(...) = 0 at the endpoints."""
    return torch.rand(batch_size, device=device) * (1 - 2 * eps) + eps
