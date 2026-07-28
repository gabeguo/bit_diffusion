import torch
import math
from .loss import _expand_dims
from .loss import grad_wrt_x_t_log_p_base_x_1_cond_x_t, grad_wrt_x_t_log_p_base_x_t_cond_x_0

class SDE:
    def __init__(self, A, score_network):
        self.A = A
        self.score_network = score_network
        return
        
    def sigma(self, t):
        pass

    def phi(self, start, end):
        pass

    def C(self, start, t_a, t_b):
        pass

    def dX_t(
        self, 
        x_t, 
        t, 
        x_cond,
        y,
        dt, 
        reverse=False,
        cfg_scale=0.0,
        ode=False,
        x_start=None, # this is the literal x_0, whereas x_cond could be a different encoding of the x_0 in the forward direction.
    ):
        dB_Q = torch.sqrt(torch.tensor(dt, device=x_t.device)) * torch.randn_like(x_t)
        if cfg_scale > 0:
          score = self.score_network.forward_with_cfg(
            x=x_t,
            t=t,
            y=y, 
            x_cond=x_cond,
            reverse=reverse,
            cfg_scale=cfg_scale,
          )
        else:
          score = self.score_network(
            x=x_t,
            t=t,
            y=y, 
            x_cond=x_cond,
            reverse=reverse,
          )
        sigma_t = _expand_dims(self.sigma(t), x_t)
        if ode:
            assert self.A == 0
            if reverse:
                analytic_base_score = grad_wrt_x_t_log_p_base_x_1_cond_x_t(
                    sde=self,
                    x_t=x_t,
                    t=t,
                    x_1=x_start,
                )
            else:
                analytic_base_score = grad_wrt_x_t_log_p_base_x_t_cond_x_0(
                    sde=self,
                    x_t=x_t,
                    t=t,
                    x_0=x_start,
                )
            assert dt > 0
            dX_t = 1/2 * sigma_t ** 2 * (score - analytic_base_score) * dt
            return dX_t
        dB_P = dB_Q + sigma_t * score * dt
        A = self.A if reverse else -self.A
        dX_t = A * x_t * dt + sigma_t * dB_P
        return dX_t

    def simulate(
        self, 
        x_start, 
        num_steps, 
        reverse=False, 
        return_all=False, 
        cfg_scale=0.0, 
        ode=False,
        x_cond=None,
        y=None,
    ):
        x_t = x_start
        if x_cond is None:
            x_cond = x_start
        if y is None:
            raise ValueError("y is required for simulation")
            y = torch.zeros((x_start.shape[0],), dtype=torch.long, device=x_start.device)
        if reverse:
            all_t = torch.linspace(1, 0, num_steps+1, device=x_start.device)
        else:
            all_t = torch.linspace(0, 1, num_steps+1, device=x_start.device)
        all_x_t = []
        for i in range(num_steps):
            x_t = x_t + self.dX_t(
                x_t=x_t, 
                t=all_t[i:i+1].expand(x_start.shape[0],), 
                x_cond=x_cond, 
                y=y, 
                dt=torch.abs(all_t[i+1] - all_t[i]).item(),
                reverse=reverse,
                cfg_scale=cfg_scale,
                ode=ode and i > 0, # there is singularity at t=0 for ODEs
                x_start=x_start if ode else None, # ODE needs analytic base score
            )
            if return_all:
                all_x_t.append(x_t)
        if return_all:
            return all_x_t
        else:
            return x_t


class FlowMatchingODE(SDE):
    """Deterministic rectified-flow transport with the SDE sampling interface."""

    def __init__(self, score_network, force_unconditional=False):
        super().__init__(A=0, score_network=score_network)
        self.force_unconditional = force_unconditional

    @staticmethod
    def _unsupported(name):
        raise NotImplementedError(
            f"FlowMatchingODE has no {name}; it is a deterministic flow."
        )

    def sigma(self, t):
        self._unsupported("diffusion coefficient sigma")

    def phi(self, start, end):
        self._unsupported("base-process transition phi")

    def C(self, start, t_a, t_b):
        self._unsupported("base-process covariance C")

    def dX_t(
        self,
        x_t,
        t,
        x_cond,
        y,
        dt,
        reverse=False,
        cfg_scale=0.0,
        ode=False,
        x_start=None,
    ):
        if self.force_unconditional:
            if cfg_scale != 0:
                raise ValueError(
                    "CFG is unavailable when FlowMatchingODE forces unconditional sampling."
                )
            velocity = self.score_network(
                x=x_t,
                t=t,
                y=y,
                x_cond=x_cond,
                reverse=False,  # one shared field (theoretically guaranteed); outer reverse only flips dt
                cond_mask=torch.zeros(
                    (x_t.shape[0],), dtype=torch.bool, device=x_t.device
                ),
            )
        elif cfg_scale > 0:
            velocity = self.score_network.forward_with_cfg(
                x=x_t,
                t=t,
                y=y,
                x_cond=x_cond,
                reverse=reverse,
                cfg_scale=cfg_scale,
            )
        else:
            velocity = self.score_network(
                x=x_t,
                t=t,
                y=y,
                x_cond=x_cond,
                reverse=reverse,
            )
        signed_dt = -dt if reverse else dt
        return velocity * signed_dt


class PeriodicVolatilitySDE(SDE):
    def __init__(self, alpha, k, eps, score_network):
        super().__init__(A=0, score_network=score_network)
        self.alpha = alpha
        self.k = k
        self.eps = eps
        return

    def sigma(self, t):
        return self.alpha / 2 * (1 - torch.cos(2 * math.pi * self.k * t)) + self.eps
    
    def phi(self, start, end):
        return torch.ones_like(start)

    def C(self, start, t_a, t_b):
        def integrand(s):
            first_term = (3 * self.alpha ** 2 / 8 + self.alpha * self.eps + self.eps ** 2) * s
            second_term = self.alpha * (self.alpha + 2 * self.eps) / (4 * math.pi * self.k) * torch.sin(2 * math.pi * self.k * s)
            third_term = self.alpha ** 2 / (32 * math.pi * self.k) * torch.sin(4 * math.pi * self.k * s)
            return first_term - second_term + third_term
        
        upper = torch.minimum(t_a, t_b)
        lower = start
        return integrand(upper) - integrand(lower)

class CosineDecayingVolatilitySDE(PeriodicVolatilitySDE):
    def __init__(self, alpha, eps, score_network):
        super().__init__(alpha=alpha, k=0.5, eps=eps, score_network=score_network)
        return
    def sigma(self, t):
        return super().sigma(t - 1)
    def C(self, start, t_a, t_b):
        return super().C(start=start-1, t_a=t_a-1, t_b=t_b-1)

class UniformVolatilitySDE(SDE):
    def __init__(self, A, K, score_network):
        super().__init__(A=A, score_network=score_network)
        self.K = K
        return

    def sigma(self, t):
        return torch.full_like(t, self.K)

    def phi(self, start, end):
        return torch.exp(-self.A * (end - start))

    def C(self, start, t_a, t_b):
        upper = torch.minimum(t_a, t_b)
        if self.A == 0:
            return (self.K ** 2) * (upper - start)

        numerator = (self.K ** 2) * torch.exp(-self.A * (t_a + t_b))
        denominator = 2 * self.A
        window = torch.exp(2 * self.A * upper) - torch.exp(2 * self.A * start)
        return numerator * window / denominator
