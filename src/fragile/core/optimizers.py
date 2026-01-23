from __future__ import annotations

from collections import deque
import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class ThermodynamicAdam(Optimizer):
    """Adam-style optimizer with a global governor for adaptive step scaling.

    Defaults to gradient-RMS varentropy; set use_loss_varentropy=True to use loss history.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        governor_sensitivity: float = 1.0,
        oscillation_brake: float = 0.5,
        history_window: int = 20,
        varentropy_min_history: int = 5,
        min_lr_scale: float = 0.5,
        max_lr_scale: float = 3.0,
        varentropy_eps: float = 1e-8,
        use_loss_varentropy: bool = False,
        cosine_anneal: bool = True,
        cosine_lr_min: float = 1e-6,
        cosine_lr_max: float | None = None,
        cosine_total_steps: int | None = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if len(betas) != 2 or not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= oscillation_brake <= 1.0:
            raise ValueError(f"Invalid oscillation_brake: {oscillation_brake}")
        if history_window < 1:
            raise ValueError(f"Invalid history_window: {history_window}")
        if varentropy_min_history < 1:
            raise ValueError(f"Invalid varentropy_min_history: {varentropy_min_history}")
        if min_lr_scale <= 0.0:
            raise ValueError(f"Invalid min_lr_scale: {min_lr_scale}")
        if max_lr_scale < min_lr_scale:
            raise ValueError(f"Invalid max_lr_scale: {max_lr_scale}")
        if cosine_total_steps is not None and cosine_total_steps <= 0:
            raise ValueError(f"Invalid cosine_total_steps: {cosine_total_steps}")
        if cosine_lr_min < 0.0:
            raise ValueError(f"Invalid cosine_lr_min: {cosine_lr_min}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "governor_sensitivity": governor_sensitivity,
            "oscillation_brake": oscillation_brake,
            "min_lr_scale": min_lr_scale,
            "max_lr_scale": max_lr_scale,
            "varentropy_eps": varentropy_eps,
            "cosine_anneal": cosine_anneal,
            "cosine_lr_min": cosine_lr_min,
            "cosine_lr_max": cosine_lr_max if cosine_lr_max is not None else lr,
            "cosine_total_steps": cosine_total_steps,
        }
        super().__init__(params, defaults)

        self._stat_history: deque[float] = deque(maxlen=history_window)
        self._varentropy_min_history = varentropy_min_history
        self._use_loss_varentropy = use_loss_varentropy
        self.last_lr_scale: float = 1.0
        self.last_lr: float = lr
        self.last_lrs: list[float] = []
        self.last_base_lr: float = lr
        self.last_base_lrs: list[float] = []
        self.last_metric_value: float | None = None
        self.last_varentropy: float | None = None
        self.last_alignment_dot: float | None = None
        self._global_step: int = 0

    @torch.no_grad()
    def step(self, closure=None, *, loss=None):  # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure: Optional callable returning the loss (used when use_loss_varentropy=True).
            loss: Optional precomputed loss value (used when use_loss_varentropy=True).
        """
        loss_value = None
        loss_out = loss
        if loss is not None:
            if self._use_loss_varentropy:
                if torch.is_tensor(loss):
                    loss_value = float(loss.detach().item())
                else:
                    loss_value = float(loss)
        elif closure is not None:
            with torch.enable_grad():
                loss_tensor = closure()
            if torch.is_tensor(loss_tensor):
                if self._use_loss_varentropy:
                    loss_value = float(loss_tensor.detach().item())
            elif self._use_loss_varentropy:
                loss_value = float(loss_tensor)
            loss_out = loss_tensor

        # Metric drives the "thermodynamic governor": loss or grad RMS.
        metric_value = None
        if self._use_loss_varentropy and loss_value is not None:
            metric_value = loss_value
        else:
            metric_value = self._compute_grad_rms()
        self.last_metric_value = metric_value

        lr_scale = 1.0
        if metric_value is not None:
            self._stat_history.append(metric_value)
            if len(self._stat_history) >= self._varentropy_min_history:
                # Varentropy acts like a temperature proxy for adaptive step scaling.
                mean = math.fsum(self._stat_history) / len(self._stat_history)
                variance = math.fsum((val - mean) ** 2 for val in self._stat_history) / len(
                    self._stat_history
                )
                std = math.sqrt(variance)
                group0 = self.param_groups[0]
                varentropy = std / (abs(mean) + group0["varentropy_eps"])
                lr_scale = math.exp(group0["governor_sensitivity"] * varentropy)
                lr_scale = min(lr_scale, group0["max_lr_scale"])
                lr_scale = max(lr_scale, group0["min_lr_scale"])
                self.last_varentropy = varentropy
            else:
                self.last_varentropy = None
        else:
            self.last_varentropy = None

        alignment_dot = None
        alignment_sum = 0.0
        alignment_count = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    msg = "ThermodynamicAdam does not support sparse gradients."
                    raise RuntimeError(msg)
                state = self.state[param]
                if "exp_avg" not in state:
                    continue
                alignment_sum += torch.sum(grad * state["exp_avg"]).item()
                alignment_count += 1
        if alignment_count > 0:
            alignment_dot = alignment_sum
        self.last_alignment_dot = alignment_dot

        # Oscillation brake: reduce step if gradients oppose momentum.
        if alignment_dot is not None and alignment_dot < 0.0:
            lr_scale = self.param_groups[0]["oscillation_brake"]

        self.last_lr_scale = lr_scale
        self.last_lrs = []
        self.last_base_lrs = []

        for group in self.param_groups:
            base_lr = group["lr"]
            cosine_total = group.get("cosine_total_steps")
            if group.get("cosine_anneal") and cosine_total is not None:
                cosine_total = int(cosine_total)
                if cosine_total > 0:
                    lr_max = float(group.get("cosine_lr_max", base_lr))
                    lr_min = float(group.get("cosine_lr_min", 0.0))
                    step = min(self._global_step, cosine_total)
                    progress = step / cosine_total
                    base_lr = lr_min + 0.5 * (lr_max - lr_min) * (
                        1.0 + math.cos(math.pi * progress)
                    )
            lr = base_lr * lr_scale
            self.last_lrs.append(lr)
            self.last_base_lrs.append(base_lr)
            beta1, beta2 = group["betas"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    msg = "ThermodynamicAdam does not support sparse gradients."
                    raise RuntimeError(msg)

                if group["weight_decay"] != 0.0:
                    # Decoupled weight decay (dissipation term).
                    param.add_(param, alpha=-lr * group["weight_decay"])

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Adam moment updates.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

        if self.last_lrs:
            self.last_lr = self.last_lrs[0]
        if self.last_base_lrs:
            self.last_base_lr = self.last_base_lrs[0]
        self._global_step += 1

        return loss_out

    def state_dict(self):  # type: ignore[override]
        state = super().state_dict()
        state["thermo_global_step"] = self._global_step
        return state

    def load_state_dict(self, state_dict):  # type: ignore[override]
        self._global_step = int(state_dict.pop("thermo_global_step", 0))
        super().load_state_dict(state_dict)

    def _compute_grad_rms(self) -> float | None:
        grad_sq_sum = 0.0
        grad_count = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    msg = "ThermodynamicAdam does not support sparse gradients."
                    raise RuntimeError(msg)
                grad_sq_sum += float(grad.detach().float().pow(2).sum().item())
                grad_count += grad.numel()
        if grad_count == 0:
            return None
        return math.sqrt(grad_sq_sum / grad_count)
