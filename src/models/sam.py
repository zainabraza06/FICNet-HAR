import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimisation (Foret et al., 2020).

    Usage in training loop
    ----------------------
    # First step: perturb weights toward the sharpest direction
    loss1 = criterion(model(x), y)
    loss1.backward()
    optimizer.first_step(zero_grad=True)

    # Second step: compute gradient at perturbed point, then step back
    loss2 = criterion(model(x), y)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

    Do NOT call optimizer.step() — it raises RuntimeError by design.

    Parameters
    ----------
    params            : model parameters
    base_optimizer_cls: inner optimiser class (e.g. torch.optim.Adam)
    rho               : perturbation ball radius (default 0.05)
    **base_kwargs     : kwargs forwarded to base_optimizer_cls (lr, weight_decay, …)
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, **base_kwargs):
        assert rho >= 0, "rho must be non-negative"
        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights by eps = rho * grad / ||grad||."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore weights and apply the base optimiser step."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.state[p]:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):
        raise RuntimeError(
            "SAM requires explicit first_step() / second_step() calls. "
            "Do not call optimizer.step() directly."
        )
