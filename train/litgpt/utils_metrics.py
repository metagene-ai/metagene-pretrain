from functools import partial

import torch
from torch.utils.hooks import RemovableHandle


_FSDP_WRAPPED_MODULE = ["_forward_module.","_fsdp_wrapped_module."]

def _remove_fsdp_prefix(name: str) -> str:
    for prefix in _FSDP_WRAPPED_MODULE:
        if prefix in name:
            return name.replace(prefix, "")
    return name

@torch.no_grad()
def log_activations_hook(
    _mod: torch.nn.Module,
    _inp: torch.Tensor,
    outp: torch.Tensor | tuple[torch.Tensor, ...],
    mod_name: str,
    log_activations: dict[str, float],
) -> None:
    if isinstance(outp, tuple):
        outp = outp[0]

    norm = outp.norm(p=2)

    name = _remove_fsdp_prefix(mod_name)

    if f"activation/{name}" not in log_activations:
        log_activations[f"activation/{name}"] = norm
    else:
        log_activations[f"activation/{name}"] += norm


class ActivationNormMetric: 
    """
    This class is used to monitor the norm of the activation of the target layers. 
    It attached hook to the forward of each layer that will log the output, and remove them after.
    """

    def __init__(self, target_layers: list[str], gradient_accumulation_steps: int):
        self.target_layers = target_layers
        self.handles: list[RemovableHandle] = []
        self._log_activations: dict[str, torch.Tensor] = {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def register_metrics_hooks(self, model: torch.nn.Module):
        """
        this function take a torch module, a list of layer name and apply a hook function that
        monitor the output norm of the layers.
        """
        handles = []
        for name, mod in model.named_modules():
            for layer in self.target_layers:
                if name.endswith(layer):
                    handle = mod.register_forward_hook(
                        partial(log_activations_hook, log_activations=self._log_activations, mod_name=name)
                    )
                    handles.append(handle)
                    break
        
        self.handles = handles

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()

    @property
    def log_activations(self) -> dict[str, torch.Tensor]:
        return {k: v / self.gradient_accumulation_steps for k, v in self._log_activations.items()}


def get_grad_norm(model: torch.nn.Module, target_layers: list[str]) -> dict[str, torch.Tensor]:
    """
    this function take a torch module and return a dictionary of the norm of the parameters grad
    """
    norms = {}
    for name, param in model.named_parameters():
        for layer in target_layers:
            if name.endswith(f"{layer}.weight"):
                if param.grad is not None:
                    name = _remove_fsdp_prefix(name)
                    norms[f"grad_norm/{name}"] = param.grad.data.norm(p=2)
    return norms