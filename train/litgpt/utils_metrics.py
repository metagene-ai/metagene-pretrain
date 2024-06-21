from functools import partial

import torch
from torch.utils.hooks import RemovableHandle


# TARGET_LAYER = ["self_attn", "lm_head"]



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
    if f"activation/{mod_name}" not in log_activations:
        log_activations[f"activation/{mod_name}"] = norm
    else:
        log_activations[f"activation/{mod_name}"] += norm



class ActivationNormMetric:

    def __init__(self, target_layers: list[str]):
        self.target_layers = target_layers
        self.handles: list[RemovableHandle] = []
        self._log_activations: dict[str, torch.Tensor] = {}

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
        return self._log_activations


