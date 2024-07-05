from torch import Tensor
import torch.nn.functional as F


F.cross_entropy



def cross_entropy_max_z_loss(
    logits: Tensor,
    targets: Tensor,
    z_loss_weight: float,
    ignore_index: int = -100,
) -> Tensor:
    """MaxZLoss.

    from the baichuan2 paper: https://arxiv.org/abs/2309.10305

    .. math::
        z_{loss} = weight z^{2}

    where z is the max logit
    """

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    
    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
    max_logits = logits.max(dim=-1)[0]
    max_logits = max_logits.where(targets != ignore_index, 0)
    # max is not differentiable. But here we just pick the indices of the max
    # value, so it's fine for backpropagation.

    z_loss = z_loss_weight * max_logits.pow(2).mean()
    return loss, z_loss

