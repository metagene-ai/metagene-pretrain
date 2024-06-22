from torch import nn, FloatTensor, IntTensor


class MaxZLoss(nn.CrossEntropyLoss):
    """MaxZLoss.

    from the baichuan2 paper: https://arxiv.org/abs/2309.10305

    .. math::
        z_{loss} = weight z^{2}

    where z is the max logit
    """

    def __init__(self, z_loss_weight: float, ignore_index: int) -> None:
        super().__init__(ignore_index=ignore_index)
        self.z_loss_weight = z_loss_weight

    def forward(
        self, logits: FloatTensor, target: FloatTensor
    ) -> tuple[FloatTensor, FloatTensor]:
        loss = super().forward(logits, target)

        max_logits = logits.max(dim=-1)[0]
        max_logits = max_logits.where(target != self.ignore_index, 0)
        # max is not differentiable. But here we just pick the indices of the max
        # value, so it's fine for backpropagation.

        z_loss = self.weight * max_logits.pow(2).mean()
        return loss, z_loss

