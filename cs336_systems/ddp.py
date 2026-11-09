import torch
import torch.distributed as dist
from torch import nn
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # Make every replica start with rank0's parameters
        self._broadcast_parameters()

    @torch.no_grad
    def _broadcast_parameters(self):
        for parameter in self.module.parameters():
            dist.broadcast(parameter, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad
    def synchronize_gradients(self):
        for parameter in self.module.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
                async_op=False,
            )
            parameter.grad.div_(self.world_size)

class FlatDDP(NaiveDDP):
    @torch.no_grad
    def synchronize_gradients(self):
        gradients = [
            parameter.grad
            for parameter in self.module.parameters()
            if parameter.grad is not None
        ]
        flat = _flatten_dense_tensors(gradients)
        dist.all_reduce(
            flat,
            op=dist.ReduceOp.SUM,
            async_op=False,
        )
        flat.div_(self.world_size)
        averaged_gradients = _unflatten_dense_tensors(flat, gradients)
        for original, updated in zip(gradients, averaged_gradients, strict=True):
            original.copy_(updated)