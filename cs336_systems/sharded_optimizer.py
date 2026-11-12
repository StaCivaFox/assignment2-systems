from typing import Type

import torch
import torch.distributed as dist

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        self.next_parameter_index = 0
        self.parameter_owners = []
        self.local_optimizer = None

        # The base constructor will call our add_param_group().
        self._initializing = True
        self._initial_local_groups = []

        super().__init__(params, defaults=kwargs)

        self._initializing = False

        if self._initial_local_groups:
            self.local_optimizer = optimizer_cls(
                self._initial_local_groups,
                **kwargs,
            )

        self._initial_local_groups.clear()

    def step(self, closure=None, **kwargs):
        # Update local optimizer
        if closure is None:
            result = self.local_optimizer.step(**kwargs)
        else:
            result = self.local_optimizer.step(closure=closure, **kwargs)
        # Synchronize model parameters among ranks
        with torch.no_grad():
            for parameter, owner in self.parameter_owners:
                dist.broadcast(parameter, src=owner)
        return result

    def add_param_group(self, param_group):
        # Validate and register the complete group.
        super().add_param_group(param_group)

        # Use the normalized group, including its default options.
        group = self.param_groups[-1]
        local_parameters = []

        for parameter in group["params"]:
            owner = self.next_parameter_index % self.world_size
            self.next_parameter_index += 1

            self.parameter_owners.append((parameter, owner))

            if owner == self.rank:
                local_parameters.append(parameter)

        if not local_parameters:
            return

        # Preserve group-specific settings such as lr and weight_decay.
        local_group = {
            key: value
            for key, value in group.items()
            if key != "params"
        }
        local_group["params"] = local_parameters

        if self._initializing:
            self._initial_local_groups.append(local_group)
        elif self.local_optimizer is None:
            # This rank has received its first owned parameters.
            self.local_optimizer = self.optimizer_cls(
                [local_group],
                **self.optimizer_kwargs,
            )
        else:
            self.local_optimizer.add_param_group(local_group)

        
        