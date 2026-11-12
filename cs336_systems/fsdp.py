from dataclasses import dataclass

import torch
import torch.distributed as dist
from cs336_basics.model import Linear, Embedding

# Helper dataclass to store shard metadata
@dataclass(frozen=True)
class ShardInfo:
    original_shape: tuple[int, ...]
    original_numel: int
    shard_numel: int
    world_size: int

@dataclass
class WeightState:
    parameter: torch.nn.Parameter
    local_shard: torch.Tensor
    info: ShardInfo
    full_buffer: torch.Tensor # Padded, flat all-gather destination
    full_weight: torch.Tensor | None = None # Unpadded, original weight view
    # Additional fields for backward full-weight release
    needs_input_grad: bool = False
    input_backward_done: bool = False
    weight_grad_ready: bool = False
    reduced_grad: torch.Tensor | None = None

# Helper function to create sharded weight for each weight tensor
@torch.no_grad()
def shard_weight(
    full_weight: torch.Tensor,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, ShardInfo]:
    original_shape = tuple(full_weight.shape)
    original_numel = full_weight.numel()
    shard_numel = (original_numel + world_size - 1) // world_size
    # Flatten original weight tensor
    flat = full_weight.detach().reshape(-1)
    # Slice the flattened weight tensor
    start = rank * shard_numel
    end = min(start + shard_numel, original_numel)
    valid_numel = max(end - start, 0)
    local_shard = torch.zeros(shard_numel, device=full_weight.device, dtype=torch.float32)
    if valid_numel > 0:
        local_shard[:valid_numel].copy_(flat[start:end])
    info = ShardInfo(
        original_shape=original_shape,
        original_numel=original_numel,
        shard_numel=shard_numel,
        world_size=world_size,
    )
    return local_shard, info

# Helper function to gather sharded weights before use
# Synchronized implementation first
@torch.no_grad()
def unshard_weight(state: WeightState):
    buffer = state.full_buffer
    # Reserve memory
    required_bytes = buffer.numel() * buffer.element_size()
    storage = buffer.untyped_storage()
    if storage.nbytes() == 0:
        storage.resize_(required_bytes)
    # All-gather into buffer
    send_buffer = state.local_shard.to(dtype=buffer.dtype).contiguous()
    dist.all_gather_into_tensor(
        buffer,
        send_buffer,
        async_op=False,
    )
    # Expose full-weight for parameter
    state.parameter.data = state.full_weight

# Helper function to release full-weight after forward
@torch.no_grad()
def reshard_weight(state: WeightState):
    state.parameter.data = state.local_shard
    buffer = state.full_buffer
    if buffer.untyped_storage().nbytes() == 0:
        return
    if buffer.is_cuda:
        buffer.record_stream(
            torch.cuda.current_stream(buffer.device)
        )
    buffer.untyped_storage().resize_(0)

@torch.no_grad()
def maybe_reshard(state: WeightState):
    if state.input_backward_done and state.weight_grad_ready:
        reshard_weight(state)

# Allocate storage once during initialization, and release and refill through forward and backward
def make_weight_state(parameter, local_shard, info, compute_dtype: torch.dtype | None = None):
    dtype = (
        local_shard.dtype
        if compute_dtype is None
        else compute_dtype
    )
    full_buffer = torch.empty(
        info.shard_numel * info.world_size,
        device=local_shard.device,
        dtype=dtype,
    )
    full_weight = full_buffer[:info.original_numel].view(info.original_shape)
    # Preserve both tensor objects, but initially hold no backing storage
    full_buffer.untyped_storage().resize_(0)
    return WeightState(
        parameter, local_shard, info, full_buffer, full_weight,
    )
    

class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.weight_states = {}
        self.state_by_parameter = {}
        self.hook_handles = []
        self.pending = []
        self.max_pending = 2
        # Delegate sharded weight for each layer for this rank
        for name, layer in self.module.named_modules():
            if isinstance(layer, (Linear, Embedding)):
                parameter = next(layer.parameters(recurse=False))
                local_shard, info = shard_weight(
                    parameter,
                    self.rank,
                    self.world_size,
                )
                parameter.data = local_shard
                state = make_weight_state(
                    parameter, local_shard, info, compute_dtype,
                )
                self.weight_states[layer] = state
                self.state_by_parameter[parameter] = state
                # Register hooks
                self.hook_handles.extend([
                    layer.register_forward_pre_hook(self._before_forward),
                    layer.register_forward_hook(self._after_forward),
                    layer.register_full_backward_pre_hook(self._before_backward),
                    layer.register_full_backward_hook(self._after_backward),
                ])

        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(
                    self._gradient_ready
                )

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _complete_oldest(self):
        # Caller ensures self.pending not empty
        state, work, reduced_grad, full_grad_buffer = self.pending.pop(0)
        work.wait()
        if reduced_grad.is_cuda:
            torch.cuda.current_stream(reduced_grad.device).synchronize()
        reduced_grad.div_(self.world_size)
        if state is not None:
            # At this point, we have weight gradients ready but not a guaranteed input gradients ready
            # so state.parameter could still be full-sized, waiting for _after_backward firing to restore sharded view
            # Therefore, don't immediately assign to state.parameter.grad here
            state.reduced_grad = reduced_grad

    def _before_forward(self, layer, inputs):
        state = self.weight_states[layer]
        state.needs_input_grad = any(
            isinstance(x, torch.Tensor) and x.requires_grad
            for x in inputs
        )
        state.input_backward_done = not state.needs_input_grad
        state.weight_grad_ready = False
        state.reduced_grad = None
        unshard_weight(state)

    def _after_forward(self, layer, inputs, outputs):
        state = self.weight_states[layer]
        reshard_weight(state)

    def _before_backward(self, layer, grad_output):
        state = self.weight_states[layer]
        unshard_weight(state)

    def _after_backward(self, layer, grad_input, grad_output):
        # Release full-weight after its use
        # It's fired after input gradients is ready, but parameter gradient ready not guaranteed
        # So it handles possible release when input gradient ready comes after weight gradient ready,
        # while _gradient_ready handles weight gradient ready comes after input gradient ready.
        state = self.weight_states[layer]
        if state.needs_input_grad:
            state.input_backward_done = True
        maybe_reshard(state)
    

    @torch.no_grad()
    def _gradient_ready(self, parameter: torch.Tensor):
        # Limit outstading reductions BEFORE allocating another full input
        while len(self.pending) >= self.max_pending:
            self._complete_oldest()
        state = self.state_by_parameter.get(parameter)
        if state is None:
            # Indicates replicated parameters
            grad = parameter.grad
            work = dist.all_reduce(
                grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            self.pending.append((None, work, grad, grad))
        else:
            info = state.info
            # Perform reduce-scatter of gradients
            grad = parameter.grad
            flat_grad = grad.detach().reshape(-1)
            full_grad_buffer = torch.zeros(
                info.shard_numel * info.world_size,
                device=grad.device,
                dtype=torch.float32,
            )
            full_grad_buffer[:info.original_numel].copy_(flat_grad)
            # Allocate space for this rank's local gradients shard
            local_grad = torch.empty(
                info.shard_numel,
                device=grad.device,
                dtype=torch.float32,
            )
            # Explicitly free temperary buffer
            # we now only need one copy of full grad held by full_grad_buffer
            parameter.grad = None
            del grad, flat_grad
            # As mentioned above, handle possible full weight release 
            # when weight gradient ready comes after input gradient ready
            state.weight_grad_ready = True
            maybe_reshard(state)
            # Launch reduce-scatter request
            work = dist.reduce_scatter_tensor(
                local_grad,
                full_grad_buffer,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            self.pending.append((state, work, local_grad, full_grad_buffer))

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        while self.pending:
            self._complete_oldest()

        for state in self.weight_states.values():
            if state.reduced_grad is None:
                # Indicates replicated parameters
                continue
            state.parameter.grad = state.reduced_grad.to(state.local_shard.dtype)
            state.reduced_grad = None

    
