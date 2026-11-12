一、估算：xl在2048上下文下要多少显存

条件：batch 4，FP32，朴素注意力实现。B200可用显存约180GB。

项目	怎么算	大约
参数	≈3.4B × 4字节	13.6 GB
梯度	同参数	13.6 GB
AdamW两个状态	2 × 参数	27 GB
固定部分小计	每个参数16字节	≈55 GB
softmax概率（为反向保存）	4 × 32头 × 2048² × 4字节 = 每层2 GiB，× 32层	≈69 GB
FFN的中间激活	每个张量 4×2048×10240×4字节 ≈ 335 MB，每层约3个，× 32层	≈32 GB
d_model大小的激活	每个 4×2048×2560×4字节 ≈ 84 MB，每层约8个，× 32层	≈21 GB
合计		≈177 GB


### fsdp
Keep per-layer metadata such as:

original weight shape and element count
local FP32 master shard
temporary full weight buffer
pending communication handles

| Hook | Intended responsibility |
|---|---|
| Forward pre-hook | Gather and make full weights available |
| Forward hook | Restore sharded representation and release temporary storage safely |
| Backward pre-hook | Reconstruct weights needed for backward |
| Parameter post-accumulation hook | Handle the completed parameter gradient and launch its reduction |

Your overall structure is close. The main corrections are:

- Sharded gradients should use reduce-scatter.
- If you release full weights after forward, you must gather them again before backward.
- Wait for pending reductions in a wrapper method called after loss.backward(), as you did for overlapping DDP. You do not need a module backward hook for that final wait.

Here is your description, completed.

1. Initialization

Your four responsibilities are correct:

1. Assign each rank a shard of each managed weight.
2. Create WeightState, retaining the registered parameter object, local FP32 master shard, original shape, and communication metadata.
3. Make the registered parameter expose its local shard outside computation.
4. Register the necessary hooks.

Also distinguish:

Sharded parameters:   Linear/Embedding weights
Replicated parameters: normalization weights, etc.

Replicated parameters need ordinary gradient synchronization.

2. Before forward

Correct:

1. Optionally cast the local shard to the computation/communication dtype.
2. All-gather the shards to reconstruct the full weight.
3. Make that full weight available to the layer.
4. Run the ordinary forward computation on this rank’s local batch.

The FP32 master shard remains available separately.

3. After forward

Conceptually:

1. Restore the parameter’s local-shard representation.
2. Release the temporary full-weight storage when it is safe to do so.

The second point is essential. This:

parameter.data = state.local_shard

does not release full storage if state.full_weight or autograd still references it.

I should make the limitation of my previous snippets explicit: switching .data demonstrates changing the layer’s active weight, but does not by itself implement correct freeing and reconstruction of autograd’s saved tensors. Reassigning .data later does not automatically redirect saved views to the new buffer. Your storage-management design must account for those saved tensors.

Prefetching future layers is optional. The useful scheduling goal is:

Gather layer i+1 while the GPU computes layer i.

Its placement depends on when work is queued and which streams execute it; it is not inherently the responsibility of the after-forward hook.

4. Before backward

Your description needs one distinction:

parameter.data = state.full_weight

only reuses a full weight that is still present.

If you actually released its storage after forward, this hook must:

1. All-gather the current master shards again.
2. Restore the full-weight storage needed by backward, including any saved views.
3. Make the full weight available to the parameter.

Keeping every layer’s full weight until backward avoids the second gather, but sacrifices the parameter-memory savings of resharding after forward.

5. After parameter-gradient accumulation

Launch reduce-scatter, rather than the DDP all-reduce.

Suppose two ranks computed full gradients on different local batches:

Rank 0: [a0, a1, a2, a3]
Rank 1: [b0, b1, b2, b3]

Reduce-scatter with SUM produces:

Rank 0 receives: [a0+b0, a1+b1]
Rank 1 receives: [a2+b2, a3+b3]

Each rank divides its result by two and obtains the averaged gradient for its own weight shard.

An all-reduce followed by slicing can produce the same mathematical result, but reduce-scatter directly delivers the required shards.

A schematic post-accumulation hook is:

def _gradient_ready(self, parameter):
    state = self.state_by_parameter[parameter]

    # Flatten and pad to shard_numel * world_size.
    # The gradient layout must match the weight-sharding layout.
    full_grad_buffer = prepare_gradient_buffer(parameter.grad, state.info)

    local_grad = torch.empty(
        state.info.shard_numel,
        device=full_grad_buffer.device,
        dtype=full_grad_buffer.dtype,
    )

    work = dist.reduce_scatter_tensor(
        local_grad,
        full_grad_buffer,
        op=dist.ReduceOp.SUM,
        async_op=True,
    )

    # Retain the input/output buffers while communication is outstanding.
    self.pending.append((state, work, local_grad, full_grad_buffer))

prepare_gradient_buffer() is a helper you still need to implement. Padding entries should be zero.

6. After the entire backward pass

Use an explicit wrapper method:

loss.backward()

fsdp_model.finish_gradient_synchronization()

optimizer.step()

For the pending sharded gradients, its essential logic is:

@torch.no_grad()
def finish_gradient_synchronization(self):
    for state, work, local_grad, full_grad_buffer in self.pending:
        work.wait()
        local_grad.div_(self.world_size)

        parameter = state.parameter

        # Replace the full gradient with the averaged local gradient.
        parameter.grad = None
        parameter.data = state.local_shard
        parameter.grad = local_grad.to(dtype=parameter.dtype)

        state.full_weight = None

    self.pending.clear()

    # Also finish and average reductions for replicated parameters.

This is the right place to ensure that, before the optimizer runs:

parameter.data = local FP32 master shard
parameter.grad = averaged FP32 gradient of that same shard

A module’s register_full_backward_hook() is unnecessary for this final coordination and is not a general guarantee that all parameter .grad fields have finished accumulating.

One memory consideration: waiting until the end while retaining every full gradient buffer can create a large peak. This is a straightforward correctness-first schedule; a memory-efficient version must limit outstanding reductions and reclaim buffers as their operations complete.

After optimizer.step(), each rank has updated its own master shard. The next all-gather reconstructs the updated full weight, so FSDP does not need the post-step parameter broadcasts used by your optimizer-state-sharding wrapper