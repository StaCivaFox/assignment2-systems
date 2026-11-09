import torch
import math
from einops import einsum, rearrange
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # Inputs
    Q_block_ptr = tl.make_block_ptr(
        # Handle batch here first, leaving two dimensional indexing for following
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + stride_kb * batch_index,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Outputs
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q_i
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    query_positions = (
        query_tile_index * Q_TILE_SIZE
        + tl.arange(0, Q_TILE_SIZE)
    ) #(Q_TILE_SIZE,)
    # Initialize intermediate results
    m = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)
    O_acc = tl.zeros((Q_TILE_SIZE, D), tl.float32)
    # Loop over key tiles 1 <= j <= T_k
    T_k = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE
    for j in range(T_k):
        # Load K_j and V_j
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # Compute scores, filtering out possible zero paddings
        scores = tl.dot(Q_i, tl.trans(K_j)) * scale
        key_positions = (
            j * K_TILE_SIZE
            + tl.arange(0, K_TILE_SIZE)
        ) #(K_TILE_SIZE,)
        # Exclude padded keys
        scores = tl.where(
            key_positions[None, :] < N_KEYS,
            scores,
            float("-inf"),
        )
        # Exclude future keys
        if is_causal:
            scores = scores + tl.where(
                query_positions[:, None] >= key_positions[None, :],
                0.0,
                -1e6,
            )
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        P_tiled = tl.exp(scores - m_new[:, None])
        l = alpha * l + tl.sum(P_tiled, axis=1)
        O_acc = alpha[:, None] * O_acc
        O_acc = tl.dot(P_tiled.to(V_j.dtype), V_j, acc=O_acc)
        m = m_new
        # Advance K and V block pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    # Normalize O_i
    O_i = O_acc / l[:, None]
    L_i = m + tl.log(l)
    # Cast dtype of O_i, then store them in global memory
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))

@torch.compile
def attention_backward(Q, K, V, O, dO, L, is_causal):
    d = Q.shape[-1]
    scale = math.sqrt(1 / d)
    scores = einsum(Q, K, "... query d, ... key d -> ... query key") * scale #(..., Nq, Nk)
    D = (O * dO).sum(dim=-1, keepdim=True) #(... Nq 1)
    if is_causal:
        query_positions = torch.arange(Q.shape[-2], device=Q.device) #(seq_len,)
        key_positions = torch.arange(K.shape[-2], device=K.device) #(seq_len,)
        scores = torch.where(
            query_positions[:, None] >= key_positions[None, :],
            scores,
            -1e6,
        )
    P = torch.exp(scores - L.unsqueeze(-1)) #(... Nq Nk)
    dV = einsum(P, dO, "... Nq Nk, ... Nq d -> ... Nk d")
    dP = einsum(dO, V, "... Nq d, ... Nk d -> ... Nq Nk")
    dS = P * (dP - D) #(... Nq Nk)
    if is_causal:
        dS = torch.where(query_positions[:, None] >= key_positions[None, :], dS, 0.0)
    dQ = einsum(dS, K, "... Nq Nk, ... Nk d -> ... Nq d") * scale
    dK = einsum(dS, Q, "... Nq Nk, ... Nq d -> ... Nk d") * scale
    return dQ, dK, dV
    

class FlashAttentionFuncTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        batch_size = Q.shape[0]
        D = Q.shape[-1]
        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device, dtype=torch.float32)
        # Launch the triton kernel
        flash_fwd_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)] (
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1),O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES,
            N_KEYS,
            scale=math.sqrt(1 / D),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = attention_backward(
            Q, K, V, O, grad_O, L, ctx.is_causal
        )
        return dQ, dK, dV, None
    

class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q = 16
        B_k = 16
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        T_q = (N_q + B_q - 1) // B_q
        T_k = (N_k + B_k - 1) // B_k
        d = Q.shape[-1]
        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device, dtype=torch.float32)
        # Core loop
        for i in range(T_q):
            start = i * B_q
            end = min(start + B_q, N_q)
            Q_i = Q[..., start:end, :] #(..., B_q, d)
            query_positions = (
                i * B_q
                + torch.arange(B_q)
            ) #(B_q,)
            # Initialize intermediate results
            m = torch.full(Q_i.shape[:-1], float("-inf"), device=Q_i.device, dtype=torch.float32) #(..., B_q)
            l = torch.zeros(Q_i.shape[:-1], device=Q_i.device, dtype=torch.float32) #(..., B_q)
            O_acc = torch.zeros_like(Q_i)
            for j in range(T_k):
                K_j = K[..., j * B_k:min((j + 1) * B_k, N_k), :] #(..., B_k, d)
                V_j = V[..., j * B_k:min((j + 1) * B_k, N_k), :] #(..., B_k, d)
                scores = einsum(Q_i, K_j, "... query d, ... key d -> ... query key") / math.sqrt(d) # (... B_q, B_k)
                if is_causal:
                    key_positions = (
                        j * B_k
                        + torch.arange(B_k)
                    )
                    allowed = query_positions[:, None] >= key_positions[None, :]
                    scores = torch.where(allowed, scores, -1e6)
                m_new = torch.maximum(m, scores.amax(dim=-1)) #(..., B_q)
                alpha = torch.exp(m - m_new) #(..., B_q)
                P_tiled = torch.exp(scores - m_new.unsqueeze(-1)) #(... B_q, B_k)
                l = alpha * l + P_tiled.sum(dim=-1) #(..., B_q)
                O_acc = alpha.unsqueeze(-1) * O_acc + einsum(P_tiled, V_j.float(), "... query key, ... key d -> ... query d") #(... B_q, d)
                m = m_new
            O_i = O_acc / l.unsqueeze(-1)
            L_i = m + torch.log(l)
            O[..., start:end, :] = O_i.to(O.dtype)
            L[..., start:end] = L_i
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = attention_backward(
            Q, K, V, O, grad_O, L, ctx.is_causal
        )
        return dQ, dK, dV, None
        



