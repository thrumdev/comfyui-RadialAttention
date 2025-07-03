# Adapted from https://github.com/mit-han-lab/radial-attention

import torch
import flashinfer
import matplotlib.pyplot as plt

def get_indptr_from_mask(mask, query):
    # query shows the device of the indptr
    # indptr (torch.Tensor) - the block index pointer of the block-sparse matrix on row dimension,
    # shape `(MB + 1,)`, where `MB` is the number of blocks in the row dimension.
    # The first element is always 0, and the last element is the number of blocks in the row dimension.
    # The rest of the elements are the number of blocks in each row.
    # the mask is already a block sparse mask
    indptr = torch.zeros(mask.shape[0] + 1, device=query.device, dtype=torch.int32)
    indptr[0] = 0
    row_counts = mask.sum(dim=1).flatten()  # Ensure 1D output [num_blocks_row]
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return indptr

def get_indices_from_mask(mask, query):
    # indices (torch.Tensor) - the block indices of the block-sparse matrix on column dimension,
    # shape `(nnz,),` where `nnz` is the number of non-zero blocks.
    # The elements in `indices` array should be less than `NB`: the number of blocks in the column dimension.
    nonzero_indices = torch.nonzero(mask)
    indices = nonzero_indices[:, 1].to(dtype=torch.int32, device=query.device)
    return indices

def shrink_mask_strict(mask, block_size=128):
    seqlen = mask.shape[0]
    block_num = seqlen // block_size
    mask = mask[:block_num * block_size, :block_num * block_size].view(block_num, block_size, block_num, block_size)
    col_densities = mask.sum(dim = 1) / block_size
    # we want the minimum non-zero column density in the block
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1/3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
    block_mask = frac_high_density_cols > 0.6
    block_mask[0:0] = True
    block_mask[-1:-1] = True
    return block_mask

def pad_qkv(input_tensor, block_size=128):
    """
    Pad the input tensor to be a multiple of the block size.
    input shape: (seqlen, num_heads, hidden_dim)
    """
    seqlen, num_heads, hidden_dim = input_tensor.shape
    # Calculate the necessary padding
    padding_length = (block_size - (seqlen % block_size)) % block_size
    # Create a padded tensor with zeros
    padded_tensor = torch.zeros((seqlen + padding_length, num_heads, hidden_dim), device=input_tensor.device, dtype=input_tensor.dtype)
    # Copy the original tensor into the padded tensor
    padded_tensor[:seqlen, :, :] = input_tensor
    
    return padded_tensor

def get_diagonal_split_mask(i, j, token_per_frame, sparse_type, query):
    assert(sparse_type in ["radial"])
    dist = abs(i - j)
    group = dist.bit_length()
    threshold = 128 # hardcoded threshold for now, which is equal to block-size
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group
    if decay_length >= threshold:
        return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
    
    split_factor = int(threshold / decay_length)
    modular = dist % split_factor
    if modular == 0:
        return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
    else:
        return torch.zeros((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)

def get_window_width(i, j, token_per_frame, sparse_type, num_frames, decay_factor=1, block_size=128, model_type=None):
    assert(sparse_type in ["radial"])
    dist = abs(i - j)
    if model_type == "wan":
        if dist < 1:
            return token_per_frame
        if dist == 1:
            return token_per_frame // 2
    elif model_type == "hunyuan":
        if dist <= 1:
            return token_per_frame
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * decay_factor
    threshold = block_size
    if decay_length >= threshold:
        return decay_length
    else:
        return threshold

def gen_log_mask_shrinked(
        query, 
        s, 
        video_token_num, 
        num_frames, 
        block_size=128, 
        sparse_type="log", 
        decay_factor=0.5, 
        model_type=None,
        num_vace_ref_frames=0
    ):
    """
    A more memory friendly version, we generate the attention mask of each frame pair at a time,
    shrinks it, and stores it into the final result
    """
    final_log_mask = torch.zeros((s // block_size, s // block_size), device=query.device, dtype=torch.bool)
    num_total_frames = num_vace_ref_frames + num_frames
    token_per_frame = video_token_num // num_total_frames
    video_text_border = video_token_num // block_size

    col_indices = torch.arange(0, token_per_frame, device=query.device).view(1, -1)
    row_indices = torch.arange(0, token_per_frame, device=query.device).view(-1, 1)
    final_log_mask[video_text_border:] = True
    final_log_mask[:, video_text_border:] = True
    for i in range(num_total_frames):
        for j in range(num_total_frames):
            if j <= num_vace_ref_frames and i >= num_vace_ref_frames:
                # attention sink: all video frames attend to the first video frame as well as all ref frames.
                local_mask = torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
            elif i < num_vace_ref_frames:
                is_target_ref = j < num_vace_ref_frames
                # reference attention: all ref frames attend only to other reference frames
                local_mask = torch.full((token_per_frame, token_per_frame), is_target_ref, device=query.device, dtype=torch.bool)
            else:
                # otherwise: we generate the mask based on the distance.
                window_width = get_window_width(i, j, token_per_frame, sparse_type, num_frames, decay_factor=decay_factor, block_size=block_size, model_type=model_type)
                local_mask = torch.abs(col_indices - row_indices) <= window_width
                split_mask = get_diagonal_split_mask(i, j, token_per_frame, sparse_type, query)
                local_mask = torch.logical_and(local_mask, split_mask)

            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size
            # get the padded size
            all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
            all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
            padded_local_mask = torch.zeros((all_length_row, all_length_col), device=query.device, dtype=torch.bool)
            padded_local_mask[remainder_row:remainder_row + token_per_frame, remainder_col:remainder_col + token_per_frame] = local_mask
            # shrink the mask
            block_mask = shrink_mask_strict(padded_local_mask, block_size=block_size)
            # set the block mask to the final log mask
            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]
            final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(
                final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end], block_mask)
    print(f"mask sparsity: {1 - final_log_mask.sum() / final_log_mask.numel()}")
    return final_log_mask

class MaskMap:
    def __init__(self, video_token_num, num_frames, num_vace_ref_frames=0):
        self.video_token_num = video_token_num
        self.num_vace_ref_frames = num_vace_ref_frames
        self.num_frames = num_frames
        self.log_mask = None
        
    def query_log_mask(self, query, sparse_type, block_size=128, decay_factor=0.5, model_type=None):
        log_mask = torch.ones((query.shape[0] // block_size, query.shape[0] // block_size), device=query.device, dtype=torch.bool)
        if self.log_mask is None:
            self.log_mask = gen_log_mask_shrinked(
                query, 
                query.shape[0], 
                self.video_token_num, 
                self.num_frames, 
                sparse_type=sparse_type, 
                decay_factor=decay_factor, 
                model_type=model_type, 
                block_size=block_size,
                num_vace_ref_frames=self.num_vace_ref_frames,
            )
        block_bound = self.video_token_num // block_size
        log_mask[:block_bound, :block_bound] = self.log_mask[:block_bound, :block_bound]
        return log_mask
    
class RadialAttention:
    def __call__(
        self, 
        query, 
        key, 
        value, 
        mask_map=None, 
        sparsity_type="radial", 
        block_size=128, 
        decay_factor=1, 
        model_type="wan",
    ):
        """
        Perform radial attention with the given query, key, value tensors and optional mask map.
        Now supports batch sizes > 1 by processing each batch independently and stacking results.
        """
        batch_size, orig_seqlen, num_head, hidden_dim = query.shape

        outputs = []
        for b in range(batch_size):
            q = query[b]
            k = key[b]
            v = value[b]

            q_p = pad_qkv(q, block_size=block_size)
            k_p = pad_qkv(k, block_size=block_size)
            v_p = pad_qkv(v, block_size=block_size)

            mask = mask_map.query_log_mask(q_p, sparsity_type, block_size=block_size, decay_factor=decay_factor, model_type=model_type) if mask_map else None
            seqlen = q_p.shape[0]
            workspace_buffer = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
            bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(
                workspace_buffer,
                backend="fa2",
            )

            indptr = get_indptr_from_mask(mask, q_p)
            indices = get_indices_from_mask(mask, q_p)
            bsr_wrapper.plan(
                indptr=indptr,
                indices=indices,
                M=seqlen,
                N=seqlen,
                R=block_size,
                C=block_size,
                num_qo_heads=num_head,
                num_kv_heads=num_head,
                head_dim=hidden_dim,
            )

            o = bsr_wrapper.run(q_p, k_p, v_p)
            del workspace_buffer
            outputs.append(o[:orig_seqlen])
        # Stack along batch dimension
        return torch.stack(outputs, dim=0)
