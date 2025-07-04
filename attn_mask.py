import torch
import numpy as np
import flashinfer
import matplotlib.pyplot as plt
from PIL import Image
import io

# Adapted from https://github.com/mit-han-lab/radial-attention

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

class FullAttentionGatherStats:
    def __init__(self, interp_frames, name, stats_dict):
        self.name = name
        self.interp_frames = interp_frames
        self.stats_dict = stats_dict

    def __call__(self, query, key, value, mask_map=None, tokens_per_frame=None, num_frames=None):
        """
        Perform full attention with the given query, key, value tensors.
        For each batch, gather two datasets:
          - temporal_distance: (attention score, flag)
          - spatial_distance: (attention score, flag)
        where flag indicates whether the target token is in interp_frames.
        Uses raw attention scores (pre-softmax), not attention weights.
        """
        # Accept tokens_per_frame and num_frames as arguments if mask_map is None or missing attributes
        if mask_map is not None and hasattr(mask_map, 'video_token_num') and hasattr(mask_map, 'num_frames'):
            tokens_per_frame = mask_map.video_token_num // mask_map.num_frames
            num_frames = mask_map.num_frames
        elif tokens_per_frame is None or num_frames is None:
            raise ValueError("tokens_per_frame and num_frames must be provided if mask_map is None or missing attributes.")
        interp_frames = torch.tensor(list(self.interp_frames), device=query.device)

        # Always handle batch dimension
        if query.ndim == 3:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        batch_size, seqlen, num_heads, hidden_dim = query.shape
        attn_outputs = []
        for b in range(batch_size):
            q = query[b]  # [S, N, D]
            k = key[b]
            v = value[b]
            # Compute attention scores: [num_heads, seqlen, seqlen]
            attn_scores = torch.einsum('ihd,jhd->hij', q, k) / (hidden_dim ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [num_heads, seqlen, seqlen]
            attn_out = torch.einsum('hij,jhd->ihd', attn_weights, v)
            attn_outputs.append(attn_out)

            # Vectorized stats computation
            idx = torch.arange(seqlen, device=q.device)
            src_frame = idx // tokens_per_frame  # [S]
            src_spatial = idx % tokens_per_frame  # [S]
            tgt_frame = src_frame.unsqueeze(0).expand(seqlen, seqlen)  # [S, S]
            tgt_spatial = src_spatial.unsqueeze(0).expand(seqlen, seqlen)  # [S, S]
            src_frame_mat = src_frame.unsqueeze(1).expand(seqlen, seqlen)  # [S, S]
            src_spatial_mat = src_spatial.unsqueeze(1).expand(seqlen, seqlen)  # [S, S]
            frame_dist = (src_frame_mat - tgt_frame).abs()  # [S, S]
            spatial_dist = (src_spatial_mat - tgt_spatial).abs()  # [S, S]
            interp_mask = (tgt_frame.unsqueeze(0) == interp_frames.view(-1, 1, 1)).any(0)  # [S, S] bool

            # For each head, flatten all (src, tgt) pairs
            # attn_weights: [num_heads, seqlen, seqlen]
            # frame_dist, spatial_dist, interp_mask: [S, S]
            temporal_dataset = []
            spatial_dataset = []
            for h in range(num_heads):
                attn_flat = attn_weights[h].flatten()  # [S*S] (post-softmax weights)
                frame_dist_flat = frame_dist.flatten()  # [S*S]
                spatial_dist_flat = spatial_dist.flatten()  # [S*S]
                interp_flag_flat = interp_mask.flatten()  # [S*S] bool
                # Each dataset: list of (distance, attn_weight, flag)
                temporal_dataset.append(torch.stack([frame_dist_flat, attn_flat, interp_flag_flat.float()], dim=1).detach().cpu())
                spatial_dataset.append(torch.stack([spatial_dist_flat, attn_flat, interp_flag_flat.float()], dim=1).detach().cpu())
            # Store datasets
            if self.name not in self.stats_dict:
                self.stats_dict[self.name] = {'temporal_dataset': [], 'spatial_dataset': []}
            self.stats_dict[self.name]['temporal_dataset'].extend(temporal_dataset)
            self.stats_dict[self.name]['spatial_dataset'].extend(spatial_dataset)
        attn_outputs = torch.stack(attn_outputs, dim=0)
        return attn_outputs

def visualize_attention_stats(stats_dict, name):
    """
    Visualize the attention stats for a given name with a stats dict as generated by FullAttentionGatherStats.
    Returns two plots as PIL images generated using matplotlib.
    One plot shows the temporal distance (x) against the average attention score (y).
    The other plot shows theof spatial distance (x) against the average attention score (y).
    Both plots have multiple lines: one where flag is True (interp frames) and one where flag is False (non-interp frames).
    They are labeled as "Target in Interpolated Frames" and "Target in Non-Interpolated Frames" respectively.
    Compute averages for each dataset by distance and flag before plotting.
    """
    if name not in stats_dict:
        print(f"No stats found for {name}")
        return None, None
    temporal_dataset = torch.cat(stats_dict[name]['temporal_dataset'], dim=0).numpy()  # [N, 3]
    spatial_dataset = torch.cat(stats_dict[name]['spatial_dataset'], dim=0).numpy()    # [N, 3]

    def plot_avg(dataset, xlabel, title):
        # dataset: [N, 3] (distance, attn_weight, flag)
        fig, ax = plt.subplots(figsize=(6, 4))
        for flag_val, label, color in zip([1.0, 0.0],
                                          ["Target in Interpolated Frames", "Target in Non-Interpolated Frames"],
                                          ["tab:blue", "tab:orange"]):
            mask = dataset[:, 2] == flag_val
            if not np.any(mask):
                continue
            dists = dataset[mask, 0]
            attn = dataset[mask, 1]
            # Compute average attention Weight for each unique distance
            unique_dists = np.unique(dists)
            avg_attn = [attn[dists == d].mean() for d in unique_dists]
            # Avoid log(0) by clipping to a small positive value
            avg_attn = np.clip(avg_attn, 1e-8, None)
            ax.plot(unique_dists, avg_attn, label=label, color=color, marker='o')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average Attention Weight (raw)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which='both', axis='y')
        ax.set_yscale('log')  # Set y-axis to logarithmic scale
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        return img

    img_temporal = plot_avg(temporal_dataset, "Temporal Distance (Frames)", f"Avg Attention Weight vs Temporal Distance ({name})")
    img_spatial = plot_avg(spatial_dataset, "Spatial Distance (Tokens)", f"Avg Attention Weight vs Spatial Distance ({name})")

    # Clear stats for this name
    stats_dict[name]['temporal_dataset'].clear()
    stats_dict[name]['spatial_dataset'].clear()
    return img_temporal, img_spatial



