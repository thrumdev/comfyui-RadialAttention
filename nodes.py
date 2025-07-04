import types
import numpy as np
import torch
from comfy.ldm.flux.math import apply_rope
from .attn_mask import RadialAttention, MaskMap, FullAttentionGatherStats, visualize_attention_stats

class WanRadialAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"min": 16, "step": 16}),
                "height": ("INT", {"min": 16, "step": 16}),
                "length": ("INT", {"min": 1, "step": 1}),
            },
            "optional": {
                "vace_trim_latent": ("INT", {"default": 0, "min": 0}),
                "frame1": ("INT", ),
                "frame2": ("INT", ),
                "frame3": ("INT", ),
            }}

    RETURN_TYPES = ("MODEL", "STATSDICT")
    FUNCTION = "run"
    CATEGORY = "RadialAttention"
    DESCRIPTION = "Wan/VACE Radial Attention Patcher"

    def run(
        self, 
        model, 
        width, 
        height, 
        length, 
        vace_trim_latent=0,
        frame1=None,
        frame2=None,
        frame3=None,
    ):
        tokens_per_frame = (width // 16) * (height // 16)

        # First frame is always independent. All remaining frames are temporally compressed by 4.
        num_frames = 1 + (length - 1) // 4
        video_token_num = tokens_per_frame * num_frames
        mask_map = MaskMap(video_token_num, num_frames, vace_trim_latent)

        diffusion_model = model.model.diffusion_model

        frames = [frame1, frame2, frame3]
        interp_frames = [f for f in frames if f is not None]

        global_stats_dict = {}

        # 1. Override attention processor for WAN blocks.
        for block in diffusion_model.blocks:
            block.self_attn.radial_attn = FullAttentionGatherStats("block", interp_frames, global_stats_dict)
            block.self_attn.mask_map = mask_map
            block.self_attn.forward = types.MethodType(radial_attn_forward, block.self_attn)

        # 2. Override attention processor for VACE blocks (if any).
        if hasattr(diffusion_model, "vace_blocks"):
            for block in diffusion_model.vace_blocks:
                block.self_attn.radial_attn = FullAttentionGatherStats("vace_block", interp_frames, global_stats_dict)
                block.self_attn.mask_map = mask_map
                block.self_attn.forward = types.MethodType(radial_attn_forward, block.self_attn)

        return (model, global_stats_dict)
    


# From WanSelfAttention, all else equal except for the radial attention
def radial_attn_forward(self, x, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q, k = apply_rope(q, k, freqs)

    x = self.radial_attn(q, k, v, mask_map=self.mask_map)
    x = x.view(b, s, n * d)  # Reshape back to [B, L, C]
    x = self.o(x)
    return x

class VisualizeAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stats_dict": ("STATSDICT",),
                "name": ("STRING",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "run"
    CATEGORY = "RadialAttention"
    DESCRIPTION = "Visualize attention stats for a given name from a stats dict. Returns two images: temporal and spatial."

    def run(self, stats_dict, name, latent):
        img_temporal, img_spatial = visualize_attention_stats(stats_dict, name)
        # Convert PIL images to numpy arrays (H, W, C), then to torch tensors (1, H, W, C), float32, normalized to [0,1]
        arr_temporal = np.array(img_temporal).astype(np.float32) / 255.0
        arr_spatial = np.array(img_spatial).astype(np.float32) / 255.0
        # Ensure shape is (H, W, C)
        if arr_temporal.ndim == 2:
            arr_temporal = arr_temporal[..., None]
        if arr_spatial.ndim == 2:
            arr_spatial = arr_spatial[..., None]
        tensor_temporal = torch.from_numpy(arr_temporal).unsqueeze(0)  # (1, H, W, C)
        tensor_spatial = torch.from_numpy(arr_spatial).unsqueeze(0)    # (1, H, W, C)
        return (tensor_temporal, tensor_spatial)

NODE_CLASS_MAPPINGS = {
    "WanRadialAttention": WanRadialAttention,
    "VisualizeAttention": VisualizeAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanRadialAttention": "Wan Radial Attention Patcher",
    "VisualizeAttention": "Visualize Attention Stats",
}
