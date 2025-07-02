import types
from comfy.ldm.flux.math import apply_rope
from .attn_mask import RadialAttention, MaskMap

class WanRadialAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "width": ("INT", {"min": 16, "step": 16}),
                "height": ("INT", {"min": 16, "step": 16}),
                "length": ("INT", {"min": 1, "step": 1}),
            },
            "optional": {
                "vace_trim_latent": ("INT", {"default": 0, "min": 0}),
            }}

    RETURN_TYPES = ("WANVIDEOMODEL",)
    FUNCTION = "run"
    CATEGORY = "vace-radial"
    DESCRIPTION = "Wan/VACE Radial Attention Patcher"

    def run(self, model, width, height, length, vace_trim_latent=0):
        tokens_per_frame = (width // 16) * (height // 16)

        # First frame is always independent. All remaining frames are temporally compressed by 4.
        num_frames = 1 + (length - 1) // 4
        video_token_num = tokens_per_frame * num_frames
        mask_map = MaskMap(video_token_num, num_frames, vace_trim_latent)

        # 1. Override attention processor for WAN blocks.
        for block in model.model.blocks:
            block.self_attn.radial_attn = RadialAttention()
            block.self_attn.mask_map = mask_map
            block.self_attn.forward = types.MethodType(radial_attn_forward, block.self_attn)

        # 2. Override attention processor for VACE blocks (if any).
        if hasattr(model.mode, "vace_blocks"):
            for block in model.model.vace_blocks:
                block.self_attn.radial_attn = RadialAttention()
                block.self_attn.mask_map = mask_map
                block.self_attn.forward = types.MethodType(radial_attn_forward, block.self_attn)

        return (model,)

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
        v = self.v(x).view(b, s, n * d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q, k = apply_rope(q, k, freqs)

    x = self.radial_attn(q, k, v, mask_map=self.mask_map)

    x = self.o(x)
    return x

NODE_CLASS_MAPPINGS = {
    "WanRadialAttention": WanRadialAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanRadialAttention": "Wan Radial Attention Patcher",
}
