"""Anima LLLite's RGB encoder and Q/K/V-input adapters, adapted to MOSS tokens.

Reuses the upstream ComfyUI Anima LLLite modules without changing their math.
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/anima/lllite.py
"""

import torch
from torch import nn
import comfy.ops
import comfy.ldm.moss_image.model as moss

from comfy.ldm.anima.lllite import AnimaLLLiteConditioning, AnimaLLLiteModule

FORMAT = 'moss_lllite'
VERSION = '1'
PREFIX = 'moss_lllite.'
TARGETS = ('q', 'k', 'v')


class MossLLLiteControl(nn.Module):
    def __init__(self, model_dim, num_blocks, rank=64, condition_dim=64,
                 condition_embedding_dim=32, condition_resblocks=3,
                 aspp_dilations=(1, 2, 4, 8), *, operations=nn, device=None, dtype=None):
        super().__init__()
        self.config = dict(model_dim=model_dim, num_blocks=num_blocks, rank=rank,
                           condition_dim=condition_dim, condition_embedding_dim=condition_embedding_dim,
                           condition_resblocks=condition_resblocks, aspp_dilations=list(aspp_dilations))
        self.conditioner = AnimaLLLiteConditioning(
            3, condition_dim, condition_embedding_dim, condition_resblocks, aspp_dilations,
            device=device, dtype=dtype, operations=operations,
        )
        self.blocks = nn.ModuleList([
            nn.ModuleDict({target: AnimaLLLiteModule(
                model_dim, condition_embedding_dim, rank,
                device=device, dtype=dtype, operations=operations,
            ) for target in TARGETS}) for _ in range(num_blocks)
        ])

    def metadata(self):
        return {'format': FORMAT, 'moss_lllite.version': VERSION,
                **{f'moss_lllite.{key}': ','.join(map(str, value)) if isinstance(value, list) else str(value)
                   for key, value in self.config.items()}}


def config_from_metadata(metadata):
    if not metadata or metadata.get('format') != FORMAT or metadata.get('moss_lllite.version') != VERSION:
        raise ValueError('Expected a MOSS LLLite checkpoint; beta2, beta3 and Anima weights are incompatible')
    config = {key: int(metadata[f'moss_lllite.{key}']) for key in (
        'model_dim', 'num_blocks', 'rank', 'condition_dim', 'condition_embedding_dim', 'condition_resblocks')}
    config['aspp_dilations'] = [int(value) for value in metadata['moss_lllite.aspp_dilations'].split(',') if value]
    return config


def apply_image_control(module, tokens, condition, strength):
    count = condition.shape[1]
    image = module(tokens[:, :count], condition, strength)
    return torch.cat((image, tokens[:, count:]), dim=1)


def project_qkv(projections, fused, tokens, modules, condition, strength=1.0):
    # Same pre-projection location as Anima/Cosmos compute_qkv's attn1_patch.
    inputs = [apply_image_control(modules[target], tokens, condition, strength) for target in TARGETS]
    if fused:
        # Preserve the native op/quantization dispatch for repacked fused checkpoints.
        return tuple(projections['qkv'](value).chunk(3, dim=-1)[index].clone()
                     for index, value in enumerate(inputs))
    return tuple(projections[target](value) for target, value in zip(TARGETS, inputs))


def controlled_moss_block(block, tokens, mod, pe, transformer_options, modules, condition, strength, mask=None):
    """MOSS's native image-generation block with LLLite before Q/K/V projections."""
    y = moss.modulated_rms_norm(tokens, block.norm1, mod[1], mod[0])
    q, k, v = project_qkv(block.proj, block.fused_qkv, y, modules, condition, strength)
    out = moss.attention(q, k, v, block.q_norm, block.k_norm, pe, block.num_heads,
                         block.comfy_attention, transformer_options, mask=mask)
    tokens = moss.gated_residual(tokens, block.proj['o'](out), mod[2])
    y = moss.modulated_rms_norm(tokens, block.norm2, mod[4], mod[3])
    y = block.proj['ffn_up_gate'](y)
    return moss.gated_residual(tokens, comfy.ops.linear_input_act(block.proj['ffn_down'], y, 'situ'), mod[5])
