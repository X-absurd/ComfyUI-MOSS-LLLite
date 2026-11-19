"""Standalone MOSS LLLite nodes using ModelPatcher-owned block replacements."""

import torch
import torch.nn.functional as F

import folder_paths
import comfy.ldm.moss_image.model
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.utils
from comfy_api.latest import io

from .lllite_model import MossLLLiteControl, PREFIX, config_from_metadata, control_autocast, controlled_moss_block

LLLiteType = io.Custom('MOSS_LLLITE_CONTROL')
CONTROL_KEY = 'moss_lllite'


class MossLLLiteLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MossLLLiteLoader', display_name='Load MOSS-LLLite', category='MOSS-LLLite',
            inputs=[io.Combo.Input('name', options=folder_paths.get_filename_list('model_patches'))],
            outputs=[LLLiteType.Output()],
        )

    @classmethod
    def execute(cls, name):
        path = folder_paths.get_full_path_or_raise('model_patches', name)
        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        control = MossLLLiteControl(**config_from_metadata(metadata), operations=comfy.ops.manual_cast,
                                   device=comfy.model_management.unet_offload_device(),
                                   dtype=comfy.utils.weight_dtype(state))
        if any(not key.startswith(PREFIX) for key in state):
            raise ValueError('MOSS LLLite checkpoint contains non-control weights')
        control.load_state_dict({key[len(PREFIX):]: value for key, value in state.items()}, strict=True)
        patcher = comfy.model_patcher.ModelPatcher(control, load_device=comfy.model_management.get_torch_device(),
                                                 offload_device=comfy.model_management.unet_offload_device())
        return io.NodeOutput(patcher)


class MossLLLitePatch:
    def __init__(self, control, image, strength, sigma_start, sigma_end):
        self.control, self.image, self.strength = control, image, strength
        self.sigma_start, self.sigma_end = sigma_start, sigma_end

    def __call__(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        sigmas = transformer_options.get('sigmas')
        if sigmas is not None and not self.sigma_end <= float(sigmas.max()) <= self.sigma_start:
            return executor(x, timestep, context, transformer_options, **kwargs)
        if x.shape[1] != 128:
            raise ValueError('Remove legacy concat Tile LoRAs before applying MOSS LLLite')
        if len(kwargs.get('ref_latents', [])) or kwargs.get('ref_kv') is not None:
            raise ValueError('MOSS LLLite currently supports image generation, without MOSS-Edit reference tokens')
        image = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * 16, x.shape[-2] * 16,
                                          'bicubic', 'center').clamp(0, 1)
        amp = x.device.type == 'cuda' and x.dtype in (torch.float16, torch.bfloat16)
        image = (image.float() * 2 - 1).to(device=x.device, dtype=torch.float32 if amp else x.dtype)
        # Preserve encoder batch shape: BF16 convolution rounding depends on it.
        image = comfy.utils.repeat_to_batch_size(image, x.shape[0])
        with control_autocast(x):
            condition = self.control.model.conditioner(image)
        mask = kwargs.get('context_attention_bias')
        if mask is not None:
            mask = F.pad(mask, (x.shape[-2] * x.shape[-1], 0))[:, None, None]
        options = transformer_options.copy()
        options[CONTROL_KEY] = {**options.get(CONTROL_KEY, {}), self: (condition, mask)}
        return executor(x, timestep, context, options, **kwargs)


class MossLLLiteBlockPatch:
    def __init__(self, patch, index, block, previous):
        self.patch, self.index, self.block, self.previous = patch, index, block, previous

    def __call__(self, args, extra):
        data = args['transformer_options'].get(CONTROL_KEY, {}).get(self.patch)
        if data is None:
            return self.previous(args, extra) if self.previous is not None else extra['original_block'](args)
        condition, mask = data

        def controlled(arguments):
            # Control tensors must not be owned by the backbone memory compiler's allocation scope.
            with comfy.model_prefetch.pause_malloc_graph():
                output = controlled_moss_block(
                    self.block, arguments['img'], arguments['mod'], arguments['pe'],
                    arguments['transformer_options'], self.patch.control.model.blocks[self.index],
                    condition, self.patch.strength, mask,
                )
            return {'img': output}

        if self.previous is not None:
            return self.previous(args, {**extra, 'original_block': controlled})
        return controlled(args)

    def to(self, device_or_dtype):
        return self


class MossLLLiteApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MossLLLiteApply', display_name='Apply MOSS-LLLite', category='MOSS-LLLite',
            description='Anima-style RGB LLLite adapted to MOSS. Strength 0 returns the input model unchanged.',
            inputs=[io.Model.Input('model'), LLLiteType.Input('control'), io.Image.Input('image'),
                    io.Float.Input('strength', default=1.0, min=0.0, max=10.0, step=0.01),
                    io.Float.Input('start_percent', default=0.0, min=0.0, max=1.0, step=0.001),
                    io.Float.Input('end_percent', default=1.0, min=0.0, max=1.0, step=0.001)],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, control, image, strength=1.0, start_percent=0.0, end_percent=1.0):
        if strength == 0 or start_percent >= end_percent:
            return io.NodeOutput(model)
        backbone = model.get_model_object('diffusion_model')
        if not isinstance(backbone, comfy.ldm.moss_image.model.MOSSImage):
            raise ValueError('MOSS LLLite requires a MOSS-Image backbone')
        key = 'diffusion_model.video_embedder.weight'
        weights = model.get_key_patches(key)[key]
        shape = comfy.lora.calculate_shape(weights[1:], weights[0][0], key)
        config = control.model.config
        if shape != (config['model_dim'], 128) or len(backbone.layers) != config['num_blocks']:
            raise ValueError('MOSS LLLite architecture does not match this backbone; remove concat Tile LoRAs')
        sampling = model.get_model_object('model_sampling')
        patch = MossLLLitePatch(control, image[..., :3], strength, float(sampling.percent_to_sigma(start_percent)),
                               float(sampling.percent_to_sigma(end_percent)))
        result = model.clone()
        result.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, patch)
        existing = result.model_options['transformer_options'].get('patches_replace', {}).get('dit', {})
        for index, block in enumerate(backbone.layers):
            result.set_model_patch_replace(MossLLLiteBlockPatch(patch, index, block, existing.get(('single_block', index))),
                                           'dit', 'single_block', index)
        result.set_additional_models(CONTROL_KEY, result.get_additional_models_with_key(CONTROL_KEY) + [control])
        return io.NodeOutput(result)


class MossReferenceAnchor:
    def __init__(self, reference, strength, grid_size):
        self.reference = reference.to(device='cpu').clone()
        self.strength, self.grid_size = strength, grid_size

    def __call__(self, args):
        denoised = args['denoised']
        if self.strength == 0:
            return denoised
        if denoised.shape[1:] != self.reference.shape[1:]:
            raise ValueError('MOSS reference anchor needs a reference latent at the sampling resolution')
        if self.reference.shape[0] not in (1, denoised.shape[0]):
            raise ValueError('MOSS reference anchor needs one reference image or one per output image')
        target = self.reference.to(device=denoised.device, dtype=denoised.dtype)
        grid = (min(self.grid_size, denoised.shape[-2]), min(self.grid_size, denoised.shape[-1]))
        difference = F.adaptive_avg_pool2d(target - denoised, grid)
        difference = F.interpolate(difference, size=denoised.shape[-2:], mode='bilinear', align_corners=False)
        return denoised + difference * self.strength


class MossReferenceAnchorApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MossReferenceAnchorApply', display_name='Apply MOSS Reference Anchor', category='MOSS-LLLite',
            description='Experimental sampling constraint for reference colors and coarse structure. '
                        'Can strongly suppress generated texture and sharpening; disabled by default. '
                        'Encode the control image with the Flux2 VAE at the sampling resolution. '
                        'This is independent of LLLite strength; disable both controls to keep the input model unchanged.',
            inputs=[io.Model.Input('model'), io.Latent.Input('reference'),
                    io.Float.Input('strength', default=0.0, min=0.0, max=1.0, step=0.05),
                    io.Int.Input('grid_size', default=8, min=1, max=32, step=1, advanced=True,
                                 tooltip='More cells preserve finer reference structure as well as color.')],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, reference, strength=0.0, grid_size=8):
        if strength == 0:
            return io.NodeOutput(model)
        if not 0 < strength <= 1 or grid_size < 1:
            raise ValueError('MOSS reference anchor needs strength in (0, 1] and a positive grid size')
        if not isinstance(model.get_model_object('diffusion_model'), comfy.ldm.moss_image.model.MOSSImage):
            raise ValueError('MOSS reference anchor requires a MOSS-Image backbone')
        samples = reference['samples']
        if samples.ndim != 4 or samples.shape[1] != 128:
            raise ValueError('MOSS reference anchor requires a 128-channel Flux2 image latent')
        samples = model.get_model_object('latent_format').process_in(samples)
        result = model.clone()
        result.set_model_sampler_post_cfg_function(MossReferenceAnchor(samples, strength, grid_size))
        return io.NodeOutput(result)
