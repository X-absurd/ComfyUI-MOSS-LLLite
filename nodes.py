import torch

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

from .model import MossTileControl, PREFIX, config_from_metadata


TileControl = io.Custom("MOSS_TILE_CONTROL")
CONTROL_KEY = "moss_tile_residual"


class MossTileControlLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossTileControlLoader", display_name="Load MOSS Tile Control",
            category="model/loaders",
            inputs=[io.Combo.Input("name", options=folder_paths.get_filename_list("model_patches"))],
            outputs=[TileControl.Output()],
        )

    @classmethod
    def execute(cls, name):
        path = folder_paths.get_full_path_or_raise("model_patches", name)
        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        config = config_from_metadata(metadata)
        control = MossTileControl(
            **config, device=comfy.model_management.unet_offload_device(),
            dtype=comfy.utils.weight_dtype(state), operations=comfy.ops.manual_cast,
        )
        if any(not key.startswith(PREFIX) for key in state):
            raise ValueError("MOSS tile checkpoint contains non-control weights")
        control.load_state_dict({key[len(PREFIX):]: value for key, value in state.items()}, strict=True)
        patcher = comfy.model_patcher.ModelPatcher(
            control, load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        return io.NodeOutput(patcher)


class MossTilePatch:
    def __init__(self, control, latent, strength, sigma_start, sigma_end):
        self.control = control
        self.latent = latent
        self.strength = strength
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

    def __call__(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        if x.shape[1] != self.control.model.config["latent_channels"]:
            raise ValueError("Remove the legacy concat tile LoRA from the MODEL chain before applying MOSS Tile Control")
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None and not self.sigma_end <= float(sigmas.max()) <= self.sigma_start:
            return executor(x, timestep, context, transformer_options, **kwargs)
        latent = comfy.utils.common_upscale(self.latent.to(device=x.device, dtype=x.dtype), x.shape[-1], x.shape[-2], "bilinear", "center")
        latent = comfy.utils.repeat_to_batch_size(latent, x.shape[0])
        condition = self.control.model.conditioner(latent)
        options = transformer_options.copy()
        options[CONTROL_KEY] = {**options.get(CONTROL_KEY, {}), self: condition}
        return executor(x, timestep, context, options, **kwargs)


class MossTileBlockPatch:
    def __init__(self, patch, index, previous):
        self.patch = patch
        self.index = index
        self.previous = previous

    def __call__(self, args, extra):
        out = self.previous(args, extra) if self.previous is not None else extra["original_block"](args)
        condition = args["transformer_options"].get(CONTROL_KEY, {}).get(self.patch)
        if condition is not None:
            # Adapter tensors outlive the backbone block's allocation scope.
            with comfy.model_prefetch.pause_malloc_graph():
                out["img"] = self.patch.control.model.blocks[self.index](out["img"], condition, self.patch.strength)
        return out

    def to(self, device_or_dtype):
        return self


class MossTileControlApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossTileControlApply", display_name="Apply MOSS Tile Control",
            category="model/patch/moss_image",
            description="Controls MOSS structure through a separate residual adapter. Strength 0 returns the input model unchanged.",
            inputs=[
                io.Model.Input("model"), TileControl.Input("control"), io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, control, vae, image, strength=1.0, start_percent=0.0, end_percent=1.0):
        if strength == 0.0 or start_percent >= end_percent:
            return io.NodeOutput(model)
        backbone = model.get_model_object("diffusion_model")
        if not isinstance(backbone, comfy.ldm.moss_image.model.MOSSImage):
            raise ValueError("MOSS Tile Control requires a MOSS-Image model")
        config = control.model.config
        for _strength, weight_patch, *_rest in model.patches.get("diffusion_model.video_embedder.weight", []):
            if (isinstance(weight_patch, tuple) and len(weight_patch) == 2 and weight_patch[0] == "set"
                    and weight_patch[1][0].shape[1] != config["latent_channels"]):
                raise ValueError("Remove the legacy concat tile LoRA from the MODEL chain before applying MOSS Tile Control")
        key = "diffusion_model.video_embedder.weight"
        # Clones share live weights; use the original weight and this patcher's patches.
        projection = model.get_key_patches(key)[key]
        projection_shape = comfy.lora.calculate_shape(projection[1:], projection[0][0], key)
        if (projection_shape != (config["model_dim"], config["latent_channels"])
                or len(backbone.layers) != config["num_blocks"]):
            raise ValueError(f"MOSS tile adapter expects projection {(config['model_dim'], config['latent_channels'])} and {config['num_blocks']} blocks; got {tuple(projection_shape)} and {len(backbone.layers)} blocks")
        latent = vae.encode(image[..., :3])
        if latent.ndim != 4 or latent.shape[1] != config["latent_channels"]:
            raise ValueError("MOSS Tile Control requires the Flux2 VAE (128 latent channels)")
        sampling = model.get_model_object("model_sampling")
        patch = MossTilePatch(control, latent, strength,
                              float(sampling.percent_to_sigma(start_percent)), float(sampling.percent_to_sigma(end_percent)))
        result = model.clone()
        result.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, patch)
        existing = result.model_options["transformer_options"].get("patches_replace", {}).get("dit", {})
        for index in range(config["num_blocks"]):
            previous = existing.get(("single_block", index))
            result.set_model_patch_replace(MossTileBlockPatch(patch, index, previous), "dit", "single_block", index)
        result.set_additional_models(CONTROL_KEY, result.get_additional_models_with_key(CONTROL_KEY) + [control])
        return io.NodeOutput(result)
