# MOSS-LLLite

Tile control for MOSS-Image in ComfyUI, adapting Anima LLLite's RGB conditioning
and attention input adapters to MOSS. Control strength is adjustable throughout
sampling. At strength **0**, the apply node returns the original input model
without encoding the control image or adding patches.

## Installation

Use a recent [ComfyUI](https://github.com/Comfy-Org/ComfyUI) with native MOSS-Image,
Anima LLLite, and the V3 node API. No additional Python packages are required.

From your ComfyUI directory:

```bash
git clone https://github.com/X-absurd/ComfyUI-MOSS-LLLite.git custom_nodes/ComfyUI-MOSS-LLLite
```

Restart ComfyUI. If you previously installed `ComfyUI-Moss-Tile-Control`, remove
that directory before installing this package to avoid duplicate node IDs.
Existing workflow node IDs are preserved.

## Usage

1. Put a **MOSS-LLLite Tile** `.safetensors` checkpoint in
   `ComfyUI/models/model_patches/`. Model weights are provided separately.
2. Add **Load MOSS-LLLite Tile** and **Apply MOSS-LLLite Tile**, under the
   **MOSS-LLLite** category.
3. Connect the MOSS `MODEL`, loaded controller, and control `IMAGE` to the apply
   node. Connect its output `MODEL` to your sampler.
4. Keep your normal text conditioning and VAE connections. The apply node takes
   RGB pixels directly and has **no VAE input**.

Start with strength `0.5`–`1.0`. Higher values usually preserve more of the control
image's structure. `start_percent` and `end_percent` select the portion of
sampling where control is active; the default is the full interval, `0` to `1`.
Other LoRAs can remain in the model chain, but their effects still apply when
this controller's strength is zero.

The controller does not merge weights into the backbone. It uses ComfyUI's
ModelPatcher, additional-model loading, and block replacement interfaces.

## Compatibility and limitations

- Supports ordinary MOSS-Image generation. MOSS-Edit reference images and cached
  reference KV inputs are currently unsupported.
- Use a controller compatible with your MOSS Base/SFT backbone.
- MOSS-LLLite checkpoints are distinct from Anima LLLite, beta2 concat Tile
  LoRAs, and beta3 residual Tile checkpoints. They are not interchangeable.
- Remove beta2 concat Tile LoRAs before applying MOSS-LLLite.
- Fused BF16 and INT8/ConvRot QKV weights use only the rows needed by each
  controlled projection, while retaining ComfyUI weight casting and patches.
  Other quantization formats retain the original full-projection path.
- Color preservation is experimental. Sampling can still change colors; an
  optional Color Match step can help correct the result.

## Existing beta3 workflows

**Load MOSS Tile Control** and **Apply MOSS Tile Control** are retained for
existing beta3 residual checkpoints. These nodes use the Flux2 VAE input.
Use the MOSS-LLLite nodes above for new LLLite checkpoints.

## Tests

From the ComfyUI directory, using its Python environment:

```bash
python -m unittest discover -s custom_nodes/ComfyUI-MOSS-LLLite/tests -v
```

The tests cover strength-zero bypass, control intervals, checkpoint loading,
native block agreement, fused/unfused projections, and existing beta3 behavior.
They use small CPU models and do not download model weights.

## Credits and license

The conditioning encoder and LLLite adapters reuse the
[Anima LLLite implementation in ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/anima/lllite.py).
MOSS integration uses ComfyUI's native
[MOSS-Image implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/moss_image/model.py).
See [NOTICE](NOTICE) for attribution and [LICENSE](LICENSE) for GPL-3.0 terms.
Model weights have their own applicable licenses.
