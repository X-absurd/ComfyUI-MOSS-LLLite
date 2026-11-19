# MOSS-LLLite

LLLite control for MOSS-Image in ComfyUI, adapting Anima LLLite's RGB conditioning
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

For Tile restoration, the recommended final release is **V1**, the final design
developed through four iterations. Use **Load MOSS-LLLite** and **Apply MOSS-LLLite** for these
weights. V1 refines details while preserving subject features, with good color
retention in the author's visual tests; exact color matching is not guaranteed.

1. Put a **MOSS-LLLite** `.safetensors` checkpoint in
   `ComfyUI/models/model_patches/`. Model weights are provided separately.
2. Add **Load MOSS-LLLite** and **Apply MOSS-LLLite**, under the
   **MOSS-LLLite** category.
3. Connect the MOSS `MODEL`, loaded controller, and control `IMAGE` to the apply
   node. Connect its output `MODEL` to your sampler.
4. Keep your normal text conditioning and VAE connections. The apply node takes
   RGB pixels directly and has **no VAE input**.

The control task depends on how the checkpoint was trained. Tile checkpoints
support image restoration; other tasks require matching MOSS-LLLite checkpoints
and control images. Renaming the nodes does not add new tasks to existing weights.

Start with strength `0.5`–`1.0`. Higher values usually preserve more of the control
image's structure. `start_percent` and `end_percent` select the portion of
sampling where control is active; the default is the full interval, `0` to `1`.
Other LoRAs can remain in the model chain, but their effects still apply when
this controller's strength is zero.

The controller does not merge weights into the backbone. It uses ComfyUI's
ModelPatcher, additional-model loading, and block replacement interfaces.

## Experimental color guide

**Update recommended if preserving reference colors matters to your workflow.**
Existing MOSS-LLLite weights remain compatible; no model download is needed.
For a Git installation, run `git pull` in `custom_nodes/ComfyUI-MOSS-LLLite`,
then restart ComfyUI and refresh the browser. For a manual installation, replace
the node folder with the latest package before restarting.

**Apply MOSS Color Guide** corrects the final sampled latent using the colors
actually decoded by the supplied Flux2 VAE. It measures coarse RGB errors,
probes three calibrated color directions, and updates only those directions.
Local comparisons preserve generated texture substantially better than the
full-channel Reference Anchor. Brightness, contrast and some detail can still
change; exact RGB preservation is not guaranteed. This is inference color
compensation, not a repaired or retrained LLLite checkpoint. Texture can remain
visibly intact while reduced local contrast makes the image look slightly less
sharp. Compare both the overall image and native-size detail crops.

1. Resize the reference to the sampling resolution, in multiples of 16.
2. Connect this image to `image`, and **flux2-vae** to `vae`.
3. Connect **Apply MOSS-LLLite** to the guide's `model`, then the guide to the
   sampler. Keep **Reference Anchor** disabled. Try strength `1`, grid `8`,
   iterations `2`; reduce strength for a partial color correction.
4. Decode with the same VAE using ordinary **VAE Decode**. Tiled decoding can
   introduce separate drift after correction.

| Parameter | Meaning | Starting point |
| --- | --- | --- |
| `strength` | Amount of color correction, independent of LLLite strength; `0` disables it | `1.0` for stronger color retention; try `0.8` if the result looks softer |
| `iterations` | Number of decoder-feedback correction rounds; more rounds cost more time | `2`; try `3` only for remaining color error |
| `grid_size` | Coarse color measurement grid, normally `grid_size × grid_size`; not image tiles | `8`; try `16` for local color shifts or small subjects on large canvases |

The guide is disabled by default. Strength `0` returns its input model
unchanged and skips calibration. Its strength is independent of LLLite: both
must be disabled to remove both reference constraints. Higher strength is not
always better. For small subjects on large canvases, inspect local color and
try grid `16`; a full-frame average can hide a local failure.

This costs additional time: calibration uses six small VAE encodes, followed
by `1 + 4 × iterations` full VAE decodes after sampling. In local private tests,
two iterations added about 3 seconds at 1.5 MP and 38 seconds at 9.5 MP; timings
depend on hardware, decoder settings and offloading. Increase to three iterations
only if the remaining color shift justifies the cost.

No additional weights are required. RGB is used to measure error; the correction
changes the latent, not the final decoded RGB pixels. For tiled sampling, use
the current tile image, not a full-frame reference to a smaller tile sampler.

## Experimental reference anchor

**Apply MOSS Reference Anchor** adds an explicit reference constraint during
denoising. It constrains low-frequency latent features, including colors and
coarse structure. It can strongly suppress newly generated texture and make
an upscaled image look as soft as its reference. It is disabled by default
and is **not recommended for detail enhancement**. This is a sampling
constraint, not an RGB Color Match operation or a newly trained LLLite model.
Existing MOSS Tile checkpoints can be used with it.

1. Resize the reference image to the final sampling resolution and encode it
   with the **Flux2 VAE** using **VAE Encode**.
2. Connect that `LATENT` to the anchor's `reference` input and the sampler's
   `latent_image` input. Their spatial dimensions must match.
3. Connect the model after **Apply MOSS-LLLite** to the anchor, then connect the
   anchor's output model to the sampler. Strength `0` leaves the input model
   unchanged. Enable it only if keeping the reference matters more than new detail.
4. Decode with ordinary **VAE Decode** when memory permits. Tiled VAE decoding
   can introduce a separate color difference after sampling.

Higher anchor strength keeps the reference more closely. A larger grid also
preserves finer reference structure, reducing freedom to redraw it. The anchor
is independent of LLLite strength: disable both to remove both constraints.
At anchor strength `0`, this node returns its input model unchanged.

For a small subject inside a large, multi-megapixel canvas, try grid size `16`
or `32`. A coarse grid can improve the background while missing a local color
shift. Larger grids do not resolve the texture tradeoff. Inspect native-size
texture crops as well as the overall image; color error and SSIM alone cannot
measure detail enhancement.

Strong style LoRAs can reduce color following even when the Tile controller
holds geometry well. For restoration, compare a model without the style LoRA
in the refinement stage; its appearance is already present in the reference.

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
