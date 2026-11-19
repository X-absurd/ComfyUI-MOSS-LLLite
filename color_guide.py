"""Final latent color correction measured through the supplied Flux2 VAE."""

import torch
import torch.nn.functional as F

import comfy.ldm.moss_image.model
import comfy.model_management
import comfy.patcher_extension
from comfy_api.latest import io


def calibrate_color_directions(vae, image):
    pixels = F.interpolate(image[..., :3].movedim(-1, 1).float(), size=(256, 256), mode='area').movedim(1, -1)
    axes = []
    for direction in ((.1, -.1 * .2126 / .7152, 0.), (0., -.1 * .0722 / .7152, .1), (.1, .1, .1)):
        shift = pixels.new_tensor(direction)
        plus = vae.encode((pixels + shift).clamp(0, 1)).float().cpu()
        minus = vae.encode((pixels - shift).clamp(0, 1)).float().cpu()
        delta = (plus - minus).mean((-2, -1)).reshape(image.shape[0], 32, 4).mean(2)
        axes.append(delta)
    axes = torch.stack(axes, dim=-1)
    return axes / axes.norm(dim=-2, keepdim=True).clamp_min(1e-6)


class MossColorGuide:
    def __init__(self, vae, image, directions, strength, grid_size, iterations):
        self.vae = vae
        self.image_shape = tuple(image.shape[:3])
        self.grid = (min(grid_size, image.shape[1] // 16), min(grid_size, image.shape[2] // 16))
        self.target = F.adaptive_avg_pool2d(image[..., :3].movedim(-1, 1).float(), self.grid).cpu()
        self.directions = directions.float().cpu().clone()
        self.strength, self.iterations = strength, iterations

    def move(self, latent, coefficients):
        delta = torch.einsum('bkhw,bck->bchw', coefficients, self.directions)
        # Apply the same correction to each of the four packed spatial phases.
        delta = delta[:, :, None].expand(-1, -1, 4, -1, -1).flatten(1, 2)
        return latent + F.interpolate(delta, size=latent.shape[-2:], mode='bilinear', align_corners=False)

    def decode_colors(self, latent):
        comfy.model_management.throw_exception_if_processing_interrupted()
        image = self.vae.decode(latent).float().cpu().clamp(0, 1)
        return F.adaptive_avg_pool2d(image.movedim(-1, 1), self.grid)

    def correct(self, samples):
        if self.strength == 0:
            return samples
        if samples.ndim != 4 or samples.shape[1] != 128 or (samples.shape[-2] * 16, samples.shape[-1] * 16) != self.image_shape[1:]:
            raise ValueError('MOSS color guide needs its image at the sampling resolution')
        if self.image_shape[0] not in (1, samples.shape[0]):
            raise ValueError('MOSS color guide needs one reference image or one per output image')
        latent = samples.float().cpu()
        current = self.decode_colors(latent)
        target = self.target if self.strength == 1 else torch.lerp(current, self.target, self.strength)
        identity = torch.eye(3)
        for _ in range(self.iterations):
            derivatives = []
            for index in range(3):
                coefficients = latent.new_zeros((samples.shape[0], 3, *self.grid))
                coefficients[:, index] = .5
                colors = self.decode_colors(self.move(latent, coefficients))
                derivatives.append((colors - current) / .5)
            jacobian = torch.stack(derivatives, dim=-1).movedim(1, -2)
            error = (target - current).movedim(1, -1)[..., None]
            normal = jacobian.transpose(-1, -2) @ jacobian
            ridge = normal.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * .02 + 1e-6
            coefficients = torch.linalg.solve(normal + identity * ridge, jacobian.transpose(-1, -2) @ error).squeeze(-1)
            coefficients *= 2 / coefficients.norm(dim=-1, keepdim=True).clamp_min(2)
            latent = self.move(latent, coefficients.movedim(-1, 1) * .8)
            current = self.decode_colors(latent)
        return latent.to(device=samples.device, dtype=samples.dtype)

    def __call__(self, executor, *args, **kwargs):
        return self.correct(executor(*args, **kwargs))


class MossColorGuideApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MossColorGuideApply', display_name='Apply MOSS Color Guide', category='MOSS-LLLite',
            description='Experimental final latent color correction using actual VAE-decoded colors. '
                        'Keeps generated texture better than Reference Anchor in local tests. '
                        'May reduce local contrast and perceived sharpness. '
                        'Adds multiple full VAE decodes. Use the same Flux2 VAE and final-resolution reference image. '
                        'This is inference compensation, not a retrained LLLite; independent of LLLite strength.',
            inputs=[io.Model.Input('model'), io.Image.Input('image', optional=True), io.Vae.Input('vae', optional=True),
                    io.Float.Input('strength', default=0.0, min=0.0, max=1.0, step=0.05,
                                   tooltip='Color correction amount, independent of LLLite. 0 disables it; try 0.8 if 1 looks softer.'),
                    io.Int.Input('grid_size', default=8, min=1, max=32, step=1, advanced=True,
                                 tooltip='Coarse color measurement grid, not image tiles. Try 16 for local color shifts on large canvases.'),
                    io.Int.Input('iterations', default=2, min=1, max=3, step=1, advanced=True,
                                 tooltip='Decoder-feedback correction rounds. More rounds add full VAE decodes; start with 2.')],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, image=None, vae=None, strength=0.0, grid_size=8, iterations=2):
        if strength == 0:
            return io.NodeOutput(model)
        if not 0 < strength <= 1 or grid_size < 1 or not 1 <= iterations <= 3:
            raise ValueError('MOSS color guide needs strength in (0, 1], a positive grid size and 1-3 iterations')
        if not isinstance(model.get_model_object('diffusion_model'), comfy.ldm.moss_image.model.MOSSImage):
            raise ValueError('MOSS color guide requires a MOSS-Image backbone')
        if image is None or vae is None:
            raise ValueError('Connect a reference image and the Flux2 VAE to enable MOSS color guide')
        if vae.latent_channels != 128:
            raise ValueError('MOSS color guide requires the 128-channel Flux2 VAE')
        if image.ndim != 4 or image.shape[-1] < 3 or min(image.shape[1:3]) < 16 or any(size % 16 for size in image.shape[1:3]):
            raise ValueError('MOSS color guide needs an RGB image at the sampling resolution, in multiples of 16')
        directions = calibrate_color_directions(vae, image)
        result = model.clone()
        result.add_wrapper(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                           MossColorGuide(vae, image, directions, strength, grid_size, iterations))
        return io.NodeOutput(result)
