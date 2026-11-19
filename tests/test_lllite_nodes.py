import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.latent_formats
import comfy.ldm.moss_image.model as moss
import comfy.ops
import safetensors.torch
import torch
import torch.nn.functional as F
from torch import nn

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('moss_lllite_test', PACKAGE / '__init__.py', submodule_search_locations=[str(PACKAGE)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
nodes = sys.modules[spec.name + '.lllite_nodes']
architecture = sys.modules[spec.name + '.lllite_model']
color_guide = sys.modules[spec.name + '.color_guide']


def fixture(fused=False):
    backbone = moss.MOSSImage(dim=32, ffn_dim=64, num_heads=2, num_layers=2, text_dim=16,
                              fused_qkv=fused, operations=nn, dtype=torch.float32, device='cpu')
    base = nn.Module()
    base.diffusion_model = backbone
    base.model_sampling = comfy.model_sampling.ModelSamplingDiscreteFlow()
    model = comfy.model_patcher.ModelPatcher(base, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
    control = architecture.MossLLLiteControl(32, 2, rank=4, condition_dim=16, condition_embedding_dim=8)
    for block in control.blocks:
        for module in block.values():
            nn.init.normal_(module.up.weight, std=0.1)
            nn.init.zeros_(module.up.bias)
            nn.init.zeros_(module.depth_embed)
    holder = comfy.model_patcher.ModelPatcher(control, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
    return model, holder, torch.rand(1, 64, 64, 3)


class LinearColorVAE:
    latent_channels = 128
    def __init__(self):
        self.inputs = []
    def encode(self, pixels):
        self.inputs.append(pixels.clone())
        channels = F.pad(pixels.mean((1, 2)), (0, 29))
        return channels.repeat_interleave(4, dim=1)[:, :, None, None].expand(-1, -1, 16, 16)
    def decode(self, samples):
        b, _, h, w = samples.shape
        colors = samples.reshape(b, 32, 4, h, w).mean(2)[:, :3]
        return F.interpolate(colors, scale_factor=16, mode='nearest').movedim(1, -1)


class NodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_zero_and_empty_interval_return_original_model(self):
        model, control, image = fixture()
        self.assertIs(nodes.MossLLLiteApply.execute(model, control, image, 0).result[0], model)
        self.assertIs(nodes.MossLLLiteApply.execute(model, control, image, 1, .8, .2).result[0], model)

    def test_reference_anchor_zero_returns_original_without_adding_hooks(self):
        model, _, _ = fixture()
        before = model.model_options.copy()
        self.assertIs(nodes.MossReferenceAnchorApply.execute(model, {}, 0).result[0], model)
        self.assertIs(nodes.MossReferenceAnchorApply.execute(model, {}).result[0], model)
        self.assertEqual(model.model_options, before)

    def test_reference_anchor_corrects_constant_bias_and_preserves_fine_variation(self):
        reference = torch.zeros(1, 128, 16, 24)
        checkerboard = ((torch.arange(16)[:, None] + torch.arange(24)[None, :]) % 2).float() * 2 - 1
        detail = checkerboard[None, None].expand(2, 128, -1, -1) * .1
        bias = torch.linspace(-.5, .5, 128)[None, :, None, None]
        anchor = nodes.MossReferenceAnchor(reference, 1, 4)
        result = anchor({'denoised': detail + bias})
        torch.testing.assert_close(result, detail, atol=1e-7, rtol=1e-6)
        zero = nodes.MossReferenceAnchor(reference, 0, 4)
        self.assertIs(zero({'denoised': detail}), detail)

    def test_reference_anchor_clones_model_reference_and_preserves_existing_hooks(self):
        model, _, _ = fixture()
        model.model.latent_format = comfy.latent_formats.Flux2()
        previous = lambda args: args['denoised']
        model.set_model_sampler_post_cfg_function(previous)
        reference = torch.randn(1, 128, 16, 24)
        saved = reference.clone()
        result = nodes.MossReferenceAnchorApply.execute(model, {'samples': reference}, .5, 4).result[0]
        self.assertIsNot(result, model)
        self.assertEqual(len(model.model_options['sampler_post_cfg_function']), 1)
        hooks = result.model_options['sampler_post_cfg_function']
        self.assertEqual(len(hooks), 2)
        self.assertIs(hooks[0], previous)
        reference.fill_(100)
        self.assertTrue(torch.equal(hooks[-1].reference, saved))
        self.assertEqual(hooks[-1].reference.device.type, 'cpu')

    def test_reference_anchor_rejects_incompatible_latents_and_sampling_shapes(self):
        model, _, _ = fixture()
        model.model.latent_format = comfy.latent_formats.Flux2()
        with self.assertRaisesRegex(ValueError, '128-channel'):
            nodes.MossReferenceAnchorApply.execute(model, {'samples': torch.zeros(1, 16, 8, 8)}, .5)
        anchor = nodes.MossReferenceAnchor(torch.zeros(2, 128, 8, 8), .5, 4)
        with self.assertRaisesRegex(ValueError, 'sampling resolution'):
            anchor({'denoised': torch.zeros(2, 128, 8, 12)})
        with self.assertRaisesRegex(ValueError, 'per output image'):
            anchor({'denoised': torch.zeros(3, 128, 8, 8)})

    def test_color_guide_zero_skips_calibration_and_keeps_input_model(self):
        model, _, _ = fixture()
        self.assertIs(color_guide.MossColorGuideApply.execute(model).result[0], model)
        with self.assertRaisesRegex(ValueError, 'Connect a reference image'):
            color_guide.MossColorGuideApply.execute(model, strength=.5)

    def test_color_feedback_corrects_palette_and_preserves_other_latent_features(self):
        vae = LinearColorVAE()
        image = torch.full((1, 64, 64, 3), .5)
        directions = color_guide.calibrate_color_directions(vae, image)
        samples = torch.full((2, 32, 4, 4, 4), .2)
        samples[:, 3:] = .7
        samples += torch.tensor([-.03, .03, -.03, .03])[None, None, :, None, None]
        samples = samples.flatten(1, 2)
        guide = color_guide.MossColorGuide(vae, image, directions, 1, 4, 2)
        result = guide(lambda: samples)
        actual = result.reshape(2, 32, 4, 4, 4)
        torch.testing.assert_close(actual[:, :3].mean(2), torch.full((2, 3, 4, 4), .5), atol=.03, rtol=0)
        self.assertTrue(torch.equal(result[:, 12:], samples[:, 12:]))
        torch.testing.assert_close(actual[:, :, 0] - actual[:, :, 1], torch.full((2, 32, 4, 4), -.06), atol=1e-7, rtol=1e-6)
        weak = color_guide.MossColorGuide(vae, image, directions, .5, 4, 2).correct(samples)
        self.assertLess(float(weak[:, :12].mean()), float(result[:, :12].mean()))
        self.assertGreater(float(weak[:, :12].mean()), float(samples[:, :12].mean()))
        zero = color_guide.MossColorGuide(None, image, directions, 0, 4, 2)
        self.assertIs(zero.correct(samples), samples)

    def test_color_guide_calibrates_per_image_and_keeps_existing_hooks(self):
        model, _, _ = fixture()
        previous = lambda args: args['denoised']
        model.set_model_sampler_post_cfg_function(previous)
        vae = LinearColorVAE()
        image = torch.full((2, 64, 64, 3), .5)
        result = color_guide.MossColorGuideApply.execute(model, image, vae, .5, 2).result[0]
        self.assertIsNot(result, model)
        self.assertEqual(model.model_options['sampler_post_cfg_function'], [previous])
        self.assertEqual(result.model_options['sampler_post_cfg_function'], [previous])
        kind = nodes.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
        self.assertFalse(model.get_all_wrappers(kind))
        wrapper, = result.get_all_wrappers(kind)
        image.fill_(100)
        torch.testing.assert_close(wrapper.target, torch.full((2, 3, 2, 2), .5))
        self.assertEqual(wrapper.directions.device.type, 'cpu')
        torch.testing.assert_close(wrapper.directions.norm(dim=-2), torch.ones(2, 3))
        self.assertEqual(len(vae.inputs), 6)
        weights = torch.tensor([.2126, .7152, .0722])
        for pixels in vae.inputs[:4]:
            self.assertEqual(pixels.shape, (2, 256, 256, 3))
            torch.testing.assert_close(pixels @ weights, torch.full((2, 256, 256), .5))

    def test_color_guide_rejects_misaligned_images_and_sampling_shapes(self):
        model, _, _ = fixture()
        vae = SimpleNamespace(latent_channels=128)
        with self.assertRaisesRegex(ValueError, 'multiples of 16'):
            color_guide.MossColorGuideApply.execute(model, torch.zeros(1, 65, 64, 3), vae, .5)
        guide = color_guide.MossColorGuide(None, torch.zeros(2, 64, 64, 3), torch.zeros(2, 32, 3), .5, 4, 2)
        with self.assertRaisesRegex(ValueError, 'sampling resolution'):
            guide.correct(torch.zeros(2, 128, 4, 6))
        with self.assertRaisesRegex(ValueError, 'per output image'):
            guide.correct(torch.zeros(3, 128, 4, 4))

    def test_real_block_matches_original_at_zero_and_changes_at_nonzero(self):
        # Native CPU attention and differentiable norm path, without kitchen CUDA kernels.
        with patch.object(comfy.model_management, 'in_training', True):
            for fused in (False, True):
                model, control, image = fixture(fused)
                backbone = model.model.diffusion_model
                x = torch.randn(2, 19, 32)
                condition = control.model.conditioner(image.movedim(-1, 1) * 2 - 1).repeat(2, 1, 1)
                mod = tuple(torch.randn(2, 1, 32) * 0.1 for _ in range(6))
                pe = backbone.rope(4, 4, 3, torch.device('cpu'))
                block = backbone.layers[0]
                baseline = block(x.clone(), mod, pe)
                zero = architecture.controlled_moss_block(block, x.clone(), mod, pe, {}, control.model.blocks[0], condition, 0)
                active = architecture.controlled_moss_block(block, x.clone(), mod, pe, {}, control.model.blocks[0], condition, 1)
                torch.testing.assert_close(zero, baseline, atol=1e-6, rtol=1e-5)
                self.assertFalse(torch.equal(active, baseline))
                self.assertTrue(torch.isfinite(active).all())

    def test_wrapper_cfg_repeat_schedule_and_prompt_weights(self):
        model, control, image = fixture()
        patcher = nodes.MossLLLiteApply.execute(model, control, image).result[0]
        wrapper = patcher.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL)[0]
        observed = {}
        def executor(x, timestep, context, options, **kwargs):
            observed.update(options); return x
        x, context = torch.randn(2, 128, 4, 4), torch.randn(2, 3, 16)
        with patch.object(control.model.conditioner, 'forward', wraps=control.model.conditioner.forward) as encode:
            wrapper(executor, x, torch.tensor([.5]), context, {}, context_attention_bias=torch.zeros(2, 3))
        self.assertEqual(encode.call_args.args[0].shape[0], 2)
        condition, mask = observed[nodes.CONTROL_KEY][wrapper]
        self.assertEqual(condition.shape, (2, 16, 8))
        self.assertTrue(torch.equal(condition[0], condition[1]))
        self.assertEqual(mask.shape, (2, 1, 1, 19))
        observed.clear()
        wrapper.sigma_start, wrapper.sigma_end = .3, .1
        wrapper(executor, x, torch.tensor([.8]), context, {'sigmas': torch.tensor([.8])})
        self.assertNotIn(nodes.CONTROL_KEY, observed)

    def test_rgb_normalization_matches_training_before_bfloat16_cast(self):
        model, control, _ = fixture()
        image = torch.arange(256, dtype=torch.float32).view(1, 1, 256, 1).expand(1, 256, 256, 3) / 255
        patcher = nodes.MossLLLiteApply.execute(model, control, image).result[0]
        wrapper = patcher.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL)[0]
        x = torch.zeros(2, 128, 16, 16, dtype=torch.bfloat16)
        with patch.object(control.model.conditioner, 'forward', return_value=torch.zeros(2, 256, 8)) as encode:
            wrapper(lambda *args, **kwargs: x, x, torch.tensor([.5]), None, {})
        expected = (image.movedim(-1, 1) * 2 - 1).to(torch.bfloat16).repeat(2, 1, 1, 1)
        self.assertTrue(torch.equal(encode.call_args.args[0], expected))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for trainer AMP parity')
    def test_cuda_conditioning_and_adapter_match_training_arithmetic(self):
        _, holder, _ = fixture()
        training = holder.model.to(device='cuda', dtype=torch.bfloat16)
        control = architecture.MossLLLiteControl(
            **training.config, operations=comfy.ops.manual_cast, device='cuda', dtype=torch.bfloat16)
        control.load_state_dict(training.state_dict())
        holder = comfy.model_patcher.ModelPatcher(control, load_device=torch.device('cuda'),
                                                 offload_device=torch.device('cpu'))
        image = torch.rand(1, 64, 96, 3)
        wrapper = nodes.MossLLLitePatch(holder, image, 1, float('inf'), 0)
        observed = {}

        def executor(x, timestep, context, options, **kwargs):
            self.assertFalse(torch.is_autocast_enabled('cuda'))
            observed.update(options)
            return x

        x = torch.randn(2, 128, 4, 6, device='cuda', dtype=torch.bfloat16)
        wrapper(executor, x, torch.tensor([.5], device='cuda'), None, {})
        actual = observed[nodes.CONTROL_KEY][wrapper][0]
        pixels = (image.movedim(-1, 1) * 2 - 1).cuda().repeat(2, 1, 1, 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = training.conditioner(pixels)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        tokens = torch.randn(2, 27, 32, device='cuda', dtype=torch.bfloat16)
        for target in architecture.TARGETS:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                image_tokens = training.blocks[0][target](tokens[:, :24], expected, .25)
            reference = torch.cat((image_tokens, tokens[:, 24:]), dim=1)
            result = architecture.apply_image_control(control.blocks[0][target], tokens, actual, .25)
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
            self.assertTrue(torch.equal(result[:, 24:], tokens[:, 24:]))
        self.assertFalse(torch.is_autocast_enabled('cuda'))

    def test_fused_projection_rows_bias_and_gradients(self):
        projection = nn.Linear(32, 96)
        inputs = [torch.randn(2, 19, 32, requires_grad=True) for _ in range(3)]
        expected = tuple(projection(value).chunk(3, dim=-1)[i] for i, value in enumerate(inputs))
        actual = architecture.project_fused_qkv(projection, inputs)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)
        parameters = [*inputs, projection.weight, projection.bias]
        reference_grads = torch.autograd.grad(sum(value.square().sum() for value in expected), parameters)
        actual_grads = torch.autograd.grad(sum(value.square().sum() for value in actual), parameters)
        for result, reference in zip(actual_grads, reference_grads):
            torch.testing.assert_close(result, reference)

    def test_fused_projection_preserves_comfy_weight_and_bias_patches(self):
        projection = comfy.ops.manual_cast.Linear(32, 96)
        projection.load_state_dict(nn.Linear(32, 96).state_dict())
        delta = torch.randn_like(projection.weight) * .01
        projection.weight_function = [lambda weight: weight + delta]
        projection.bias_function = [lambda bias: bias + .1]
        inputs = [torch.randn(2, 19, 32) for _ in range(3)]
        expected = tuple(projection(value).chunk(3, dim=-1)[i] for i, value in enumerate(inputs))
        actual = architecture.project_fused_qkv(projection, inputs)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)

    def test_custom_projection_retains_its_forward(self):
        class CustomLinear(nn.Linear):
            def forward(self, value):
                return super().forward(value) * 2

        projection = CustomLinear(32, 96)
        inputs = [torch.randn(1, 19, 32) for _ in range(3)]
        self.assertIs(architecture.project_fused_qkv(projection, inputs), NotImplemented)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for native INT8 ConvRot')
    def test_fused_int8_convrot_rows_match_native_projection(self):
        projection = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(
            256, 768, bias=False, device='cuda')
        weight = architecture.QuantizedTensor.from_float(
            torch.randn(768, 256, device='cuda', dtype=torch.bfloat16),
            'TensorWiseINT8Layout', per_channel=True, convrot=True, convrot_groupsize=256)
        projection.weight = nn.Parameter(weight, requires_grad=False)
        projection.quant_format = 'int8_tensorwise'
        projection.layout_type = 'TensorWiseINT8Layout'
        inputs = [torch.randn(2, 32, 256, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        expected = tuple(projection(value).chunk(3, dim=-1)[i] for i, value in enumerate(inputs))
        actual = architecture.project_fused_qkv(projection, inputs)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
        delta = torch.randn(768, 256, device='cuda', dtype=torch.bfloat16) * .01
        projection.weight_function = [lambda value: value + delta]
        expected = tuple(projection(value).chunk(3, dim=-1)[i] for i, value in enumerate(inputs))
        actual = architecture.project_fused_qkv(projection, inputs)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)

    def test_cfg_condition_batch_order(self):
        model, control, image = fixture()
        image = image.repeat(2, 1, 1, 1)
        image[1] = 1 - image[1]
        wrapper = nodes.MossLLLitePatch(control, image, 1, float('inf'), 0)
        observed = {}
        def executor(x, timestep, context, options, **kwargs):
            observed.update(options); return x
        wrapper(executor, torch.randn(4, 128, 4, 4), torch.tensor([.5]), torch.randn(4, 3, 16), {})
        actual = observed[nodes.CONTROL_KEY][wrapper][0]
        pixels = image.movedim(-1, 1).repeat(2, 1, 1, 1) * 2 - 1
        reference = control.model.conditioner(pixels)
        torch.testing.assert_close(actual, reference)

    def test_loader_roundtrip(self):
        model, control, image = fixture()
        state = {architecture.PREFIX + k: v for k, v in control.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control.safetensors'
            safetensors.torch.save_file(state, path, metadata=control.model.metadata())
            with patch.object(nodes.folder_paths, 'get_full_path_or_raise', return_value=str(path)):
                loaded = nodes.MossLLLiteLoader.execute('control.safetensors').result[0]
            for name, value in control.model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.model.state_dict()[name]))

    def test_reject_concat_projection_and_reference_tokens(self):
        model, control, image = fixture()
        model.add_patches({'diffusion_model.video_embedder.weight': ('set', (torch.randn(32, 384),))})
        with self.assertRaises(ValueError): nodes.MossLLLiteApply.execute(model, control, image)
        wrapper = nodes.MossLLLitePatch(control, image, 1, float('inf'), 0)
        with self.assertRaises(ValueError):
            wrapper(lambda *a: None, torch.zeros(1, 128, 4, 4), torch.zeros(1), torch.zeros(1, 3, 16),
                    ref_latents=[torch.zeros(1, 128, 4, 4)])


if __name__ == '__main__':
    unittest.main()
