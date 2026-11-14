import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.ldm.moss_image.model as moss
import comfy.ops
import safetensors.torch
import torch
from torch import nn

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('moss_lllite_test', PACKAGE / '__init__.py', submodule_search_locations=[str(PACKAGE)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
nodes = sys.modules[spec.name + '.lllite_nodes']
architecture = sys.modules[spec.name + '.lllite_model']


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


class NodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_zero_and_empty_interval_return_original_model(self):
        model, control, image = fixture()
        self.assertIs(nodes.MossLLLiteApply.execute(model, control, image, 0).result[0], model)
        self.assertIs(nodes.MossLLLiteApply.execute(model, control, image, 1, .8, .2).result[0], model)

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
        wrapper(executor, x, torch.tensor([.5]), context, {}, context_attention_bias=torch.zeros(2, 3))
        condition, mask = observed[nodes.CONTROL_KEY][wrapper]
        self.assertEqual(condition.shape, (2, 16, 8))
        self.assertTrue(torch.equal(condition[0], condition[1]))
        self.assertEqual(mask.shape, (2, 1, 1, 19))
        observed.clear()
        wrapper.sigma_start, wrapper.sigma_end = .3, .1
        wrapper(executor, x, torch.tensor([.8]), context, {'sigmas': torch.tensor([.8])})
        self.assertNotIn(nodes.CONTROL_KEY, observed)

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
